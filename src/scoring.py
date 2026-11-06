"""Batch scoring with every trained artifact, and loading results into the predictions database."""

import json
import os
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.explainability.explain import ChurnExplainer, explain_sentence, top_contributions
from src.features.build import build_features, build_sequences, history
from src.models.deep import SPEND_FEATURES, reconstruction_error
from src.segmentation.cluster import assign
from src.utils import get_logger, load, load_config

log = get_logger(__name__)
PREDICTION_COLUMNS = {
    "churn_prob",
    "clv_pred",
    "next_purchase_prob",
    "anomaly_score",
    "is_anomaly",
    "segment_id",
    "segment",
    "pca_x",
    "pca_y",
    "revenue_at_risk",
    "risk_band",
    "shap",
    "explanation",
    "top_driver",
    "recommendations",
    "value_tier",
}


@cache
def bundle(artifacts: str) -> dict[str, Any]:
    """Load (once per artifacts directory) every model needed for scoring."""
    cfg = {**load_config(), "paths": {**load_config()["paths"], "artifacts": artifacts}}
    card = load("model_card.json", cfg)
    churn = load("churn_model.joblib", cfg)
    return {
        "card": card,
        "churn": churn,
        "clv": load("clv_model.joblib", cfg),
        "lstm": load("lstm_model.joblib", cfg),
        "autoencoder": load("autoencoder.joblib", cfg),
        "segments": load("segments.joblib", cfg),
        "skus": load("sku_stats.joblib", cfg),
        "reference": load("engagement_reference.joblib", cfg),
        "explainer": ChurnExplainer(churn, load("background.joblib", cfg), load_config()["seed"]),
    }


def engine(url: str | None = None) -> Engine:
    """SQLAlchemy engine for DATABASE_URL (SQLite file fallback for local runs)."""
    return create_engine(url or os.getenv("DATABASE_URL", "sqlite:///vantara.db"), pool_pre_ping=True)


def recommend(tx: pd.DataFrame, cutoff: pd.Timestamp, features: pd.DataFrame, n: int = 5) -> pd.Series:
    """Best-selling recent products in each customer's two favourite categories that they have not bought yet."""
    hist = history(tx, cutoff)
    sales = hist[~hist["is_return"]]
    recent = sales[sales["invoice_date"] >= cutoff - pd.Timedelta(days=90)]
    top = (
        recent.groupby(["category", "stock_code"])
        .agg(description=("description", "first"), buyers=("customer_id", "nunique"), price=("price", "median"))
        .reset_index()
        .sort_values("buyers", ascending=False)
        .groupby("category")
        .head(25)
    )
    by_cat = {c: g.to_dict("records") for c, g in top.groupby("category")}
    bought = sales.groupby("customer_id")["stock_code"].agg(set)
    cat_cols = [c for c in features.columns if c.startswith("cat_")]
    fav = features[cat_cols].to_numpy().argsort(axis=1)[:, ::-1][:, :2]
    names = [c[4:].replace("_", " ") for c in cat_cols]
    out = {}
    for cid, picks in zip(features.index, fav, strict=True):
        seen = bought.get(cid, set())
        pool = [p for i in picks for p in by_cat.get(names[i], []) if p["stock_code"] not in seen]
        out[cid] = [
            {
                "stock_code": p["stock_code"],
                "description": p["description"].title(),
                "category": p["category"],
                "price": round(float(p["price"]), 2),
            }
            for p in pool[:n]
        ]
    return pd.Series(out)


def score_features(
    features: pd.DataFrame, sequences: np.ndarray | None, cfg: dict[str, Any], explain: bool = True
) -> pd.DataFrame:
    """Churn, CLV, next-purchase probability, anomaly flag, segment and SHAP explanation per customer."""
    b = bundle(cfg["paths"]["artifacts"])
    cols, bands = b["card"]["features"], cfg["risk_bands"]
    X = features[cols]
    prep, ae = b["autoencoder"]
    err = reconstruction_error(ae, prep.transform(X[SPEND_FEATURES]))
    out = pd.DataFrame(
        {
            "churn_prob": b["churn"].predict_proba(X)[:, 1],
            "clv_pred": np.clip(b["clv"].predict(X), 0, None),
            "next_purchase_prob": np.nan if sequences is None else b["lstm"].predict_proba(sequences)[:, 0],
            "anomaly_score": err,
            "is_anomaly": err > b["card"]["anomaly"]["threshold"],
        },
        index=features.index,
    ).join(assign(features, b["segments"], cfg))
    out["revenue_at_risk"] = out["churn_prob"] * out["clv_pred"]
    out["risk_band"] = pd.cut(
        out["churn_prob"], [-0.01, bands["medium"], bands["high"], 1.01], labels=["Low", "Medium", "High"]
    ).astype(str)
    if explain:
        values = b["explainer"].shap_values(X)
        contribs = [top_contributions(row, X.iloc[i]) for i, row in enumerate(values)]
        out["shap"] = [json.dumps(c) for c in contribs]
        out["explanation"] = [explain_sentence(p, c, bands) for p, c in zip(out["churn_prob"], contribs, strict=True)]
        out["top_driver"] = [next((c["label"] for c in cs if c["shap"] > 0), cs[0]["label"]) for cs in contribs]
    return out


def score_customers(
    tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict[str, Any], explain: bool = True, batch: bool = False
) -> pd.DataFrame:
    """Build features at ``cutoff`` and score them; returns features joined with predictions.

    ``batch=True`` ranks engagement against the full customer base instead of the uploaded customers only.
    """
    b = bundle(cfg["paths"]["artifacts"])
    features = build_features(tx, cutoff, cfg, skus=b["skus"], reference=b["reference"] if batch else None)
    scored = features.join(score_features(features, build_sequences(tx, cutoff, features.index, cfg), cfg, explain))
    if explain:
        scored["recommendations"] = recommend(tx, cutoff, features).map(json.dumps)
    return scored


def publish(
    tx: pd.DataFrame, current: pd.DataFrame, previous: pd.DataFrame, cfg: dict[str, Any], url: str | None = None
) -> None:
    """Write customers, transactions, predictions and segments tables to the database."""
    tiers = cfg["value_tiers"]
    pct = current["clv_pred"].rank(pct=True)
    current["value_tier"] = np.select([pct >= tiers["high"], pct >= tiers["mid"]], ["High", "Mid"], "Low")
    prev = previous[["churn_prob", "clv_pred", "recency_days"]].add_prefix("prev_")
    current = current.join(prev)
    current["last_purchase"] = (current["snapshot"] - pd.to_timedelta(current["recency_days"], unit="D")).dt.date
    current["first_purchase"] = (current["snapshot"] - pd.to_timedelta(current["tenure_days"], unit="D")).dt.date
    lines = tx[tx["customer_id"].notna()][
        [
            "customer_id",
            "invoice",
            "invoice_date",
            "month",
            "stock_code",
            "description",
            "category",
            "quantity",
            "price",
            "revenue",
            "is_return",
            "is_admin",
            "country",
        ]
    ]
    seg = current.groupby("segment").agg(
        size=("churn_prob", "size"),
        avg_spend=("monetary_total", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        churn_rate=("churn_prob", "mean"),
        avg_clv=("clv_pred", "mean"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        q4_share=("q4_share", "mean"),
        revenue_at_risk=("revenue_at_risk", "sum"),
    )
    seg["action"] = seg.index.map(cfg["segmentation"]["actions"])
    pred_cols = [c for c in current.columns if c in PREDICTION_COLUMNS or c.startswith("prev_")]
    current["scored_at"] = current["snapshot"].dt.date
    tables = {
        "customers": current.drop(columns=[*pred_cols, "snapshot"]),
        "predictions": current[[*pred_cols, "scored_at"]],
        "segments": seg,
        "transactions": lines.set_index("customer_id"),
    }
    db = engine(url)
    with db.begin() as conn:
        for table in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
    for table, df in tables.items():
        df.reset_index().to_sql(table, db, index=False, chunksize=30000 // (len(df.columns) + 1), method="multi")
    with db.begin() as conn:
        conn.execute(text("CREATE INDEX ix_tx_customer ON transactions (customer_id)"))
        conn.execute(text("CREATE UNIQUE INDEX ix_customers_id ON customers (customer_id)"))
        conn.execute(text("CREATE UNIQUE INDEX ix_predictions_id ON predictions (customer_id)"))
    log.info("published to database", extra={"customers": len(current), "transactions": len(lines)})
