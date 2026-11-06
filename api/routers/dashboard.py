"""Read endpoints consumed by the dashboard."""

import json
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from statsmodels.tsa.statespace.sarimax import SARIMAX

from api import schemas
from api.deps import Filters, customers, models, query
from src.utils import load, load_config

router = APIRouter(tags=["dashboard"])
SORTABLE = Literal[
    "revenue_at_risk",
    "churn_prob",
    "clv_pred",
    "recency_days",
    "frequency",
    "monetary_total",
    "next_purchase_prob",
    "customer_id",
]


def _delta(now: float, before: float | None) -> schemas.Delta:
    return schemas.Delta(value=float(now), previous=None if before is None or pd.isna(before) else float(before))


@router.get("/kpis", response_model=schemas.Kpis)
def kpis(filters: Filters = Depends()) -> schemas.Kpis:
    """Headline KPIs vs the previous 90-day period, plus the churn-risk distribution."""
    df, thr = customers(filters), models()["card"]["threshold"]
    prev = df[df["prev_churn_prob"].notna()]
    edges = np.linspace(0, 1, 11)
    bins = pd.cut(df["churn_prob"].clip(0, 1), edges, include_lowest=True, labels=edges[:-1])
    dist = df.groupby(bins, observed=False)["revenue_at_risk"].agg(["size", "sum"])
    return schemas.Kpis(
        customers=len(df),
        active_customers=_delta((df["recency_days"] <= 90).sum(), (prev["prev_recency_days"] <= 90).sum()),
        at_risk_customers=_delta((df["churn_prob"] >= thr).sum(), (prev["prev_churn_prob"] >= thr).sum()),
        revenue_at_risk=_delta(df["revenue_at_risk"].sum(), (prev["prev_churn_prob"] * prev["prev_clv_pred"]).sum()),
        avg_clv=_delta(df["clv_pred"].mean() if len(df) else 0, prev["prev_clv_pred"].mean() if len(prev) else None),
        threshold=thr,
        risk_distribution=[
            schemas.Bin(
                bucket=f"{lo * 100:.0f}–{lo * 100 + 10:.0f}%",
                low=lo,
                customers=int(r["size"]),
                revenue_at_risk=float(r["sum"]),
            )
            for lo, r in dist.iterrows()
        ],
        risk_bands={k: int(v) for k, v in df["risk_band"].value_counts().items()},
    )


def forecast(series: pd.Series, horizon: int = 3) -> pd.DataFrame:
    """Seasonal ARIMA forecast with a 80% confidence band (falls back to a flat forecast on short series)."""
    index = pd.period_range(pd.Period(series.index[-1], "M") + 1, periods=horizon, freq="M").strftime("%Y-%m")
    if len(series) < 15:
        return pd.DataFrame({"forecast": series.iloc[-3:].mean(), "lower": np.nan, "upper": np.nan}, index=index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(
            np.log(series.clip(lower=1).to_numpy()), order=(1, 0, 0), seasonal_order=(0, 1, 0, 12), trend="c"
        ).fit(disp=False)
    pred = fit.get_forecast(horizon)
    ci = np.exp(pred.conf_int(alpha=0.2))
    return pd.DataFrame({"forecast": np.exp(pred.predicted_mean), "lower": ci[:, 0], "upper": ci[:, 1]}, index=index)


@router.get("/trends", response_model=list[schemas.TrendPoint])
def trends(filters: Filters = Depends()) -> list[schemas.TrendPoint]:
    """Monthly net revenue for the filtered customers with a 3-month forecast."""
    ids = (
        customers(filters)["customer_id"]
        if any([filters.segment, filters.tier, filters.date_from, filters.date_to])
        else None
    )
    sql = "SELECT month, customer_id, SUM(revenue) AS revenue FROM transactions WHERE is_admin = :f"
    params = {"f": False}
    if filters.country:
        sql += " AND country IN (" + ",".join(f":c{i}" for i in range(len(filters.country))) + ")"
        params |= {f"c{i}": c for i, c in enumerate(filters.country)}
    df = query(sql + " GROUP BY month, customer_id", **params)
    if ids is not None:
        df = df[df["customer_id"].isin(ids)]
    monthly = df.groupby("month").agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique")).iloc[:-1]
    if monthly.empty:
        return []
    fc = forecast(monthly["revenue"])
    fc.loc[monthly.index[-1]] = [monthly["revenue"].iat[-1]] * 3
    out = monthly.join(fc, how="outer").sort_index()
    return [
        schemas.TrendPoint(month=m, **{k: None if pd.isna(v) else float(v) for k, v in r.items()})
        for m, r in out.iterrows()
    ]


@router.get("/customers", response_model=schemas.CustomerPage)
def list_customers(
    filters: Filters = Depends(),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    sort_by: SORTABLE = "revenue_at_risk",
    order: Literal["asc", "desc"] = "desc",
    risk_band: list[str] | None = Query(None),
    search: str | None = None,
    anomalies_only: bool = False,
) -> schemas.CustomerPage:
    """Paginated, filterable, sortable customer leaderboard (default: highest revenue at risk first)."""
    df = customers(filters)
    if risk_band:
        df = df[df["risk_band"].isin(risk_band)]
    if search:
        df = df[df["customer_id"].astype(str).str.startswith(search.strip())]
    if anomalies_only:
        df = df[df["is_anomaly"].astype(bool)]
    df = df.sort_values(sort_by, ascending=order == "asc")
    rows = df.iloc[(page - 1) * page_size : page * page_size]
    return schemas.CustomerPage(items=rows.to_dict("records"), total=len(df), page=page, page_size=page_size)


@router.get("/customers/{customer_id}", response_model=schemas.CustomerProfile)
def customer(customer_id: int) -> schemas.CustomerProfile:
    """Full profile: scores, SHAP drivers, plain-language explanation, purchase timeline and recommendations."""
    df = customers()
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {customer_id} was not found among scored customers.")
    r = row.iloc[0].to_dict()
    orders = query(
        "SELECT invoice, MIN(invoice_date) AS date, SUM(revenue) AS value, SUM(quantity) AS items, "
        "MAX(CASE WHEN is_return THEN 1 ELSE 0 END) AS is_return FROM transactions "
        "WHERE customer_id = :c AND is_admin = :f GROUP BY invoice ORDER BY date",
        c=customer_id,
        f=False,
    )
    cats = {k[4:].replace("_", " "): float(v) for k, v in r.items() if k.startswith("cat_") and v > 0}
    r = {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in r.items()}
    r |= {"contributions": json.loads(r["shap"]), "recommendations": json.loads(r["recommendations"])}
    return schemas.CustomerProfile(
        **r,
        timeline=orders.assign(
            is_return=orders["is_return"].astype(bool),
            items=orders["items"].abs(),
            date=pd.to_datetime(orders["date"]).dt.date,
        ).to_dict("records"),
        category_mix=dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        base_value=models()["explainer"].base_value,
        shap_link=models()["explainer"].link,
        segment_action=load_config()["segmentation"]["actions"].get(r["segment"], ""),
    )


@router.get("/customers/{customer_id}/lime", response_model=schemas.LimeComparison)
def customer_lime(customer_id: int) -> schemas.LimeComparison:
    """LIME local explanation for one customer, compared with its SHAP explanation."""
    df = customers()
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(404, f"Customer {customer_id} was not found among scored customers.")
    b = models()
    weights = b["explainer"].lime_weights(row.iloc[0], load_config()["lime_samples"])
    shap = json.loads(row.iloc[0]["shap"])
    top_lime, top_shap = [w["feature"] for w in weights[:5]], [c["feature"] for c in shap[:5]]
    shared = [f for f in top_lime if f in top_shap]
    return schemas.LimeComparison(lime=weights, shap=shap, shared_top_features=shared, agreement=len(shared) / 5)


@router.get("/segments", response_model=schemas.Segments)
def segments(filters: Filters = Depends()) -> schemas.Segments:
    """Segment profiles with recommended actions and a 2-D PCA scatter (sampled)."""
    df = customers(filters)
    seg = query("SELECT segment, action FROM segments").set_index("segment")["action"]
    prof = df.groupby("segment").agg(
        size=("customer_id", "size"),
        avg_spend=("monetary_total", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        churn_rate=("churn_prob", "mean"),
        avg_clv=("clv_pred", "mean"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        q4_share=("q4_share", "mean"),
        revenue_at_risk=("revenue_at_risk", "sum"),
    )
    prof["action"] = seg.reindex(prof.index).fillna("")
    sample = df.sample(min(len(df), 2500), random_state=0)
    return schemas.Segments(
        segments=prof.sort_values("revenue_at_risk", ascending=False).reset_index().to_dict("records"),
        scatter=sample.rename(columns={"pca_x": "x", "pca_y": "y"})[
            ["customer_id", "x", "y", "segment", "churn_prob"]
        ].to_dict("records"),
    )


@router.get("/models")
def model_metadata() -> dict:
    """Model comparison, ROC curve, confusion matrix, CLV and segmentation diagnostics."""
    card = models()["card"]
    return {**card, "segmentation": load("segmentation_report.json")}


@router.get("/filters", response_model=schemas.FilterOptions)
def filter_options() -> schemas.FilterOptions:
    """Values available for the global filter bar."""
    df = customers()
    dates = pd.to_datetime(df["last_purchase"])
    return schemas.FilterOptions(
        countries=df["country"].value_counts().index.tolist(),
        segments=sorted(df["segment"].unique()),
        tiers=["High", "Mid", "Low"],
        min_date=dates.min().date(),
        max_date=dates.max().date(),
        scored_at=pd.to_datetime(df["scored_at"]).max().date(),
    )
