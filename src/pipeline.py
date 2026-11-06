"""Single-command pipeline: raw workbook -> clean data -> features -> models -> segments -> XAI -> scored database.

Usage: ``python -m src.pipeline`` (all stages) or ``python -m src.pipeline --stages data features``.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data.clean import clean, load_raw, standardize
from src.data.validate import validate
from src.explainability.explain import ChurnExplainer, label, pdp
from src.features.build import (
    build_features,
    build_labels,
    build_sequences,
    data_end,
    engagement_reference,
    feature_columns,
    sku_stats,
)
from src.models import classical, deep
from src.models.evaluate import classification_report, log_run, pick_threshold, regression_report, roc_points
from src.segmentation.cluster import fit_segments
from src.utils import get_logger, load_config, save, seed_everything

log = get_logger(__name__)
STAGES = ["data", "features", "train", "segment", "score"]


def _p(cfg: dict[str, Any], name: str) -> Path:
    path = Path(cfg["paths"]["processed"]) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def stage_data(cfg: dict[str, Any], raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load, clean and validate transactions; write the interim parquet."""
    tx = validate(clean(load_raw(cfg) if raw is None else standardize(raw), cfg), cfg)
    Path(cfg["paths"]["interim"]).parent.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(cfg["paths"]["interim"])
    return tx


def snapshot(tx: pd.DataFrame, cfg: dict[str, Any], horizon: int) -> pd.DataFrame:
    """Features at ``end - horizon`` joined with the labels observed over the following horizon."""
    cutoff = data_end(tx) - pd.Timedelta(days=horizon)
    f = build_features(tx, cutoff, cfg)
    return f.join(build_labels(tx, cutoff, horizon, f.index))


def stage_features(cfg: dict[str, Any]) -> None:
    """Build the churn and CLV training snapshots and the current scoring snapshot."""
    tx = pd.read_parquet(cfg["paths"]["interim"])
    snapshot(tx, cfg, cfg["labels"]["churn_horizon_days"]).to_parquet(_p(cfg, "churn_snapshot.parquet"))
    snapshot(tx, cfg, cfg["labels"]["clv_horizon_days"]).to_parquet(_p(cfg, "clv_snapshot.parquet"))
    current = build_features(tx, data_end(tx), cfg)
    current.to_parquet(_p(cfg, "features.parquet"))
    save(engagement_reference(current), "engagement_reference.joblib", cfg)
    save(sku_stats(tx[tx["customer_id"].notna() & ~tx["is_admin"]]), "sku_stats.joblib", cfg)
    log.info("features built")


def _split(df: pd.DataFrame, cfg: dict[str, Any], stratify: str | None) -> tuple[pd.DataFrame, ...]:
    tr, va, te = cfg["models"]["split"]
    strat = df[stratify] if stratify else None
    train, rest = train_test_split(df, train_size=tr, stratify=strat, random_state=cfg["seed"])
    val, test = train_test_split(
        rest, test_size=te / (va + te), stratify=rest[stratify] if stratify else None, random_state=cfg["seed"]
    )
    return train, val, test


def stage_train(cfg: dict[str, Any]) -> None:
    """Train, tune, compare and persist every churn, CLV and deep-learning model."""
    seed_everything(cfg["seed"])
    tx = pd.read_parquet(cfg["paths"]["interim"])
    data = pd.read_parquet(_p(cfg, "churn_snapshot.parquet"))
    cols = feature_columns(data.drop(columns=["churn", "future_revenue"]))
    train, val, test = _split(data, cfg, "churn")
    Xtr, Xva, Xte = train[cols], val[cols], test[cols]
    ytr, yva, yte = train["churn"], val["churn"], test["churn"]
    min_recall = cfg["models"]["min_recall"]
    results, models, probas = {}, {}, {}

    def record(name: str, family: str, p_val: np.ndarray, p_test: np.ndarray, info: dict[str, Any]) -> None:
        thr = pick_threshold(yva.to_numpy(), p_val, min_recall)
        results[name] = {
            "family": family,
            "val_roc_auc": classification_report(yva, p_val, thr)["roc_auc"],
            **{k: v for k, v in info.items() if k != "seconds"},
            **classification_report(yte, p_test, thr),
        }
        probas[name] = p_test
        log_run("churn", name, info.get("params", {}), results[name], info.get("seconds", 0), cfg)

    pos_weight = float((1 - ytr.mean()) / ytr.mean())
    for name, model in classical.churn_models(cfg["seed"], pos_weight).items():
        best, info = classical.tune(name, model, Xtr, ytr, cfg)
        if name == "xgboost":
            best = classical.refit_xgboost_early_stopping(best, Xtr, ytr, Xva, yva)
            info["params"]["best_iteration"] = int(best.best_iteration)
        models[name] = best
        record(name, "classical", best.predict_proba(Xva)[:, 1], best.predict_proba(Xte)[:, 1], info)

    scaler = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), FunctionTransformer(deep.to_float32))
    scaler.fit(Xtr)
    S = {k: scaler.transform(v) for k, v in [("tr", Xtr), ("va", Xva), ("te", Xte)]}
    start = time.perf_counter()
    ann = deep.train_classifier(deep.ANN, S["tr"], ytr.to_numpy(), S["va"], yva.to_numpy(), cfg)
    record(
        "ann",
        "deep",
        ann.predict_proba(S["va"])[:, 1],
        ann.predict_proba(S["te"])[:, 1],
        {"params": {"epochs": len(ann.history)}, "seconds": time.perf_counter() - start},
    )

    seqs = build_sequences(tx, data["snapshot"].iat[0], data.index, cfg)
    pos = pd.Series(np.arange(len(data)), index=data.index)
    Q = {k: seqs[pos[d.index]] for k, d in [("tr", train), ("va", val), ("te", test)]}
    start = time.perf_counter()
    lstm = deep.train_classifier(deep.SequenceLSTM, Q["tr"], ytr.to_numpy(), Q["va"], yva.to_numpy(), cfg)
    record(
        "lstm",
        "deep",
        lstm.predict_proba(Q["va"])[:, 1],
        lstm.predict_proba(Q["te"])[:, 1],
        {"params": {"epochs": len(lstm.history)}, "seconds": time.perf_counter() - start},
    )

    production = max(classical.TREE_MODELS, key=lambda n: results[n]["val_roc_auc"])
    model, threshold = models[production], results[production]["threshold"]
    explainer = ChurnExplainer(model, Xtr, cfg["seed"])
    importance = np.abs(explainer.shap_values(Xtr.sample(min(1000, len(Xtr)), random_state=cfg["seed"]))).mean(0)
    top = [cols[i] for i in np.argsort(-importance)]
    errors = {a: (probas[a] >= results[a]["threshold"]) != yte.to_numpy() for a in probas}

    ae_prep = make_pipeline(
        FunctionTransformer(deep.log1p_pos),
        SimpleImputer(strategy="median"),
        StandardScaler(),
        FunctionTransformer(deep.to_float32),
    )
    ae_prep.fit(Xtr[deep.SPEND_FEATURES])
    A = {k: ae_prep.transform(v[deep.SPEND_FEATURES]) for k, v in [("tr", Xtr), ("va", Xva), ("te", Xte)]}
    ae = deep.train_autoencoder(A["tr"], A["va"], cfg)
    ae_threshold = float(np.quantile(deep.reconstruction_error(ae, A["va"]), cfg["models"]["deep"]["anomaly_quantile"]))
    flagged = deep.reconstruction_error(ae, A["te"]) > ae_threshold
    anomaly = {
        "threshold": ae_threshold,
        "test_flag_rate": float(flagged.mean()),
        "flagged_median_spend": float(Xte["monetary_total"][flagged].median()) if flagged.any() else None,
        "overall_median_spend": float(Xte["monetary_total"].median()),
        "loss_curve": deep.loss_curve(ae),
    }
    log_run("anomaly", "autoencoder", {"epochs": len(ae.history)}, anomaly, 0, cfg)

    clv = pd.read_parquet(_p(cfg, "clv_snapshot.parquet"))
    ctr, cva, cte = _split(clv, cfg, None)
    clv_model, clv_name, clv_val = classical.train_clv(
        ctr[cols], ctr["future_revenue"], cva[cols], cva["future_revenue"], cfg
    )
    clv_test = regression_report(cte["future_revenue"].to_numpy(), clv_model.predict(cte[cols]))

    for name, obj in {
        "churn_model.joblib": model,
        "ann_model.joblib": (scaler, ann),
        "lstm_model.joblib": lstm,
        "autoencoder.joblib": (ae_prep, ae),
        "clv_model.joblib": clv_model,
        "background.joblib": Xtr.sample(min(2000, len(Xtr)), random_state=cfg["seed"]),
    }.items():
        save(obj, name, cfg)
    recent = Xte["recency_days"] <= 365
    ranked = sorted(results, key=lambda n: (results[n]["roc_auc"], results[n]["recall"]), reverse=True)
    save(
        {
            "production_model": production,
            "threshold": threshold,
            "features": cols,
            "feature_labels": {c: label(c) for c in cols},
            "medians": Xtr.median().fillna(0).to_dict(),
            "snapshot": str(data["snapshot"].iat[0].date()),
            "churn_rate": float(data["churn"].mean()),
            "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
            "comparison": [{"model": n, **results[n]} for n in ranked],
            "roc_curve": roc_points(yte.to_numpy(), probas[production]),
            "recent_customers_roc_auc": classification_report(
                yte[recent], probas[production][recent.to_numpy()], threshold
            )["roc_auc"],
            "global_importance": [
                {"feature": c, "label": label(c), "importance": float(importance[cols.index(c)])} for c in top[:15]
            ],
            "pdp": pdp(model, Xtr, top[:3]),
            "error_correlation": {
                f"{a}|{b}": float(np.corrcoef(errors[a], errors[b])[0, 1]) for a in errors for b in errors if a < b
            },
            "loss_curves": {"ann": deep.loss_curve(ann), "lstm": deep.loss_curve(lstm)},
            "clv": {
                "model": clv_name,
                "validation": clv_val,
                "test": clv_test,
                "horizon_days": cfg["labels"]["clv_horizon_days"],
            },
            "anomaly": anomaly,
        },
        "model_card.json",
        cfg,
    )
    log.info(
        "training complete", extra={"production_model": production, "test_roc_auc": results[production]["roc_auc"]}
    )


def stage_segment(cfg: dict[str, Any]) -> None:
    """Fit and persist the segmentation bundle on the current customer base."""
    bundle = fit_segments(pd.read_parquet(_p(cfg, "features.parquet")), cfg)
    save(bundle, "segments.joblib", cfg)
    save(bundle["report"], "segmentation_report.json", cfg)


def stage_score(cfg: dict[str, Any]) -> None:
    """Score the current customer base (and the previous period) and load everything into the database."""
    from src.scoring import publish, score_customers

    tx = pd.read_parquet(cfg["paths"]["interim"])
    current = score_customers(tx, data_end(tx), cfg)
    previous = score_customers(
        tx, data_end(tx) - pd.Timedelta(days=cfg["labels"]["churn_horizon_days"]), cfg, explain=False
    )
    publish(tx, current, previous, cfg)


def run(stages: list[str], cfg: dict[str, Any] | None = None) -> None:
    """Run the requested stages in order."""
    cfg = cfg or load_config()
    fns = {
        "data": stage_data,
        "features": stage_features,
        "train": stage_train,
        "segment": stage_segment,
        "score": stage_score,
    }
    for stage in stages:
        start = time.perf_counter()
        fns[stage](cfg)
        log.info("stage finished", extra={"stage": stage, "seconds": round(time.perf_counter() - start, 1)})


def needed_stages(cfg: dict[str, Any]) -> list[str]:
    """Stages required to make a fresh deployment servable (used by ``--if-needed``)."""
    from sqlalchemy import inspect

    from src.scoring import engine

    if not Path(cfg["paths"]["artifacts"], "model_card.json").exists():
        return STAGES
    if not inspect(engine()).has_table("predictions"):
        return ["data", "score"] if not Path(cfg["paths"]["interim"]).exists() else ["score"]
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--if-needed", action="store_true", help="Only run what a fresh deployment is missing")
    args = parser.parse_args()
    stages = needed_stages(load_config()) if args.if_needed else args.stages
    log.info("pipeline plan", extra={"stages": stages})
    run(stages)
