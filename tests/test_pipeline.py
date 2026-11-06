"""End-to-end pipeline on synthetic data: every stage runs and produces the promised artifacts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.explainability.explain import explain_sentence, phrase
from src.models.evaluate import pick_threshold
from src.utils import load


def test_all_models_trained_and_compared(trained: dict) -> None:
    card = load("model_card.json", trained)
    names = {r["model"] for r in card["comparison"]}
    assert {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "lightgbm",
        "knn",
        "ann",
        "lstm",
    } <= names
    assert card["production_model"] in {"random_forest", "xgboost", "lightgbm"}
    for row in card["comparison"]:
        assert {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"} <= set(row)
    assert {"mae", "rmse", "r2"} <= set(card["clv"]["test"])
    assert card["anomaly"]["threshold"] > 0


def test_every_run_is_logged(trained: dict) -> None:
    runs = [json.loads(line) for line in (Path(trained["paths"]["artifacts"]) / "runs.jsonl").read_text().splitlines()]
    assert {r["task"] for r in runs} == {"churn", "clv", "anomaly"}
    assert all("train_seconds" in r for r in runs)


def test_segments_have_business_names(trained: dict) -> None:
    report = load("segmentation_report.json", trained)
    assert report["k"] >= 3 and "silhouette" in report["kmeans"] and "ari_vs_kmeans" in report["gmm"]
    names = set(load("segments.joblib", trained)["names"].values())
    assert all(not n.isdigit() for n in names) and len(names) == report["k"]


def test_scored_tables_in_database(trained: dict) -> None:
    from api.deps import query

    preds = query("SELECT * FROM predictions")
    assert preds["churn_prob"].between(0, 1).all() and (preds["clv_pred"] >= 0).all()
    assert preds["explanation"].str.contains("churn risk").all()
    assert set(query("SELECT segment FROM segments")["segment"]) == set(preds["segment"])


def test_threshold_respects_min_recall() -> None:
    y = pd.Series([0, 0, 1, 1, 1, 0, 1, 0]).to_numpy()
    p = pd.Series([0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.6, 0.7]).to_numpy()
    t = pick_threshold(y, p, 0.75)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.75


@pytest.mark.parametrize("prob, risky", [(0.85, True), (0.1, False)])
def test_plain_language_sentence(prob: float, risky: bool) -> None:
    contribs = [
        {"feature": "recency_days", "label": "", "value": 214.0, "shap": 1.2 if risky else -1.2},
        {"feature": "cat_Christmas", "label": "", "value": 0.4, "shap": 0.3 if risky else -0.3},
    ]
    text = explain_sentence(prob, contribs, {"medium": 0.4, "high": 0.7})
    assert ("Flagged" in text) == risky and "214 days ago" in text and "Christmas" in text
    assert phrase("country_Germany", 1) == "they are based in Germany"
