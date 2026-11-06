"""Metrics, threshold selection and the experiment run log."""

import json
import time
from typing import Any

import numpy as np
from sklearn import metrics

from src.utils import artifact


def pick_threshold(y: np.ndarray, proba: np.ndarray, min_recall: float) -> float:
    """Highest-F1 threshold whose recall on the churned class is at least ``min_recall``."""
    best, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = proba >= t
        if metrics.recall_score(y, pred, zero_division=0) >= min_recall:
            f1 = metrics.f1_score(y, pred, zero_division=0)
            best, best_f1 = (t, f1) if f1 > best_f1 else (best, best_f1)
    return float(best)


def classification_report(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """All Section 11 classification metrics at a given threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": metrics.accuracy_score(y, pred),
        "precision": metrics.precision_score(y, pred, zero_division=0),
        "recall": metrics.recall_score(y, pred, zero_division=0),
        "f1": metrics.f1_score(y, pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y, proba),
        "threshold": threshold,
        "confusion_matrix": metrics.confusion_matrix(y, pred).tolist(),
    }


def regression_report(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and R² on the original scale plus R² on the log scale."""
    return {
        "mae": metrics.mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y, pred))),
        "r2": metrics.r2_score(y, pred),
        "r2_log": metrics.r2_score(np.log1p(y), np.log1p(np.clip(pred, 0, None))),
    }


def roc_points(y: np.ndarray, proba: np.ndarray, n: int = 60) -> list[dict[str, float]]:
    """Down-sampled ROC curve for charting."""
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def log_run(
    task: str, model: str, params: dict[str, Any], scores: dict[str, Any], seconds: float, cfg: dict[str, Any]
) -> None:
    """Append one model run (params, metrics, training time) to the JSON-lines experiment log."""
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task,
        "model": model,
        "params": params,
        "train_seconds": round(seconds, 2),
        **scores,
    }
    with artifact("runs.jsonl", cfg).open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
