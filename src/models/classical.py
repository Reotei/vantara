"""Classical churn classifiers and the CLV regressor, tuned with cross-validated random search."""

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.models.evaluate import log_run, regression_report
from src.utils import get_logger

log = get_logger(__name__)
TREE_MODELS = {"random_forest", "xgboost", "lightgbm"}


def churn_models(seed: int, pos_weight: float) -> dict[str, BaseEstimator]:
    """Untuned estimators; scale-sensitive models get an imputer + StandardScaler pipeline."""

    def scaled(m: BaseEstimator) -> Pipeline:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)

    return {
        "logistic_regression": scaled(LogisticRegression(class_weight="balanced", solver="saga", max_iter=5000)),
        "decision_tree": make_pipeline(
            SimpleImputer(strategy="median"), DecisionTreeClassifier(class_weight="balanced", random_state=seed)
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed),
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=pos_weight, n_estimators=400, eval_metric="auc", n_jobs=-1, random_state=seed
        ),
        "lightgbm": LGBMClassifier(
            class_weight="balanced", n_estimators=400, subsample_freq=1, verbose=-1, random_state=seed
        ),
        "knn": scaled(KNeighborsClassifier()),
    }


def _grid(model: BaseEstimator, grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    prefix = f"{model.steps[-1][0]}__" if isinstance(model, Pipeline) else ""
    return {prefix + k: v for k, v in grid.items()}


def tune(
    name: str, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]
) -> tuple[BaseEstimator, dict]:
    """Random search with stratified k-fold CV on the training set only; returns refitted best model + info."""
    mc = cfg["models"]
    search = RandomizedSearchCV(
        model,
        _grid(model, mc["grids"][name]),
        n_iter=mc["n_iter"],
        scoring="roc_auc",
        cv=StratifiedKFold(mc["cv_folds"], shuffle=True, random_state=cfg["seed"]),
        random_state=cfg["seed"],
        n_jobs=1 if name in TREE_MODELS else -1,
    )
    start = time.perf_counter()
    search.fit(X, y)
    info = {
        "params": search.best_params_,
        "cv_roc_auc": search.best_score_,
        "cv_roc_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "seconds": time.perf_counter() - start,
    }
    log.info("tuned model", extra={"model": name, "cv_roc_auc": round(search.best_score_, 4)})
    return search.best_estimator_, info


def refit_xgboost_early_stopping(
    model: XGBClassifier, X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> XGBClassifier:
    """Refit the tuned XGBoost with early stopping on the validation fold."""
    best = clone(model).set_params(n_estimators=2000, early_stopping_rounds=50)
    best.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return best


def train_clv(
    X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, cfg: dict[str, Any]
) -> tuple[BaseEstimator, str, dict[str, dict[str, float]]]:
    """Compare Ridge, Tweedie LightGBM and log-target LightGBM; keep the best cross-validated R² (raw £ scale)."""
    seed, folds = cfg["seed"], cfg["models"]["cv_folds"]
    lgbm = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=seed,
    )
    candidates = {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 20))
        ),
        "lightgbm_tweedie": clone(lgbm).set_params(objective="tweedie"),
        "lightgbm_log": TransformedTargetRegressor(clone(lgbm), func=np.log1p, inverse_func=np.expm1),
    }
    reports, fitted = {}, {}
    for name, model in candidates.items():
        start = time.perf_counter()
        cv = cross_val_score(model, X_tr, y_tr, cv=KFold(folds, shuffle=True, random_state=seed), scoring="r2")
        fitted[name] = model.fit(X_tr, y_tr)
        reports[name] = {**regression_report(y_val.to_numpy(), fitted[name].predict(X_val)), "cv_r2": float(cv.mean())}
        log_run("clv", name, {}, reports[name], time.perf_counter() - start, cfg)
    best = max(reports, key=lambda n: reports[n]["cv_r2"])
    log.info("selected CLV model", extra={"model": best, "val_r2": round(reports[best]["r2"], 4)})
    return fitted[best], best, reports
