"""SHAP / LIME wrappers, partial dependence, and the plain-language 'why flagged' template."""

from typing import Any

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import BaseEstimator
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline

# Human-readable label and the phrase used when a feature drives a prediction.
LABELS: dict[str, tuple[str, str]] = {
    "recency_days": ("Days since last order", "their last order was {v:.0f} days ago"),
    "tenure_days": ("Customer tenure (days)", "they have been a customer for {v:.0f} days"),
    "frequency": ("Total orders", "they have placed {v:.0f} orders"),
    "avg_order_value": ("Average order value", "they spend £{v:,.0f} per order on average"),
    "avg_basket_items": ("Items per order", "they buy {v:.0f} items per order"),
    "avg_basket_lines": ("Products per order", "they buy {v:.0f} different products per order"),
    "gap_mean": ("Avg. days between orders", "they usually order every {v:.0f} days"),
    "gap_std": ("Order-gap variability", "their order timing varies by ±{v:.0f} days"),
    "orders_last_90d": ("Orders in last 90 days", "they placed {v:.0f} orders in the last 90 days"),
    "revenue_last_90d": ("Spend in last 90 days", "they spent £{v:,.0f} in the last 90 days"),
    "orders_same_season_last_year": (
        "Orders in same season last year",
        "they placed {v:.0f} orders in this season last year",
    ),
    "active_months_last_year": ("Active months (last 12)", "they ordered in {v:.0f} of the last 12 months"),
    "overdue_ratio": ("Overdue vs usual gap", "it has been {v:.1f}× their usual time between orders"),
    "order_trend": ("Order frequency trend", "their order frequency trend is {v:+.2f} orders/month"),
    "q4_share": ("Holiday-season spend share", "{v:.0%} of their spend falls in Oct–Dec"),
    "month_concentration": ("Spend concentration by month", "their spend is concentrated in a few months"),
    "markdown_share": ("Markdown purchase share", "{v:.0%} of their purchases were below the usual price"),
    "product_popularity": ("Product mainstreamness", "they mostly buy {q} products"),
    "monetary_total": ("Lifetime spend", "they have spent £{v:,.0f} in total"),
    "return_rate": ("Return rate", "they returned {v:.0%} of their spend"),
    "engagement_score": ("Engagement score", "their engagement score is {v:.0f}/100"),
}


def label(feature: str) -> str:
    """Human-readable feature name."""
    if feature.startswith("cat_"):
        return f"{feature[4:].replace('_', ' ')} spend share"
    if feature.startswith("country_"):
        return f"Based in {feature[8:].replace('_', ' ')}"
    return LABELS.get(feature, (feature.replace("_", " ").capitalize(), ""))[0]


def phrase(feature: str, value: float) -> str:
    """Sentence fragment describing a feature value."""
    if feature.startswith("cat_"):
        return f"{value:.0%} of their spend is on {feature[4:].replace('_', ' ')}"
    if feature.startswith("country_"):
        return f"they {'are' if value else 'are not'} based in {feature[8:].replace('_', ' ')}"
    if feature == "product_popularity":
        return LABELS[feature][1].format(q="niche" if value < 0.05 else "best-selling")
    template = LABELS.get(feature, ("", f"their {feature.replace('_', ' ')} is {{v:.2f}}"))[1]
    return template.format(v=0 if pd.isna(value) else value)


def _split(model: BaseEstimator) -> tuple[Any, Any]:
    return (model[:-1], model[-1]) if isinstance(model, Pipeline) else (None, model)


class ChurnExplainer:
    """TreeSHAP for the production churn model plus LIME for local comparison."""

    def __init__(self, model: BaseEstimator, background: pd.DataFrame, seed: int) -> None:
        self.model, self.columns = model, list(background.columns)
        self.prep, final = _split(model)
        self.tree = shap.TreeExplainer(final)
        # scikit-learn forests explain probabilities directly; boosters explain log-odds.
        self.link = "identity" if type(final).__module__.startswith("sklearn") else "logit"
        self.lime = LimeTabularExplainer(
            background.fillna(background.median()).to_numpy(),
            mode="classification",
            feature_names=self.columns,
            class_names=["Stays", "Churns"],
            discretize_continuous=True,
            random_state=seed,
        )

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP contributions (log-odds space) toward churn, shape (n, d)."""
        Xt = X if self.prep is None else pd.DataFrame(self.prep.transform(X), columns=self.columns, index=X.index)
        values = np.asarray(self.tree.shap_values(Xt))
        return values[..., 1] if values.ndim == 3 else values

    @property
    def base_value(self) -> float:
        base = np.atleast_1d(self.tree.expected_value)
        return float(base[-1])

    def lime_weights(self, row: pd.Series, samples: int) -> list[dict[str, Any]]:
        """LIME local weights for one customer (positive = pushes toward churn)."""
        predict = lambda a: self.model.predict_proba(pd.DataFrame(a, columns=self.columns))  # noqa: E731
        exp = self.lime.explain_instance(
            row[self.columns].fillna(0).to_numpy(float), predict, num_features=8, num_samples=samples
        )
        lookup = dict(exp.as_map()[1])
        return [
            {"feature": self.columns[i], "label": label(self.columns[i]), "rule": rule, "weight": float(lookup[i])}
            for (rule, _), i in zip(exp.as_list(), [i for i, _ in exp.as_map()[1]], strict=True)
        ]


def top_contributions(shap_row: np.ndarray, values: pd.Series, n: int = 8) -> list[dict[str, Any]]:
    """Largest absolute SHAP contributions with labels and raw values."""
    order = np.argsort(-np.abs(shap_row))[:n]
    return [
        {
            "feature": values.index[i],
            "label": label(values.index[i]),
            "value": None if pd.isna(values.iat[i]) else float(values.iat[i]),
            "shap": float(shap_row[i]),
        }
        for i in order
    ]


def explain_sentence(probability: float, contributions: list[dict[str, Any]], bands: dict[str, float]) -> str:
    """Translate SHAP contributions into one sentence a marketing analyst can read."""
    risky = probability >= bands["medium"]
    drivers = [c for c in contributions if (c["shap"] > 0) == risky][:3]
    level = "High" if probability >= bands["high"] else "Moderate" if risky else "Low"
    reasons = [phrase(c["feature"], c["value"]) for c in drivers]
    joined = (
        ", ".join(reasons[:-1]) + (" and " if len(reasons) > 1 else "") + reasons[-1] if reasons else "no single factor"
    )
    verb = "Flagged" if risky else "Looks stable"
    return f"{level} churn risk ({probability:.0%}): {verb} mainly because {joined}."


def pdp(model: BaseEstimator, X: pd.DataFrame, features: list[str]) -> dict[str, list[dict[str, float]]]:
    """Partial dependence of predicted churn probability on each feature (for charts)."""
    out = {}
    for f in features:
        res = partial_dependence(model, X, [f], grid_resolution=30, percentiles=(0.02, 0.98))
        out[f] = [{"x": float(x), "y": float(y)} for x, y in zip(res["grid_values"][0], res["average"][0], strict=True)]
    return out
