"""Pydantic request/response models (also the source of the dashboard's generated TypeScript types)."""

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model


class Delta(BaseModel):
    value: float
    previous: float | None


class Bin(BaseModel):
    bucket: str
    low: float
    customers: int
    revenue_at_risk: float


class Kpis(BaseModel):
    customers: int
    active_customers: Delta
    at_risk_customers: Delta
    revenue_at_risk: Delta
    avg_clv: Delta
    threshold: float
    risk_distribution: list[Bin]
    risk_bands: dict[str, int]


class TrendPoint(BaseModel):
    month: str
    revenue: float | None = None
    customers: int | None = None
    forecast: float | None = None
    lower: float | None = None
    upper: float | None = None


class CustomerRow(BaseModel):
    customer_id: int
    country: str
    segment: str
    value_tier: str
    risk_band: str
    churn_prob: float
    clv_pred: float
    revenue_at_risk: float
    next_purchase_prob: float
    top_driver: str
    recency_days: int
    frequency: int
    monetary_total: float
    last_purchase: date
    is_anomaly: bool


class CustomerPage(BaseModel):
    items: list[CustomerRow]
    total: int
    page: int
    page_size: int


class Contribution(BaseModel):
    feature: str
    label: str
    value: float | None
    shap: float


class Product(BaseModel):
    stock_code: str
    description: str
    category: str
    price: float


class Order(BaseModel):
    invoice: str
    date: date
    value: float
    items: int
    is_return: bool


class CustomerProfile(CustomerRow):
    tenure_days: int
    avg_order_value: float
    return_rate: float
    engagement_score: float
    first_purchase: date
    anomaly_score: float
    pca_x: float
    pca_y: float
    base_value: float
    shap_link: str = Field(description="'identity': SHAP values are probabilities; 'logit': log-odds")
    explanation: str
    contributions: list[Contribution]
    recommendations: list[Product]
    timeline: list[Order]
    category_mix: dict[str, float]
    segment_action: str
    prev_churn_prob: float | None


class LimeWeight(BaseModel):
    feature: str
    label: str
    rule: str
    weight: float


class LimeComparison(BaseModel):
    lime: list[LimeWeight]
    shap: list[Contribution]
    shared_top_features: list[str]
    agreement: float = Field(description="Share of LIME's top features that are also in SHAP's top features")


class Segment(BaseModel):
    segment: str
    size: int
    avg_spend: float
    avg_order_value: float
    churn_rate: float
    avg_clv: float
    avg_recency: float
    avg_frequency: float
    q4_share: float
    revenue_at_risk: float
    action: str


class ScatterPoint(BaseModel):
    customer_id: int
    x: float
    y: float
    segment: str
    churn_prob: float


class Segments(BaseModel):
    segments: list[Segment]
    scatter: list[ScatterPoint]


class FilterOptions(BaseModel):
    countries: list[str]
    segments: list[str]
    tiers: list[str]
    min_date: date
    max_date: date
    scored_at: date


class Prediction(BaseModel):
    customer_id: int | None = None
    churn_prob: float
    risk_band: str
    clv_pred: float
    revenue_at_risk: float
    next_purchase_prob: float | None
    is_anomaly: bool
    segment: str
    explanation: str
    contributions: list[Contribution]


class BatchResult(BaseModel):
    scored: int
    skipped_inactive: int
    cutoff: date
    rows_received: int
    rows_used: int
    items: list[Prediction]


class Health(BaseModel):
    status: str
    database: bool
    models_loaded: bool
    production_model: str | None


def feature_request_model(features: list[str], medians: dict[str, float]) -> type[BaseModel]:
    """Single-customer request model: RFM fields are required, the rest default to training medians."""
    required = {"recency_days", "frequency", "monetary_total"}
    fields: dict[str, Any] = {
        f: (float, Field(... if f in required else medians.get(f, 0.0), ge=None if f == "order_trend" else 0))
        for f in features
    }
    fields["customer_id"] = (int | None, None)
    return create_model("CustomerFeatures", __config__=ConfigDict(extra="forbid"), **fields)
