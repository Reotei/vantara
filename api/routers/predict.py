"""Scoring endpoints: single customer, CSV batch, and service health."""

import io
import json

import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from pydantic import ValidationError
from sqlalchemy import text

from api import schemas
from api.deps import db, models
from src.data.clean import COLUMNS, clean, standardize
from src.features.build import data_end
from src.scoring import score_customers, score_features
from src.utils import get_logger, load_config

router = APIRouter(tags=["scoring"])
log = get_logger(__name__)
MAX_UPLOAD_MB = 60


def _predictions(scored: pd.DataFrame) -> list[schemas.Prediction]:
    return [
        schemas.Prediction(
            customer_id=int(i),
            contributions=json.loads(r["shap"]),
            **{
                k: (None if pd.isna(r[k]) else r[k]) if k == "next_purchase_prob" else r[k]
                for k in [
                    "churn_prob",
                    "risk_band",
                    "clv_pred",
                    "revenue_at_risk",
                    "next_purchase_prob",
                    "is_anomaly",
                    "segment",
                    "explanation",
                ]
            },
        )
        for i, r in scored.iterrows()
    ]


@router.post("/predict", response_model=schemas.Prediction)
def predict(payload: dict = Body(..., examples=[{"recency_days": 120, "frequency": 4, "monetary_total": 1250.0}])):
    """Score one customer from engineered features. Unspecified features default to the training median."""
    b = models()
    model = schemas.feature_request_model(b["card"]["features"], b["card"]["medians"])
    try:
        record = model.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False)) from exc
    cid = record.pop("customer_id")
    features = pd.DataFrame([record], index=pd.Index([cid or 0], name="customer_id"))
    result = _predictions(score_features(features, None, load_config()))[0]
    return result.model_copy(update={"customer_id": cid})


@router.post("/predict/batch", response_model=schemas.BatchResult)
async def predict_batch(file: UploadFile = File(..., description="Transaction-level CSV in Online Retail II format")):
    """Score every customer in an uploaded transaction CSV (cutoff = day after its last transaction)."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    try:
        raw = pd.read_csv(io.BytesIO(content), dtype={"Invoice": str, "StockCode": str})
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(422, f"Could not read CSV: {exc}") from exc
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise HTTPException(422, f"Missing required columns: {', '.join(missing)}. Expected: {', '.join(COLUMNS)}.")
    cfg = load_config()
    try:
        tx = clean(standardize(raw), cfg)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, f"Invalid values in upload: {exc}") from exc
    tx = tx[tx["customer_id"].notna()]
    if tx.empty:
        raise HTTPException(422, "No rows with a valid Customer ID and positive price were found.")
    cutoff = data_end(tx)
    scored = score_customers(tx, cutoff, cfg, batch=True)
    log.info("batch scored", extra={"rows": len(raw), "customers": len(scored)})
    return schemas.BatchResult(
        scored=len(scored),
        skipped_inactive=int(tx["customer_id"].nunique() - len(scored)),
        cutoff=cutoff.date(),
        rows_received=len(raw),
        rows_used=len(tx),
        items=sorted(_predictions(scored), key=lambda p: -p.revenue_at_risk),
    )


@router.get("/health", response_model=schemas.Health)
def health() -> schemas.Health:
    """Liveness/readiness: database reachable and models loaded."""
    try:
        with db().connect() as conn:
            conn.execute(text("SELECT 1"))
        database = True
    except Exception:  # noqa: BLE001 - any connection failure means "not ready"
        database = False
    try:
        card = models()["card"]
    except HTTPException:
        card = None
    status = "ok" if database and card else "degraded"
    return schemas.Health(
        status=status,
        database=database,
        models_loaded=card is not None,
        production_model=card and card["production_model"],
    )
