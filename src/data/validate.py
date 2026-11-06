"""Schema and quality gates that must pass before feature engineering runs."""

from typing import Any

import pandas as pd
import pandera.pandas as pa

from src.utils import get_logger

log = get_logger(__name__)


def transaction_schema(cfg: dict[str, Any]) -> pa.DataFrameSchema:
    """Pandera schema for the cleaned transaction table."""
    lo, hi = pd.Timestamp(cfg["data"]["min_date"]), pd.Timestamp(cfg["data"]["max_date"])
    return pa.DataFrameSchema(
        {
            "invoice": pa.Column(str),
            "stock_code": pa.Column(str),
            "quantity": pa.Column(int, pa.Check.ne(0)),
            "price": pa.Column(float, pa.Check.gt(0)),
            "invoice_date": pa.Column("datetime64[ns]", pa.Check.in_range(lo, hi)),
            "customer_id": pa.Column("Int64", nullable=True),
            "country": pa.Column(str),
            "revenue": pa.Column(float),
        },
        checks=[
            pa.Check(
                lambda d: d["customer_id"].isna().mean() <= cfg["data"]["max_missing_customer_rate"],
                error="missing customer_id rate above threshold",
            ),
            pa.Check(lambda d: not d.duplicated().any(), error="exact duplicate rows present"),
        ],
    )


def validate(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Raise ``pandera.errors.SchemaErrors`` listing every violation; log a summary when clean."""
    try:
        transaction_schema(cfg).validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        log.error("data validation failed", extra={"failures": err.failure_cases.head(20).to_dict("records")})
        raise
    log.info(
        "data validation passed", extra={"rows": len(df), "missing_customer_rate": df["customer_id"].isna().mean()}
    )
    return df
