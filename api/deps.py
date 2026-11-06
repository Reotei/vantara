"""Shared API dependencies: database access, model bundle and global dashboard filters."""

import os
import time
from dataclasses import dataclass
from datetime import date
from functools import cache, lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException, Query
from sqlalchemy import text

from src.scoring import bundle, engine
from src.utils import load_config


@cache
def db():
    """Process-wide SQLAlchemy engine."""
    return engine(os.getenv("DATABASE_URL"))


def models() -> dict[str, Any]:
    """Loaded model bundle (503 if artifacts are missing)."""
    try:
        return bundle(load_config()["paths"]["artifacts"])
    except FileNotFoundError as exc:
        raise HTTPException(503, "Models are not trained yet. Run `python -m src.pipeline`.") from exc


def query(sql: str, **params: Any) -> pd.DataFrame:
    """Run a parameterised query and return a DataFrame."""
    with db().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


@dataclass
class Filters:
    """Global filters shared by every dashboard page."""

    segment: list[str] | None = Query(None)
    country: list[str] | None = Query(None)
    tier: list[str] | None = Query(None, description="Value tier: High, Mid, Low")
    date_from: date | None = Query(None, description="Last purchase on or after")
    date_to: date | None = Query(None, description="Last purchase on or before")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a customer-level frame (needs segment, country, value_tier, last_purchase)."""
        mask = pd.Series(True, index=df.index)
        for col, values in [("segment", self.segment), ("country", self.country), ("value_tier", self.tier)]:
            if values:
                mask &= df[col].isin(values)
        last = pd.to_datetime(df["last_purchase"])
        if self.date_from:
            mask &= last >= pd.Timestamp(self.date_from)
        if self.date_to:
            mask &= last <= pd.Timestamp(self.date_to)
        return df[mask]


CUSTOMER_SQL = """
SELECT c.*, p.* FROM customers c JOIN predictions p ON p.customer_id = c.customer_id
"""


@lru_cache(maxsize=2)
def _customers(_bucket: int) -> pd.DataFrame:
    df = query(CUSTOMER_SQL)
    return df.loc[:, ~df.columns.duplicated()]


def customers(filters: Filters | None = None) -> pd.DataFrame:
    """Joined customer + prediction frame (cached for a minute; scores only change on pipeline runs)."""
    df = _customers(int(time.time() // 60))
    return filters.apply(df) if filters else df
