"""Feature engineering correctness and point-in-time (leakage) guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean, standardize
from src.features.build import (
    build_features,
    build_labels,
    build_sequences,
    data_end,
    engagement_reference,
    feature_columns,
)


@pytest.fixture(scope="module")
def tx(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return clean(standardize(raw), cfg)


@pytest.fixture(scope="module")
def cutoff(tx: pd.DataFrame) -> pd.Timestamp:
    return data_end(tx) - pd.Timedelta(days=90)


def test_features_ignore_everything_after_cutoff(tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict) -> None:
    """Leakage test: deleting or rewriting post-cutoff data must not change any feature."""
    full = build_features(tx, cutoff, cfg)
    past_only = build_features(tx[tx["invoice_date"] < cutoff], cutoff, cfg)
    future = tx[tx["invoice_date"] >= cutoff]
    tampered = pd.concat(
        [
            tx[tx["invoice_date"] < cutoff],
            future.assign(quantity=future["quantity"] * 50, revenue=future["revenue"] * 50),
        ]
    )
    pd.testing.assert_frame_equal(full, past_only)
    pd.testing.assert_frame_equal(full, build_features(tampered, cutoff, cfg))


def test_sequences_ignore_everything_after_cutoff(tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict) -> None:
    ids = build_features(tx, cutoff, cfg).index
    np.testing.assert_array_equal(
        build_sequences(tx, cutoff, ids, cfg), build_sequences(tx[tx["invoice_date"] < cutoff], cutoff, ids, cfg)
    )


def test_labels_only_use_the_horizon_window(tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict) -> None:
    ids = build_features(tx, cutoff, cfg).index
    labels = build_labels(tx, cutoff, 90, ids)
    buyers = tx[(tx["invoice_date"] >= cutoff) & ~tx["is_return"] & ~tx["is_admin"]]["customer_id"].unique()
    assert labels.loc[labels.index.isin(buyers), "churn"].eq(0).all()
    assert labels.loc[~labels.index.isin(buyers), "churn"].eq(1).all()
    assert (labels["future_revenue"] >= 0).all()


def test_rfm_values_match_hand_computation(tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict) -> None:
    f = build_features(tx, cutoff, cfg)
    cid = f["frequency"].idxmax()
    mine = tx[(tx["customer_id"] == cid) & (tx["invoice_date"] < cutoff) & ~tx["is_admin"] & ~tx["is_return"]]
    assert f.at[cid, "frequency"] == mine["invoice"].nunique()
    assert f.at[cid, "recency_days"] == (cutoff - mine["invoice_date"].max()).days
    assert f.at[cid, "avg_order_value"] == pytest.approx(mine.groupby("invoice")["revenue"].sum().mean())


def test_feature_ranges(tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict) -> None:
    f = build_features(tx, cutoff, cfg)
    assert (f["recency_days"] <= cfg["labels"]["active_window_days"]).all()
    assert f[["q4_share", "return_rate", "markdown_share"]].stack().between(0, 1).all()
    assert f.filter(like="cat_").sum(axis=1).round(6).eq(1).all()
    assert f["engagement_score"].between(0, 100).all()
    assert "country" not in feature_columns(f) and "snapshot" not in feature_columns(f)


def test_same_day_repeat_orders_do_not_create_infinite_features(
    tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict
) -> None:
    """Regression: two orders on the same day gave gap_mean = 0 and an infinite overdue ratio."""
    first = tx[(tx["invoice_date"] < cutoff) & tx["customer_id"].notna() & ~tx["is_return"] & ~tx["is_admin"]].iloc[[0]]
    twin = first.assign(customer_id=pd.array([99999], dtype="Int64"), invoice="X1")
    twins = pd.concat([tx, twin, twin.assign(invoice="X2")])
    f = build_features(twins, cutoff, cfg)
    assert np.isfinite(f[feature_columns(f)].fillna(0).to_numpy()).all()


def test_engagement_score_does_not_depend_on_who_else_is_scored(
    tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict
) -> None:
    """Regression: engagement was a percentile within the scored batch, so small uploads got different scores."""
    full = build_features(tx, cutoff, cfg)
    few = full.index[:5]
    batch = build_features(tx[tx["customer_id"].isin(few)], cutoff, cfg, reference=engagement_reference(full))
    np.testing.assert_allclose(batch["engagement_score"], full.loc[few, "engagement_score"])


def test_count_features_are_zero_not_missing_for_lapsed_customers(
    tx: pd.DataFrame, cutoff: pd.Timestamp, cfg: dict
) -> None:
    """Regression: customers with no orders in the last year got NaN active months, later imputed to the median."""
    f = build_features(tx, cutoff, cfg)
    lapsed = f[f["recency_days"] > 365]
    assert len(lapsed) and lapsed["active_months_last_year"].eq(0).all()
    counts = ["frequency", "orders_last_90d", "orders_same_season_last_year", "active_months_last_year"]
    assert not f[counts].isna().any().any()
