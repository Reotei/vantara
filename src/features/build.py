"""Customer-level features, labels and purchase sequences with strict point-in-time cutoffs.

Every function only reads transactions dated strictly before ``cutoff`` (labels read the window after it),
so the same code is safe for training snapshots, live scoring and batch uploads.
"""

from typing import Any

import numpy as np
import pandas as pd

META = ["country", "snapshot"]


def data_end(tx: pd.DataFrame) -> pd.Timestamp:
    """First day after the last transaction: the 'today' of a dataset."""
    return tx["invoice_date"].max().normalize() + pd.Timedelta(days=1)


def history(tx: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Identified-customer product lines strictly before the cutoff."""
    return tx[(tx["invoice_date"] < cutoff) & tx["customer_id"].notna() & ~tx["is_admin"]]


def sku_stats(hist: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU popularity (share of customers buying it) and median price — the frequency encoding."""
    sales = hist[~hist["is_return"]]
    g = sales.groupby("stock_code")
    return pd.DataFrame(
        {
            "popularity": g["customer_id"].nunique() / max(sales["customer_id"].nunique(), 1),
            "median_price": g["price"].median(),
        }
    )


def _orders(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            date=("invoice_date", "min"),
            value=("revenue", "sum"),
            items=("quantity", "sum"),
            lines=("stock_code", "nunique"),
            category=("category", lambda s: s.mode().iat[0]),
        )
        .sort_values("date", kind="stable")
    )


def _trend_slope(orders: pd.DataFrame, cutoff: pd.Timestamp, months: int) -> pd.Series:
    """Least-squares slope of monthly order counts over the trailing window (orders/month per month)."""
    age = (cutoff - orders["date"]).dt.days // 30
    recent = orders[age < months].assign(t=lambda d: months - 1 - (cutoff - d["date"]).dt.days // 30)
    counts = recent.pivot_table(index="customer_id", columns="t", values="invoice", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=range(months), fill_value=0)
    x = np.arange(months) - (months - 1) / 2
    return pd.Series(counts.to_numpy() @ x / (x @ x), index=counts.index)


def engagement_reference(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Sorted R/F/M values of a population, so engagement percentiles can be computed for any later batch."""
    return {
        "recency": np.sort(-features["recency_days"].to_numpy()),
        "frequency": np.sort(features["frequency"].to_numpy()),
        "monetary": np.sort(features["monetary_total"].to_numpy()),
    }


def _percentile(values: pd.Series, reference: np.ndarray) -> np.ndarray:
    """Average-rank percentile of each value within ``reference`` (matches ``rank(pct=True)`` on the same data)."""
    left, right = np.searchsorted(reference, values, "left"), np.searchsorted(reference, values, "right")
    return (left + right + 1) / 2 / len(reference)


def build_features(
    tx: pd.DataFrame,
    cutoff: pd.Timestamp,
    cfg: dict[str, Any],
    skus: pd.DataFrame | None = None,
    reference: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Engineer the customer feature table for customers active within the scoring window before ``cutoff``."""
    fc = cfg["features"]
    hist = history(tx, cutoff)
    sales, returns = hist[~hist["is_return"]], hist[hist["is_return"]]
    orders = _orders(sales)
    g = orders.groupby("customer_id")
    last = g["date"].max()
    ids = last.index[last >= cutoff - pd.Timedelta(days=cfg["labels"]["active_window_days"])]
    gaps = g["date"].diff().dt.days
    recent = orders[orders["date"] >= cutoff - pd.Timedelta(days=90)].groupby("customer_id")
    year_ago = cutoff - pd.Timedelta(days=365)
    last_season = orders[(orders["date"] >= year_ago) & (orders["date"] < year_ago + pd.Timedelta(days=90))]
    returned = -returns.groupby("customer_id")["revenue"].sum()
    rev = sales.groupby("customer_id")["revenue"]
    month_rev = sales.pivot_table(
        index="customer_id", columns=sales["invoice_date"].dt.month, values="revenue", aggfunc="sum", fill_value=0
    )
    shares = month_rev.div(month_rev.sum(axis=1), axis=0)
    cats = sales.pivot_table(index="customer_id", columns="category", values="revenue", aggfunc="sum", fill_value=0)
    cats = cats.reindex(columns=[*fc["categories"], "Other"], fill_value=0)
    skus = sku_stats(hist) if skus is None else skus
    priced = sales.join(skus, on="stock_code")
    country = hist.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0])

    f = pd.DataFrame(
        {
            "recency_days": (cutoff - last).dt.days,
            "tenure_days": (cutoff - g["date"].min()).dt.days,
            "frequency": g.size(),
            "avg_order_value": g["value"].mean(),
            "avg_basket_items": g["items"].mean(),
            "avg_basket_lines": g["lines"].mean(),
            "gap_mean": gaps.groupby(orders["customer_id"]).mean(),
            "gap_std": gaps.groupby(orders["customer_id"]).std(),
            "orders_last_90d": recent.size(),
            "revenue_last_90d": recent["value"].sum(),
            "orders_same_season_last_year": last_season.groupby("customer_id").size(),
            "active_months_last_year": orders[orders["date"] >= year_ago]
            .groupby("customer_id")["date"]
            .apply(lambda d: d.dt.to_period("M").nunique()),
            "order_trend": _trend_slope(orders, cutoff, fc["trend_months"]),
            "q4_share": sales[sales["invoice_date"].dt.month >= 10].groupby("customer_id")["revenue"].sum() / rev.sum(),
            "month_concentration": (shares**2).sum(axis=1),
            "markdown_share": (priced["price"] < fc["markdown_ratio"] * priced["median_price"])
            .groupby(priced["customer_id"])
            .mean(),
            "product_popularity": priced.groupby("customer_id")["popularity"].mean(),
        }
    ).reindex(ids)
    gross = rev.sum().reindex(ids)
    f["monetary_total"] = (gross - returned.reindex(ids).fillna(0)).clip(lower=0)
    f["return_rate"] = (returned.reindex(ids).fillna(0) / gross).clip(0, 1)
    zero_fill = [
        "orders_last_90d",
        "revenue_last_90d",
        "orders_same_season_last_year",
        "active_months_last_year",
        "order_trend",
        "q4_share",
    ]
    f[zero_fill] = f[zero_fill].fillna(0)
    f["overdue_ratio"] = f["recency_days"] / f["gap_mean"].clip(lower=1)
    f = f.join(cats.div(cats.sum(axis=1), axis=0).add_prefix("cat_").rename(columns=lambda c: c.replace(" ", "_")))
    f["country"] = country.reindex(ids)
    for c in fc["top_countries"]:
        f[f"country_{c.replace(' ', '_')}"] = (f["country"] == c).astype(int)
    ref = reference or engagement_reference(f)
    f["engagement_score"] = 100 * np.mean(
        [
            _percentile(-f["recency_days"], ref["recency"]),
            _percentile(f["frequency"], ref["frequency"]),
            _percentile(f["monetary_total"], ref["monetary"]),
        ],
        axis=0,
    )
    numeric = f.columns.difference(["country"])
    f[numeric] = f[numeric].astype(float)
    f["snapshot"] = cutoff
    f.index = f.index.astype("int64")
    return f.rename_axis("customer_id")


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Model input columns (everything except metadata)."""
    return [c for c in features.columns if c not in META]


def build_labels(tx: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int, ids: pd.Index) -> pd.DataFrame:
    """Churn (no purchase in the horizon) and future net revenue for the given customers."""
    end = cutoff + pd.Timedelta(days=horizon_days)
    future = tx[
        (tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end) & tx["customer_id"].notna() & ~tx["is_admin"]
    ]
    buyers = future.loc[~future["is_return"], "customer_id"].unique()
    revenue = future.groupby("customer_id")["revenue"].sum().reindex(ids).fillna(0).clip(lower=0)
    return pd.DataFrame({"churn": (~ids.isin(buyers)).astype(int), "future_revenue": revenue.to_numpy()}, index=ids)


def build_sequences(tx: pd.DataFrame, cutoff: pd.Timestamp, ids: pd.Index, cfg: dict[str, Any]) -> np.ndarray:
    """Left-padded sequences of each customer's last orders: value, size, gap, age and category one-hot."""
    seq_len = cfg["models"]["deep"]["seq_len"]
    cats = [*cfg["features"]["categories"], "Other"]
    hist = history(tx, cutoff)
    orders = _orders(hist[~hist["is_return"] & hist["customer_id"].isin(ids)])
    orders["gap"] = orders.groupby("customer_id")["date"].diff().dt.days.fillna(0) / 30
    orders["age"] = (cutoff - orders["date"]).dt.days / 365
    orders = orders.groupby("customer_id").tail(seq_len)
    steps = np.column_stack(
        [
            np.log1p(orders["value"].clip(lower=0)),
            np.log1p(orders["items"].clip(lower=0)),
            orders["gap"],
            orders["age"],
            pd.get_dummies(pd.Categorical(orders["category"], categories=cats)).to_numpy(float),
        ]
    )
    out = np.zeros((len(ids), seq_len, steps.shape[1]), dtype=np.float32)
    row = pd.Series(np.arange(len(ids)), index=ids.astype("int64"))[orders["customer_id"].astype("int64")].to_numpy()
    pos = seq_len - 1 - orders.groupby("customer_id").cumcount(ascending=False).to_numpy()
    out[row, pos] = steps
    return out
