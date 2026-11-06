"""Load the Online Retail II workbook and turn it into a clean, typed transaction table."""

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)
COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read both workbook sheets (downloading the file if absent) into one chronological table."""
    path = Path(cfg["paths"]["raw"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading raw dataset", extra={"url": cfg["paths"]["raw_url"]})
        urllib.request.urlretrieve(cfg["paths"]["raw_url"], path)
    df = pd.concat(pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str}).values())
    log.info("loaded raw transactions", extra={"rows": len(df)})
    return standardize(df)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case and coerce types."""
    df = df.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()
    for col in ["invoice", "stock_code"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["description"] = df["description"].fillna("").astype(str).str.upper().str.split().str.join(" ")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df["country"] = df["country"].astype(str).str.strip()
    return df.sort_values("invoice_date", kind="stable").reset_index(drop=True)


def categorize(descriptions: pd.Series, categories: dict[str, list[str]]) -> pd.Series:
    """Map product descriptions to a category by first matching keyword."""
    result = pd.Series("Other", index=descriptions.index)
    unassigned = pd.Series(True, index=descriptions.index)
    for name, words in categories.items():
        hit = unassigned & descriptions.str.contains("|".join(rf"\b{w}" for w in words), regex=True)
        result[hit], unassigned = name, unassigned & ~hit
    return result


def drop_reversed_entry_errors(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Remove extreme-quantity lines that were fully cancelled (and their cancellations)."""
    big = df[df["quantity"].abs() >= threshold]
    key = ["customer_id", "stock_code"]
    sales, returns = big[big["quantity"] > 0], big[big["quantity"] < 0].assign(quantity=lambda d: -d["quantity"])
    pairs = sales.reset_index().merge(returns.reset_index(), on=[*key, "quantity"], suffixes=("_s", "_r"))
    drop = set(pairs["index_s"]) | set(pairs["index_r"])
    log.info("dropped reversed entry errors", extra={"rows": len(drop)})
    return df.drop(index=list(drop))


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply the documented cleaning rules and add derived columns."""
    rows = len(df)
    df = df[df["price"] > 0]  # zero/negative prices are bad-debt adjustments and manual corrections
    df = drop_reversed_entry_errors(df, cfg["data"]["extreme_quantity"])
    lookup = df[df["description"] != ""].groupby("stock_code")["description"].agg(lambda s: s.mode().iat[0])
    df = df.assign(description=df["stock_code"].map(lookup).fillna(df["description"])).drop_duplicates()
    df["is_admin"] = ~df["stock_code"].str.match(cfg["data"]["product_code_pattern"])
    df["is_return"] = df["invoice"].str.startswith("C") | (df["quantity"] < 0)
    df["revenue"] = df["quantity"] * df["price"]
    df["category"] = np.where(df["is_admin"], "Admin", categorize(df["description"], cfg["features"]["categories"]))
    q1, q3 = df["quantity"].quantile([0.25, 0.75])
    df["is_bulk"] = df["quantity"] > q3 + cfg["data"]["iqr_multiplier"] * (q3 - q1)
    df["month"] = df["invoice_date"].dt.strftime("%Y-%m")
    log.info("cleaned transactions", extra={"rows_in": rows, "rows_out": len(df)})
    return df.reset_index(drop=True)
