"""Cleaning and validation rules."""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.clean import categorize, clean, standardize
from src.data.validate import validate


@pytest.fixture(scope="module")
def tx(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return clean(standardize(raw), cfg)


def test_lines_differing_only_in_description_spelling_are_deduplicated(tx: pd.DataFrame, cfg: dict) -> None:
    """Regression: standardising descriptions created new exact duplicates that failed validation."""
    assert not tx.duplicated().any()
    validate(tx, cfg)


def test_cleaning_rules(tx: pd.DataFrame) -> None:
    assert (tx["price"] > 0).all()
    assert tx.loc[tx["stock_code"] == "POST", "is_admin"].all()
    assert not tx.loc[tx["stock_code"] == "85123A", "is_admin"].any()
    assert tx.loc[tx["invoice"].str.startswith("C"), "is_return"].all()
    assert tx.loc[tx["stock_code"] == "85123A", "description"].nunique() == 1
    assert tx["invoice_date"].is_monotonic_increasing


def test_reversed_extreme_orders_are_removed(raw: pd.DataFrame, cfg: dict) -> None:
    bad = pd.DataFrame(
        [
            ["581483", "23843", "PAPER CRAFT", 80995, "2011-12-09", 2.08, 16446.0, "United Kingdom"],
            ["C581484", "23843", "PAPER CRAFT", -80995, "2011-12-09", 2.08, 16446.0, "United Kingdom"],
        ],
        columns=raw.columns,
    )
    out = clean(standardize(pd.concat([raw, bad])), cfg)
    assert not (out["customer_id"] == 16446).any()


def test_categorize_first_keyword_wins(cfg: dict) -> None:
    out = categorize(pd.Series(["CHRISTMAS LUNCH BAG", "RED MUG", "MYSTERY ITEM"]), cfg["features"]["categories"])
    assert out.tolist() == ["Christmas", "Kitchen", "Other"]


def test_validation_fails_loudly(tx: pd.DataFrame, cfg: dict) -> None:
    with pytest.raises(pa.errors.SchemaErrors):
        validate(tx.assign(price=-1.0), cfg)
