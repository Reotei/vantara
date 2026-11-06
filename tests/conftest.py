"""Shared fixtures: a synthetic Online-Retail-II-shaped dataset and a fast, isolated config."""

import copy
import os

import numpy as np
import pandas as pd
import pytest

from src.utils import load_config

PRODUCTS = {
    "22086": "PAPER CHAIN KIT 50'S CHRISTMAS",
    "85099B": "JUMBO BAG RED RETROSPOT",
    "85123A": "WHITE HANGING HEART T-LIGHT HOLDER",
    "21212": "PACK OF 72 RETROSPOT CAKE CASES",
    "22423": "REGENCY CAKESTAND 3 TIER",
    "84879": "ASSORTED COLOUR BIRD ORNAMENT",
    "22197": "POPCORN HOLDER",
    "21034": "REX CASH+CARRY JUMBO SHOPPER",
    "22720": "SET OF 3 CAKE TINS PANTRY DESIGN",
    "23084": "RABBIT NIGHT LIGHT",
    "22111": "SCOTTIE DOG HOT WATER BOTTLE",
    "21754": "HOME BUILDING BLOCK WORD",
    "22910": "PAPER CHAIN KIT VINTAGE CHRISTMAS",
    "47566": "PARTY BUNTING",
    "22382": "LUNCH BAG SPACEBOY DESIGN",
    "20725": "LUNCH BAG RED RETROSPOT",
    "22457": "NATURAL SLATE HEART CHALKBOARD",
    "21731": "RED TOADSTOOL LED NIGHT LIGHT",
    "22960": "JAM MAKING SET WITH JARS",
    "84946": "ANTIQUE SILVER TEA GLASS ETCHED",
    "21232": "STRAWBERRY CERAMIC TRINKET POT",
    "22139": "RETROSPOT TEA SET CERAMIC 11 PC",
    "84077": "WORLD WAR 2 GLIDERS ASSTD DESIGNS",
    "22492": "MINI PAINT SET VINTAGE",
}
COUNTRIES = ["United Kingdom"] * 6 + ["Germany", "France", "EIRE", "Spain"]
START, END = pd.Timestamp("2009-12-01 08:00"), pd.Timestamp("2011-12-09 12:00")


def make_raw(n_customers: int = 260, seed: int = 0) -> pd.DataFrame:
    """Raw transactions with the real schema and its known quality issues (returns, guests, admin codes, dupes)."""
    rng = np.random.default_rng(seed)
    codes, rows, inv = list(PRODUCTS), [], 500000
    for cid in range(12346, 12346 + n_customers):
        kind = rng.choice(["loyal", "churner", "seasonal", "new", "whale"], p=[0.3, 0.3, 0.15, 0.15, 0.1])
        first = START + pd.Timedelta(days=int(rng.integers(0, 120 if kind != "new" else 600)))
        last = END if kind in {"loyal", "whale", "seasonal"} else first + pd.Timedelta(days=int(rng.integers(30, 500)))
        gap = {"loyal": 25, "churner": 40, "seasonal": 15, "new": 30, "whale": 20}[kind]
        country, day = rng.choice(COUNTRIES), first
        while day < min(last, END):
            if kind != "seasonal" or day.month >= 10:
                inv += 1
                for code in rng.choice(codes, size=int(rng.integers(1, 6)), replace=False):
                    qty = int(rng.integers(1, 24) * (20 if kind == "whale" else 1))
                    rows.append(
                        [
                            str(inv),
                            code,
                            PRODUCTS[code],
                            qty,
                            day,
                            float(rng.choice([0.85, 1.25, 2.95, 4.95])),
                            float(cid),
                            country,
                        ]
                    )
                if rng.random() < 0.08:
                    rows.append(
                        [
                            f"C{inv}",
                            rows[-1][1],
                            rows[-1][2],
                            -1,
                            min(day + pd.Timedelta(days=3), END),
                            rows[-1][5],
                            float(cid),
                            country,
                        ]
                    )
                if rng.random() < 0.3:
                    rows.append([str(inv), "POST", "POSTAGE", 1, day, 18.0, float(cid), country])
            day += pd.Timedelta(days=int(rng.exponential(gap)) + 1, hours=int(rng.integers(0, 8)))
    for i in range(int(len(rows) * 0.15)):  # guest checkouts without a customer id
        rows.append(
            [
                str(900000 + i),
                codes[i % len(codes)],
                PRODUCTS[codes[i % len(codes)]],
                2,
                START + pd.Timedelta(hours=i * 7),
                1.65,
                np.nan,
                "United Kingdom",
            ]
        )
    rows += [
        rows[0],
        ["999999", "85123A", "white hanging heart  t-light holder", 4, END, 0.0, 12346.0, "United Kingdom"],
        [str(inv + 1), "85123A", "WHITE HANGING HEART T-LIGHT HOLDER ", 3, END, 2.55, 12346.0, "United Kingdom"],
        [str(inv + 1), "85123A", "WHITE HANGING HEART TLIGHT HOLDR", 3, END, 2.55, 12346.0, "United Kingdom"],
    ]
    cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def cfg(tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("vantara")
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(
        interim=str(root / "interim.parquet"),
        processed=str(root / "processed"),
        artifacts=str(root / "artifacts"),
        figures=str(root / "figures"),
    )
    cfg["models"].update(n_iter=2, cv_folds=3)
    cfg["models"]["deep"].update(max_epochs=6, patience=2, batch_size=64)
    cfg["models"]["grids"]["random_forest"]["n_estimators"] = [50]
    os.environ["DATABASE_URL"] = f"sqlite:///{root / 'test.db'}"
    return cfg


@pytest.fixture(scope="session")
def trained(cfg: dict, raw: pd.DataFrame) -> dict:
    """Run the full pipeline once on synthetic data; API tests read the resulting artifacts and database."""
    from src import pipeline, utils

    pipeline.stage_data(cfg, raw)
    pipeline.run(["features", "train", "segment", "score"], cfg)
    os.environ["ARTIFACTS_DIR"] = cfg["paths"]["artifacts"]
    utils.load_config.cache_clear()
    return cfg
