"""API integration tests against the synthetic pipeline's artifacts and database."""

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_raw


@pytest.fixture(scope="module")
def client(trained: dict) -> TestClient:
    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def some_id(client: TestClient) -> int:
    return client.get("/customers", params={"page_size": 1}).json()["items"][0]["customer_id"]


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] and body["models_loaded"]


def test_kpis_and_filters(client: TestClient) -> None:
    opts = client.get("/filters").json()
    everyone = client.get("/kpis").json()
    one_segment = client.get("/kpis", params={"segment": opts["segments"][0]}).json()
    assert 0 < one_segment["customers"] < everyone["customers"]
    assert sum(b["customers"] for b in everyone["risk_distribution"]) == everyone["customers"]
    assert everyone["active_customers"]["previous"] is not None


def test_trends_include_forecast_band(client: TestClient) -> None:
    points = client.get("/trends").json()
    actual = [p for p in points if p["revenue"] is not None]
    future = [p for p in points if p["revenue"] is None]
    assert len(actual) >= 15 and len(future) == 3
    assert all(p["lower"] <= p["forecast"] <= p["upper"] for p in future)


def test_customers_sorted_paginated_and_searchable(client: TestClient, some_id: int) -> None:
    page = client.get("/customers", params={"page_size": 10, "sort_by": "revenue_at_risk"}).json()
    values = [r["revenue_at_risk"] for r in page["items"]]
    assert values == sorted(values, reverse=True) and len(values) == 10
    page2 = client.get("/customers", params={"page_size": 10, "page": 2}).json()
    assert {r["customer_id"] for r in page2["items"]}.isdisjoint({r["customer_id"] for r in page["items"]})
    found = client.get("/customers", params={"search": str(some_id)}).json()["items"]
    assert some_id in [r["customer_id"] for r in found]
    high = client.get("/customers", params={"risk_band": "High", "page_size": 200}).json()["items"]
    assert all(r["risk_band"] == "High" for r in high)
    assert client.get("/customers", params={"sort_by": "password"}).status_code == 422


def test_customer_profile_and_explanations(client: TestClient, some_id: int) -> None:
    profile = client.get(f"/customers/{some_id}").json()
    assert profile["contributions"] and profile["timeline"] and "churn risk" in profile["explanation"]
    lime = client.get(f"/customers/{some_id}/lime").json()
    assert len(lime["lime"]) == 8 and 0 <= lime["agreement"] <= 1
    assert client.get("/customers/1").status_code == 404
    assert client.get("/customers/1/lime").status_code == 404


def test_segments_and_models(client: TestClient) -> None:
    seg = client.get("/segments").json()
    assert seg["segments"] and seg["scatter"] and all(s["action"] for s in seg["segments"])
    card = client.get("/models").json()
    assert card["comparison"] and card["roc_curve"] and card["segmentation"]["k"] >= 3


def test_single_prediction_validates_and_is_fast(client: TestClient) -> None:
    body = {"recency_days": 300, "frequency": 1, "monetary_total": 40.0, "customer_id": 99}
    durations = []
    for _ in range(20):
        start = time.perf_counter()
        res = client.post("/predict", json=body)
        durations.append(time.perf_counter() - start)
    out = res.json()
    assert res.status_code == 200 and out["customer_id"] == 99 and 0 <= out["churn_prob"] <= 1
    assert np.percentile(durations, 95) < 0.4
    bad = client.post("/predict", json={"recency_days": -5, "frequency": 1})
    assert bad.status_code == 422
    assert client.post("/predict", json={**body, "unknown_field": 1}).status_code == 422


def test_batch_scoring_csv(client: TestClient) -> None:
    raw = make_raw(n_customers=40, seed=7)
    csv = raw.to_csv(index=False).encode()
    res = client.post("/predict/batch", files={"file": ("tx.csv", io.BytesIO(csv), "text/csv")})
    body = res.json()
    assert res.status_code == 200 and body["scored"] > 0 and body["rows_received"] == len(raw)
    risks = [i["revenue_at_risk"] for i in body["items"]]
    assert risks == sorted(risks, reverse=True)


def test_batch_rejects_bad_files(client: TestClient) -> None:
    missing = client.post("/predict/batch", files={"file": ("x.csv", b"Invoice,Quantity\n1,2\n", "text/csv")})
    assert missing.status_code == 422 and "Missing required columns" in missing.json()["detail"]
    empty = client.post("/predict/batch", files={"file": ("x.csv", b"", "text/csv")})
    assert empty.status_code == 422
