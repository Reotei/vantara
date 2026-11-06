"""Export a read-only snapshot of the API as static JSON, for hosting the dashboard without a backend.

Usage: python scripts/export_static.py frontend/public/snapshot
The dashboard reads it when built with VITE_STATIC=1 (see README, "Hosted snapshot").
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlencode

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.main import app  # noqa: E402

SHARDS = 32
OVERVIEW = {"page_size": 6, "sort_by": "revenue_at_risk", "order": "desc", "risk_band": ["High", "Medium"]}


def fnv1a(text: str) -> str:
    """32-bit FNV-1a hash (mirrored in frontend/src/api/static.ts)."""
    h = 0x811C9DC5
    for byte in text.encode():
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def canonical(path: str, params: dict) -> str:
    pairs = sorted((k, str(v)) for k, vs in params.items() for v in (vs if isinstance(vs, list) else [vs]))
    return f"{path}?{urlencode(pairs)}"


def main(out: Path) -> None:
    client = TestClient(app)
    out.mkdir(parents=True, exist_ok=True)
    get = lambda path, params=None: client.get(path, params=params).raise_for_status().json()  # noqa: E731

    def write(path: str, params: dict, body: object) -> None:
        (out / f"{fnv1a(canonical(path, params))}.json").write_text(json.dumps(body, separators=(",", ":")))

    options = get("/filters")
    filters = [{}] + [{"segment": [s]} for s in options["segments"]] + [{"tier": [t]} for t in options["tiers"]]
    filters += [{"country": [c]} for c in options["countries"][:6]]
    for f in filters:
        for path in ["/kpis", "/trends", "/segments"]:
            write(path, f, get(path, f))
    for path in ["/filters", "/models", "/health"]:
        write(path, {}, get(path))

    everyone = get("/customers", {"page_size": 200, "page": 1})
    rows = everyone["items"]
    for page in range(2, everyone["total"] // 200 + 2):
        rows += get("/customers", {"page_size": 200, "page": page})["items"]
    customers = {r["customer_id"]: r for r in rows}
    assert everyone["total"] == len(customers)
    (out / "customers.json").write_text(json.dumps(list(customers.values()), separators=(",", ":")))

    shards: dict[int, dict] = {}
    for cid in customers:
        shards.setdefault(cid % SHARDS, {})[cid] = get(f"/customers/{cid}")
    for i, shard in shards.items():
        (out / f"profiles-{i}.json").write_text(json.dumps(shard, separators=(",", ":")))
    top = sorted(rows, key=lambda r: -r["revenue_at_risk"] * (r["risk_band"] != "Low"))[:400]
    (out / "lime.json").write_text(
        json.dumps({r["customer_id"]: get(f"/customers/{r['customer_id']}/lime") for r in top})
    )
    print(f"exported {len(filters)} filter views, {len(customers)} profiles, {len(top)} LIME explanations to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "frontend/public/snapshot"))
