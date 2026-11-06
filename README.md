# Vantara Customer Intelligence

Vantara Customer Intelligence predicts which customers are about to stop buying, what each one is worth over the next
year, and why. It covers every customer seen in the last two years of orders. For each one it produces:

- **Churn risk**: the probability of no purchase in the next 90 days, with SHAP and LIME explanations and a plain-English reason.
- **Customer lifetime value (CLV)**: predicted net revenue over the next 12 months, and **revenue at risk** (churn risk × CLV).
- **Next-purchase probability**: from a sequence model of each customer's order history.
- **Segment and recommended play**: one of six behavioural segments, each with a suggested marketing action.
- **Unusual-spending flag**: set by an autoencoder for manual review.
- **Product recommendations**: popular items in the customer's favourite categories that they haven't bought yet.

| Churn ROC-AUC | Churn recall | CLV R² | API p95 (single prediction) |
|---|---|---|---|
| **0.811** | **0.874** | **0.69** | **≈ 140 ms** |

All figures are measured once on held-out test customers. The full write-up is in [`docs/final_report.pdf`](docs/final_report.pdf).

![Architecture](docs/architecture_diagram.png)

## Quick start (Docker)

```bash
cp .env.example .env        # optional: change the database password
docker-compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API + Swagger docs | http://localhost:8000/docs |

On the first run, the `pipeline` service downloads the Online Retail II workbook, cleans and validates it, trains every model,
and loads the scores into PostgreSQL. This takes about 10 minutes on a laptop. Later runs reuse the saved artifacts and start
in seconds. To force a full retrain:

```bash
docker-compose run --rm pipeline python -m src.pipeline
```

## Hosted snapshot (no backend)

The dashboard can also be published as a static site that reads a read-only export of the scores. Every page,
filter, profile and chart works. Batch scoring and on-demand LIME are limited to what was exported (LIME is
included for the 400 customers with the most revenue at risk).

```bash
python scripts/export_static.py /tmp/snapshot                     # needs DATABASE_URL + trained artifacts
cd frontend && VITE_STATIC=1 VITE_BASE=./ npx vite build --outDir ../site && cp -r /tmp/snapshot ../site/
```

Upload `site/` to any static host.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.pipeline                       # raw data → features → models → segments → scores (SQLite by default)
uvicorn api.main:app --reload                # http://localhost:8000/docs
cd frontend && npm install && npm run dev    # http://localhost:5173 (proxies /api to :8000)
```

To use PostgreSQL locally, set `DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/vantara`.

Single stages can be re-run with `python -m src.pipeline --stages train score`. The stages are `data`, `features`, `train`,
`segment` and `score`.

### Tests and quality

```bash
pytest --cov=src                     # 29 tests, ~97% coverage on src/; includes leakage and API integration tests
ruff check src api tests && black --check src api tests
python notebooks/build_notebooks.py  # regenerate + execute the three notebooks (writes docs/figures)
python docs/build_report.py          # regenerate docs/final_report.pdf
```

The tests run the whole pipeline on a synthetic dataset that has the real schema and the same data-quality problems, so they
don't need the 45 MB workbook.

## Using the dashboard

| Page | What it answers |
|---|---|
| **Overview** | How much revenue is at risk, how that compares with last quarter, and who to call first |
| **Churn risk** | A sortable leaderboard ranked by revenue at risk, with filters for risk band and unusual spending |
| **Segments** | Segment profiles with recommended plays, a customer map (PCA) and side-by-side comparisons |
| **Customer profile** | Churn gauge, CLV, the reasons for the score (SHAP waterfall, with an optional LIME check), purchase timeline and recommended products |
| **Batch scoring** | Upload a transaction CSV and get scored results as a table, CSV or PDF report |
| **Model quality** | Model comparison, ROC curve, confusion matrix, feature effects, and CLV / anomaly / segmentation checks |

The global filters (segment, country, value tier, last-purchase date) apply to every page and are kept in the URL, so a
filtered view can be shared as a link. Press <kbd>Ctrl</kbd>+<kbd>K</kbd> to jump to any customer.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Checks the database is reachable and the models are loaded |
| POST | `/predict` | Score one customer from engineered features (only RFM is required; other features default to training medians) |
| POST | `/predict/batch` | Score a transaction CSV in Online Retail II format |
| GET | `/models` | Model comparison, ROC curve, confusion matrix, PDPs, CLV, anomaly and segmentation diagnostics |
| GET | `/kpis`, `/trends`, `/segments` | Dashboard aggregates; all accept `segment`, `country`, `tier`, `date_from`, `date_to` |
| GET | `/customers` | Paginated, filterable, sortable customer list |
| GET | `/customers/{id}`, `/customers/{id}/lime` | Customer profile with SHAP; LIME computed on demand |
| GET | `/filters` | Values for the filter bar |

All configuration comes from environment variables: `DATABASE_URL`, `ARTIFACTS_DIR`, `LOG_LEVEL` and `CORS_ORIGINS`.
Model settings, paths and thresholds live in `config/config.yaml`.

## How it works

1. **Data** (`src/data`): both workbook sheets are merged and cleaned. Exact duplicates are removed after descriptions are
   standardised; zero and negative prices are dropped; huge orders that were fully cancelled are removed as entry errors;
   postage and fee codes are flagged; returns are kept as a signal. A pandera schema stops the pipeline with a full
   violation report if any check fails.
2. **Features** (`src/features`): 40+ customer features are computed from data strictly before a cutoff date. They cover RFM,
   basket size, order-frequency trend, gaps between orders, seasonality, category mix, return rate, markdown share, product
   popularity, country and an engagement score. Tests prove that changing post-cutoff data leaves every feature unchanged.
3. **Labels**: churn means no purchase in the 90 days after the cutoff. CLV means net revenue in the 365 days after a separate cutoff.
4. **Models** (`src/models`):
   - Six classical churn models and two deep-learning models (ANN, LSTM), plus an autoencoder for anomalies.
   - Tuning uses 5-fold cross-validated random search on the 70% training split. Each model's alert threshold is tuned on the
     15% validation split so it catches at least 70% of churners. The 15% test split is used once.
   - Every run is logged to `models_artifacts/runs.jsonl`.
5. **Segments** (`src/segmentation`): K-Means, with k chosen by silhouette and a minimum segment size, cross-checked against a
   Gaussian mixture. Each segment gets a business name and a recommended action.
6. **Explainability** (`src/explainability`): TreeSHAP for every customer, LIME on demand, partial dependence plots, and a
   sentence template that turns SHAP values into plain English.
7. **Serving**: `src/scoring.py` writes the `customers`, `transactions`, `predictions` and `segments` tables. FastAPI reads them,
   and the dashboard talks only to the API.

## Decisions

- **Dataset source.** The pipeline downloads the UCI workbook from a public GitHub mirror (`config.yaml → paths.raw_url`),
  because it can't be fetched from UCI non-interactively. Row and customer counts match UCI exactly: 1,067,371 rows and
  5,942 customers. To use your own copy, put `online_retail_II.xlsx` in `data/raw/`.
- **Which customers are scored.** Everyone with an order in the two years before the cutoff, which matches the PRD's ~5.9k
  customers. Accuracy on customers active in the last year is reported separately (ROC-AUC 0.805).
- **KNN instead of SVM** as the distance-based model. The PRD allows either, and `SVC(probability=True)` is deprecated in
  scikit-learn 1.9.
- **Production churn model.** The tree ensemble with the best *validation* ROC-AUC, currently a random forest. Tree models
  give exact, fast SHAP values. The LSTM scores almost as well and powers next-purchase probability.
- **CLV model.** Ridge, Tweedie LightGBM and log-target LightGBM were compared, and the one with the best cross-validated R² was
  kept (Tweedie). Future revenue is very skewed: most customers spend £0, and a few wholesalers spend six figures.
- **Class imbalance.** Churn is about 56% in this population, so class weights are used instead of SMOTE.
- **Choosing k for segments.** The most granular k whose silhouette is within 0.025 of the best and whose smallest segment holds at
  least 3% of customers. Pure maximum silhouette picks 3–4 segments, too coarse for marketing, and one of them was a 1% cluster.
- **Leaderboard default.** The Churn risk page opens filtered to medium- and high-risk customers. Otherwise, very large
  low-risk accounts top a revenue-at-risk ranking. The "Low risk" chip shows them again.
- **PyTorch via skorch** for all deep models, so they plug into the same scikit-learn tooling as the classical ones.
- **Frontend stack.** shadcn/ui's registry couldn't be reached from the build machine, so its components are built from the
  same open-source primitives (Radix UI, Tailwind, cmdk). The design takes cues from Linear, Tremor and Magic UI patterns
  (glass cards, animated KPI numbers, cursor spotlight) using original layouts.
- **Repository layout.** It follows PRD §15, plus `requirements-dev.txt` (dev-only dependencies), `pyproject.toml`
  (lint and test settings) and `src/pipeline.py` / `src/scoring.py`, the single entry points for training and scoring.

Progress, decisions and a bug log (each bug has a regression test) are in [`PROGRESS.md`](PROGRESS.md).

## Data licence

Online Retail II is distributed under CC BY 4.0. Cite it as: Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C5CG6D
