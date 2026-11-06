# Progress

Single source of truth for where the build stands. Update after every stage.

## Done
- Repo scaffolded per PRD §15; config in `config/config.yaml`; JSON structured logging; seeds fixed.
- Data: Online Retail II verified against UCI (1,067,371 rows, 5,942 customers). Cleaning + pandera validation (`src/data`).
- Features: 40+ customer features with point-in-time cutoffs; leakage tests (`tests/test_features.py`).
- Models: LR, decision tree, RF, XGBoost, LightGBM, KNN, ANN, LSTM, autoencoder; CLV (Ridge / Tweedie LightGBM / log LightGBM); all runs in `models_artifacts/runs.jsonl`.
- Segmentation: K-Means (most granular k within 0.025 of best silhouette, min segment 3%) + GMM cross-check; 6 named personas.
- XAI: TreeSHAP per customer, LIME on demand, PDPs, plain-language sentence.
- API: FastAPI with all §12.1 endpoints; Postgres (SQLite fallback); integration tests.
- Frontend: React 19 + Vite + Tailwind 4 + Recharts + TanStack; all 7 pages.
- Docker: `docker-compose up` → db, pipeline (`--if-needed`), api, frontend (nginx).
- Notebooks executed (`notebooks/*.ipynb`), figures in `docs/figures`, `docs/final_report.pdf`, architecture/ER/workflow diagrams.
- Final metrics (test): churn ROC-AUC 0.811, recall 0.874; CLV R² 0.69; API p95 ≈ 140 ms; 29 tests, 97% coverage; ruff + black clean.
- Read-only hosted snapshot published as a claude.ai artifact (`scripts/export_static.py` + `VITE_STATIC=1` build).

## Next
- Push to GitHub once a repo is available (code currently only in the build workspace + the zip sent in chat).
- Live deployment with backend (Render / Railway / HF Spaces) once the owner picks a host and provides access.
- Run `docker-compose up --build` once on a machine with Docker Hub access (images could not be pulled in the build environment).

## Decisions (also in README)
- Dataset fetched from a GitHub mirror of the UCI workbook (UCI blocked from the build machine); row/customer counts match UCI exactly.
- Customer universe = everyone seen in the 2 years before the cutoff (PRD's ~5.9k customers). AUC on the last-year-active subset is reported separately.
- Churn label: no purchase in the 90 days after the cutoff. CLV: net revenue in the 365 days after the cutoff (separate snapshot).
- SVM replaced by KNN (PRD allows either; `SVC(probability=True)` is deprecated in scikit-learn 1.9).
- Production churn model restricted to tree ensembles (exact TreeSHAP); chosen by validation ROC-AUC.
- Class imbalance handled with class weights (churn rate ≈ 56% in this universe, so SMOTE is unnecessary).
- PyTorch (via skorch) for all deep models.

## Bug log (each has a regression test)
1. Description standardisation created new exact duplicates → dedupe after standardising (`test_lines_differing_only_in_description_spelling_are_deduplicated`).
2. Same-day repeat orders produced an infinite overdue ratio → clip gap at 1 day (`test_same_day_repeat_orders_do_not_create_infinite_features`).
3. Engagement score was a percentile within whichever customers were being scored, so batch uploads scored differently from the full base → percentiles now use a persisted reference population (`test_engagement_score_does_not_depend_on_who_else_is_scored`).
4. Customers with no orders in the last year had NaN "active months", which the imputer filled with the median → filled with 0 (`test_count_features_are_zero_not_missing_for_lapsed_customers`).
