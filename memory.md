# Project Memory

Append one entry after every implementation iteration. Record what changed, decisions made, validation performed, and the next concrete step. Keep older entries unchanged.

## Iteration 0 — 2026-09-14 — Planning baseline

### Work completed

- Read `churn_prediction_spec_driven_plan.md` in full.
- Established this file as the append-only implementation log.
- Captured the delivery sequence: setup, data foundation, labeling, feature engineering, model development, registry/scoring, business validation, deployment, then monitoring.

### Decisions and constraints

- Use MySQL for raw data, point-in-time labels, versioned features, model metadata, and predictions.
- Use XGBoost for binary churn prediction with time-based train/validation/test splits.
- Treat PR-AUC as the primary model metric; also track precision/recall, recall at the top decile, and Brier score.
- Prevent leakage by excluding events after each observation date and joining every feature row to exactly one label row.
- Do not promote a model unless it passes agreed acceptance thresholds.
- Keep v1 batch-oriented; real-time scoring, causal explanations, and multi-product churn remain out of scope.

### Validation

- Confirmed the plan defines requirements, data and feature schemas, modeling/evaluation rules, architecture, deployment, monitoring, risks, and phased tasks.
- No implementation or pipeline validation was performed in this documentation-only iteration.

### Next step

- Review the existing Phase 0 setup against the plan, resolve the open churn definition and prediction horizon, then record the next iteration here.

## Iteration 1 — 2026-09-14 — Phase 1 data foundation

### Work completed

- Implemented an explicit MySQL `customers_raw` schema for every field in the supplied Telco churn CSV.
- Implemented CSV loading, whitespace normalization, type conversion, and fail-fast checks for schema drift, empty input, blank or duplicate customer IDs, invalid binary values, negative/invalid numeric values, and core-column blank rates above 5%.
- Added a transactional full-snapshot load so reruns replace raw rows deterministically and failed inserts roll back.
- Replaced placeholder credentials with the `DATABASE_URI` environment variable and corrected the source CSV path.
- Wired the ingestion stage into `run_pipeline.py`, documented its commands, pinned its three dependencies, and ignored Python cache files.
- Added one focused standard-library regression test covering the real dataset and duplicate-ID rejection.

### Decisions and constraints

- Followed the repository's non-negotiable flat-file architecture: the available CSV populates `customers_raw`; normalized event tables from the broader plan are deferred until corresponding source data exists.
- Preserved the 11 blank `TotalCharges` values as SQL `NULL`; all other supplied core fields passed validation.
- Treated the source CSV as a complete snapshot rather than building incremental-load machinery without an incremental source.

### Validation

- `python -m src.ingest --validate-only`: passed, 7,043 rows validated.
- `python -m unittest tests.test_ingest`: passed, 1 test.
- Live MySQL schema creation and insertion were not run because `DATABASE_URI` and a MySQL runtime are not available in this environment.

### Next step

- Configure a reachable MySQL database, run `python run_pipeline.py`, verify 7,043 rows in `customers_raw`, then implement Phase 2 labeling from the supplied `Churn` outcome or a stakeholder-approved future-window definition.
