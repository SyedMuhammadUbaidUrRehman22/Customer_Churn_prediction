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

## Iteration 2 — 2026-09-24 — Phase 2 snapshot labeling

### Work completed

- Connected to MySQL 8.0.46, loaded the validated CSV, and verified 7,043 unique customer rows including 1,869 supplied churn outcomes.
- Added the `churn_labels` table and deterministic `telco_snapshot_v1` materialization from `customers_raw.churned`.
- Added fail-fast checks for exactly one label per customer and the presence of both label classes.
- Wired label materialization into `run_pipeline.py` and added a focused regression test.

### Decisions and constraints

- Used the supplied `Churn` field as a benchmark outcome because no dated subscription or churn events are available.
- Left `observation_date` and `label_window_end` null rather than fabricating point-in-time evidence.
- This label set supports benchmark development only; it cannot substantiate a production future-window label or time-based evaluation.

### Validation

- Live raw load: 7,043 rows, 7,043 distinct customers, 1,869 churned.
- Full unit suite: 2 tests passed.
- Live label materialization: 7,043 rows, 7,043 distinct customers, 1,869 churned, and zero fabricated observation/window dates.

### Next step

- Materialize a versioned snapshot feature set with leakage controls, then train only as an explicitly non-temporal benchmark until dated events are supplied.

## Iteration 3 — 2026-09-24 — Phase 3 snapshot feature store

### Work completed

- Added a versioned `feature_store` containing the 19 available non-label predictors from the Telco snapshot.
- Added deterministic full-snapshot materialization, exact customer-coverage validation, and the specified 5% null-rate gate.
- Excluded the churn outcome and ingestion metadata from the feature list and added a regression test for that leakage boundary.
- Wired feature materialization into the main pipeline.

### Decisions and constraints

- Preserved the 11 missing `total_charges` values for explicit model-pipeline imputation.
- Did not fabricate observation dates or unavailable rolling-window usage, billing, or support features.
- Versioned this constrained benchmark feature set as `telco_snapshot_v1`.

### Validation

- Full unit suite: 3 tests passed.
- Live feature materialization: 7,043 rows, 7,043 distinct customers, 11 null `total_charges` values, and no `churned` column.
- Two consecutive full pipeline runs produced identical row counts with zero label or feature orphans.

### Next step

- Build an explicitly non-temporal XGBoost benchmark with a reproducible split, preprocessing, PR-AUC evaluation, and no production-promotion claim.

## Iteration 4 — 2026-09-24 — README and Phase 0–3 review

### Work completed

- Rebuilt the README with section navigation, expandable reference panels, a pipeline diagram, a 19-field dictionary, read-only verification queries, troubleshooting, and a phase roadmap.
- Added `docs/phase_0_3_review.md` with prioritized findings and evidence. No application code or live schema was changed during this review.
- Preserved the user's stop at Phase 3; Phase 4 remains unstarted.

### Validation and corrections

- Existing unit suite: 3 tests passed; included CSV: 7,043 rows validated.
- Read-only MySQL audit: 7,043 rows in each table; 7,043 version-filtered label/feature joins; zero label mismatches and zero mismatches across all 19 copied predictors.
- Confirmed 1,869 churned, 5,174 retained, 11 missing total charges, and no populated observation/window dates.
- Temporary-CSV probes reproduced fractional-tenure truncation, accepted infinity/out-of-range numbers, invalid categories, a blank contract, and an overlong ID. Empty feature counts also passed the validator.
- PyMySQL is now installed locally (1.2.3), but the configured driver fails with the helper's `connection_timeout` argument. The successful database audit explicitly used the pinned MySQL Connector driver.
- Identified separate-stage commits combined with cascading deletes as a refresh failure risk, and a nullable date CHECK that permits partial date pairs.
- Earlier target-column exclusion checks do not demonstrate temporal leakage safety. Snapshot implementation also does not satisfy the production temporal requirements or missing business sign-off gates in the SDD.

### Next step

- Address the recorded Phase 0–3 findings and regression coverage before considering model development. The review documents proposed fixes; it does not implement them.

### Documentation checks

- Verified 43 local links and fragments, balanced code fences, and all 14 expandable README panels.
- Git whitespace checks passed; `.env` remains ignored. GitHub visual rendering was not previewed during this review.
