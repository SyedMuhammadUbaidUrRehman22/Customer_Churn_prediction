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

## Iteration 5 — 2026-09-25 — Phase 4 XGBoost snapshot benchmark

### Work completed

- Read project memory and the full SDD; implemented the requested XGBoost phase as the previously anticipated non-temporal benchmark.
- Added the explicit snapshot evaluation amendment to SDD Section 5.2: deterministic stratified 60/20/20 splits, eight randomized candidates, validation early stopping and threshold selection, untouched final test.
- Implemented MySQL-only version-filtered loading with consistent-read coverage checks and independent feature-domain validation, train-only median imputation/one-hot encoding, class weighting, metrics, original-feature importances, and local model/preprocessor/report/split artifacts.
- Fixed the shared PyMySQL timeout keyword and pinned already-installed XGBoost 3.1.3, scikit-learn 1.8.0, and joblib 1.5.3. No new framework was introduced; existing sklearn utilities handle preprocessing and search.
- Updated README, the legacy primary-metric instruction, and `docs/phase_4_evaluation.md`. Generated model directories remain local and ignored.

### Validation

- Five regression tests passed, including deterministic customer-disjoint splits, invalid training domains, training-only imputer fit, artifact reload metric equivalence, and both driver timeout variants.
- Live MySQL training read 7,043 customers without refreshing or mutating tables. Split sizes: 4,225 / 1,409 / 1,409.
- Eight-candidate verified run saved to `models/phase4_benchmark_verified/`: test average precision 0.658753 (baseline 0.265436), ROC-AUC 0.845402, precision 0.589912, recall 0.719251, F1 0.648193, top-decile recall 0.294118, top-decile precision 0.780142, Brier 0.164171.
- Initial tests exposed object-typed CSV labels and NaN in XGBoost's parameter metadata; fixed integer normalization and JSON null serialization before the successful rerun. `models/phase4_benchmark/` is an incomplete initial artifact directory, not the verified run.

### Decisions and next step

- The user authorized proceeding to XGBoost. Training is standalone (`python -m src.train`); `run_pipeline.py` continues to refresh only Phases 1–3.
- No time-based evaluation, probability-calibration claim, stakeholder acceptance, registry, or scoring was fabricated. `approved=false` remains unconditional.
- The upstream review findings R1/R2/R4/R5/R6 remain open; training guards protect this entry point but do not repair ingestion or schema behavior. R3's timeout mismatch is fixed; `.env` still needs explicit process-environment loading.
- Next: repair the outstanding foundation findings before production use, then Phase 5 registry/scoring under the benchmark-only boundary or after obtaining temporal data and stakeholder gates.

## Iteration 6 — 2026-09-25 — Phase 5 registry and benchmark scoring

### Work completed

- Read the full SDD and project memory; implemented the authorized Phase 5 snapshot adaptation and documented it in SDD Section 7.
- Added explicit InnoDB `model_registry` and `model_predictions` schemas. Registry rows record typed held-out metrics, feature/label versions, training time, local artifact directory, report/dataset hashes, and provisional tier boundaries. No database JSON blobs or new dependencies were introduced.
- Added immutable, repeatable registration of trusted local artifacts through `python -m src.registry --artifact-dir PATH` and opt-in registration after successful training with `python -m src.train --register`.
- Added label-free benchmark scoring through `python -m src.score --benchmark --model-id ID`. Reused the training feature-domain checks in `src.features` rather than duplicating validation or requiring labels during scoring.
- Added report/model/preprocessor hash checks before deserialization, matching recorded package versions and feature order, and transactionally replaced scores under a model row lock. DDL runs separately from score publication.
- Updated README, reconciled legacy table names in `Copilot_context`, and added `docs/phase_5_registry_scoring.md`.

### Validation and live state

- Local suite: 10 tests passed. One additional opt-in MySQL integration test passed independently.
- Verified artifact registration returned `xgb_9ee0132da5749c7e72124a8c`, with one registry row and `approved=0`. Re-registering returned the same ID without changing metadata.
- Live scoring wrote 7,043 benchmark rows. Two runs produced identical customer IDs, probabilities, risk tiers, and feature-snapshot hash `323b82ee993494aaf9216ffe1dffe53f523ea25934c52494ff204a4f8bea8659`.
- An injected failure after prediction inserts rolled back completely, preserving previous scores and timestamps. Default and explicit production scoring refused to run. MySQL rejected direct approval of a benchmark model via CHECK; the test transaction rolled back.
- The initial live test assumed CHECK violations were IntegrityError; this PyMySQL version reports OperationalError. Updated the test to assert the DBAPI CHECK error (3819), then reran successfully. No application change was needed for that driver behavior.
- Regression checks cover artifact tampering before deserialization, feature/dependency mismatches, tier boundaries/nonfinite scores, label-free invalid inputs, isolated SQL rollback/model-mode preservation, and training-before-registration ordering.
- No raw, label, or feature refresh was run. The Phase 4 verified artifacts were reused unchanged.

### Decisions and next step

- All predictions are explicitly marked `benchmark`. No active-customer cohort was inferred from churn labels. Default production scoring remains unavailable until dated features, an active-customer source, and an accepted model exist.
- Provisional high boundary is the validation-F1 threshold (0.59137362241745), medium starts at half that threshold (0.295686811208725); Phase 6 must review outreach criteria and tiers. Scores are not claimed to be calibrated future probabilities.
- Predictions retain only the latest batch per model/mode and survive raw snapshot refreshes; they can therefore be stale until rescored. Timestamp and feature hash identify their source snapshot. Local artifacts must remain available at their registered location.
- Existing upstream findings R1/R2/R4/R5/R6 remain open. Phase 6 business acceptance and foundation fixes are next; million-customer SLA, deployment, and monitoring are not delivered. `run_pipeline.py` remains the Phase 1–3 refresh command.

## Iteration 7 — 2026-09-25 — Phase 6 business validation

### Work completed

- Read the full SDD and memory; inspected implementation, schemas, tests, trusted model artifacts, live registry, predictions, and labels before coding.
- Added `src.business_validation` with deterministic score/customer-ID ranking, top-1/5/10/20% and custom-N scenarios, precision/recall/cumulative capture, and provisional tier counts and churn rates. Reused Phase 5 tier rules and persisted Phase 4 test metrics without retraining or rescoring.
- Added Markdown, exact JSON, and ranked CSV report generation. The default live artifacts are in `models/phase6_business_validation/`; a separate custom N=100 report is in `models/phase6_custom/`.
- Added the explicit `business_validation` MySQL schema and optional immutable, repeatable pending-record persistence. Four scenario records are stored; duplicate scenario Ns share a record within an analysis.
- Added unresolved business criteria to the existing `src.config` module, focused local and opt-in MySQL regressions, the Phase 6 workflow document, README updates, and a narrow SDD snapshot amendment.

### Decisions and constraints

- PR-AUC minimum, precision@K minimum, stakeholder outreach capacity, and refresh cadence remain unresolved. The existing SDD 0.5 feature-importance guardrail remains diagnostic. CLI custom Ns are scenarios, not stakeholder decisions.
- Full-snapshot outreach and tier statistics include training customers and are explicitly distinguished from saved held-out metrics. Labels remain the supplied Telco Churn outcome; no dates, future-window labels, calibrated future probabilities, active cohort, costs, or stakeholder decisions were fabricated.
- Business and stakeholder status remain pending. Configuration cannot grant approval; the database rejects benchmark approval. The registered benchmark remains `approved=false`.
- Consistent reads require exact customer coverage, one score batch, Phase 5 tier consistency, a matching feature snapshot hash, and the evaluated feature/label dataset hash. Reports retain source hashes and scoring time; repeat saves preserve first-persistence date and timestamp.
- No dependencies were added. Phase 4/5 behavior and raw/label/feature/prediction data were unchanged. Deployment, orchestration, monitoring, retraining automation, and the existing upstream findings remain outside this iteration.

### Validation

- Local suite: 14 tests passed, with both opt-in MySQL tests skipped by default. Focused checks cover ranking ties, formulas, tier totals/boundaries, unresolved criteria, approval refusal, missing models, mismatched coverage, stale scores, changed labels, dated labels, mixed score timestamps, and deterministic reruns.
- Live MySQL source: one unapproved benchmark model, 7,043 unique benchmark predictions and labels, 1,869 churners, and no observation/window dates. Registered artifacts verified successfully.
- Default CLI with `--save` succeeded. Top 1/5/10/20% selected 71/353/705/1,409 customers and captured 67/309/550/969 supplied churners. High/medium/low tier counts were 2,320/1,816/2,907.
- Phase 6 opt-in MySQL integration passed: two runs produced byte-identical report artifacts, four unique pending records, and exact equality between persisted values and report-derived rows. The benchmark stakeholder-approval attempt was rejected and rolled back; registry approval and prediction rows remained unchanged.
- Default analysis SHA-256: `cf44c4f6de4abc889fd72f0c6f4e37e5b8b80df0dc3a3ddf0a39852e8623e526`. The custom N=100 CLI also succeeded without persisting extra scenarios.

### Next step

- The Phase 6 analysis layer is implemented, but actual business acceptance remains blocked on stakeholder metric thresholds, outreach capacity/costs, cadence, tier policy, and documented sign-off. Obtain these decisions and the dated features/events, future-window labels, temporal evaluation, and verified active cohort required for production approval before proceeding to Phase 7 deployment; address the outstanding foundation findings as well.
