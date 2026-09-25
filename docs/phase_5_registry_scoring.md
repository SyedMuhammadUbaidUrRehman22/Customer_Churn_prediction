# Phase 5: model registry and benchmark scoring

Validated 2026-09-25 on MySQL 8.0.46, database `customer_churn`. The trusted Phase 4 model is registered as `xgb_9ee0132da5749c7e72124a8c`; all 7,043 snapshot customers have benchmark predictions. Raw data, labels, and feature tables were not refreshed.

## Delivered behavior

- `src.registry` creates the two explicit InnoDB schemas, verifies trusted local model/preprocessor/report artifacts, and records the model's time, feature and label versions, held-out metrics, provisional tier boundaries, artifact location, dataset hash, and report hash. Hyperparameters, importances, and dependency versions remain in the pinned report, without database JSON blobs.
- Re-registering the same artifacts returns the same ID and preserves metadata. A conflicting registration fails. The report hash is verified against MySQL before loading artifacts; model and preprocessor bytes are verified against that report before deserialization. Recorded package versions and feature order must match. These checks detect changes, not malicious artifacts supplied during initial registration: use trusted training outputs only.
- `src.train --register` registers only after successful training and artifact persistence. A registration failure leaves the local artifacts available for a retry through `src.registry`.
- `src.score --benchmark --model-id ID` loads the registered model and fitted preprocessing, reads the version-filtered feature store without churn labels, validates customer coverage/domains, and stores probabilities and risk tiers. There is no label-derived active-customer filter.
- Scoring obtains a registry row lock before its feature read and uses a repeatable-read transaction through prediction publication. Delete and batched insert either commit together or roll back. Other models/modes remain untouched. Predictions have a registry foreign key and deliberately no foreign key to the replaceable raw snapshot.

## Evidence

| Check | Result |
| --- | --- |
| Local regression suite | 10 tests passed |
| Opt-in MySQL integration | 1 test passed |
| Repeat registration | Same ID, one immutable registry row |
| Benchmark score command | 7,043 rows stored |
| Repeat scoring | Identical customers, probabilities, tiers, and snapshot hash |
| Failure after inserts | Previous scores and timestamps preserved by rollback |
| Default/explicit production scoring | Refused; scores unchanged |
| Direct benchmark approval attempt | MySQL CHECK rejected it; transaction rolled back |
| Artifact tampering | Report/model/preprocessor mutations rejected before joblib loading |
| Feature/dependency mismatch | Rejected before deserialization |

Feature snapshot SHA-256: `323b82ee993494aaf9216ffe1dffe53f523ea25934c52494ff204a4f8bea8659`. Registry `approved=0`, `evaluation_scope='benchmark'`; prediction `scoring_mode='benchmark'`. The registry stores Phase 4 held-out average precision 0.658753. Scoring all snapshot customers does not produce another unbiased evaluation.

High risk begins at the model's validation-F1 threshold, 0.59137362241745; medium begins at half, 0.295686811208725. These are exploratory display tiers, not stakeholder-approved outreach rules. Probabilities remain uncalibrated snapshot model outputs.

## Commands

Configure `DATABASE_URI` as described in the README, then:

```powershell
.\venv\Scripts\python.exe -m src.registry --artifact-dir models/phase4_benchmark_verified
.\venv\Scripts\python.exe -m src.score --benchmark --model-id xgb_9ee0132da5749c7e72124a8c
```

For a new training run, `python -m src.train --register` prints a new model ID. Default `python -m src.score` currently refuses because no approved production model exists. No approval-bypass command is provided.

## Boundaries and next phase

This completes the snapshot adaptation of Phase 5. Production active-customer scoring remains unavailable: it needs dated features, a verified active-customer source, temporal validation, and stakeholder acceptance. Even manually altering a row to production/approved does not enable the missing cohort implementation. Phase 6 is business review of acceptance metrics, outreach capacity, and tier policy; the existing upstream findings R1/R2/R4/R5/R6 also remain open.

The job uses in-memory inference for 7,043 rows; a million-customer SLA has not been tested. Artifacts are local and ignored by Git, so another machine needs the trusted artifact directory and matching environment. Moving registered artifacts requires deliberate metadata migration. Prediction refreshes keep the latest batch per model/mode; retained scores may become stale after raw refresh and are identifiable by timestamp/hash. `run_pipeline.py` still orchestrates Phases 1–3 only.

Implementation references: [MySQL implicit commits](https://dev.mysql.com/doc/refman/8.4/en/implicit-commit.html) explain separate schema setup; [scikit-learn persistence](https://scikit-learn.org/stable/model_persistence.html) explains trusted artifacts and matching dependency environments.
