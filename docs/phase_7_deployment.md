# Phase 7: deployment infrastructure

Implemented for the production execution boundary on 2026-09-26. **Deployment infrastructure is available; actual production deployment remains blocked.** The only supplied data and trained model are the Telco snapshot benchmark: 7,043 customers, `feature_set_version=telco_snapshot_v1`, `evaluation_scope=benchmark`, `approved=0`.

## Architecture and boundary

```text
Scheduler (disabled until an owner chooses a cadence)
   ↓
Production preflight
   ↓
Raw-source and active-customer validation
   ↓
Feature refresh: consume the external ETL's dated materialization
   ↓
Approved production model and verified preprocessing
   ↓
Batch scoring without labels
   ↓
Complete-batch validation
   ↓
Atomic publication to MySQL
   ↓
Retention / BI
```

`python -m src.deploy` owns production orchestration. It reuses the shared database helper, feature-domain validator, artifact loader, risk-tier policy, and prediction writer. It does not run training, snapshot CSV ingestion, snapshot labeling, or Phase 6 exploratory business analysis. `run_pipeline.py` remains the Phase 1–3 development refresh command.

The production raw-data and ETL boundary is explicit. A real upstream ingestion/ETL job must supply `customers`, `subscriptions`, `billing_events`, `usage_events`, and `support_tickets` following the dated-source contract in the [specification](../churn_prediction_spec_driven_plan.md). Its published `feature_store` must contain `observation_date`, `feature_set_version`, `customer_id`, and the existing ordered 19 `FEATURE_COLUMNS`, with at most one row per customer/version/date. The current model interface supports those 19 feature definitions only; another feature design requires a separately versioned training/scoring change.

`FEATURE_REFRESH` consumes the materialization for the exact requested date and version and validates it before inference. It does not copy the undated Telco snapshot or invent historical features. The current snapshot schema has no `observation_date`; `--init-schema` deliberately does not change it or create empty production source tables. Genuine source integration, a dated feature-store schema migration, and point-in-time ETL remain prerequisites. The upstream ETL must atomically publish complete dated snapshots and prove that no event after the observation date contributes to a feature. The deployment runner cannot reconstruct that proof from a materialized row alone.

Active customers come only from real subscriptions where `status='active'`, `start_date <= observation_date`, and `end_date IS NULL OR end_date > observation_date`. Multiple active subscriptions count as one customer. All subscription rows are validated independently of cohort selection, so invalid rows cannot disappear behind the active filter. The runner never reads churn labels or the supplied `Churn` outcome.

## Commands

Install [requirements.txt](../requirements.txt) in the same Python/package environment recorded by the trusted production artifacts. Configure the process environment before starting Python; the application does not automatically load `.env`.

Create operational tables separately from scoring, using an account permitted to perform DDL:

```powershell
$env:APP_ENV = 'production'
.\venv\Scripts\python.exe -m src.deploy --init-schema
```

This ensures registry/prediction, `deployment_runs`/`production_state`, and Phase 8 cohort/outcome schemas exist. It is repeatable and does not approve any model. MySQL DDL is separate because it implicitly commits. Runtime execution requires the tables to exist and needs read access to source/features/artifacts plus the necessary select/insert/update/delete privileges on operational tables. Archival needs select/insert access to retained cohort tables; reporting needs read access, while the external label job owns outcome writes. Configure the URI in a protected process environment, never in a committed scheduler file.

Once all production prerequisites exist, run:

```powershell
$env:APP_ENV = 'production'
.\venv\Scripts\python.exe -m src.deploy --as-of YYYY-MM-DD
```

POSIX equivalent:

```sh
APP_ENV=production ./venv/bin/python -m src.deploy --as-of YYYY-MM-DD
```

Replace `YYYY-MM-DD` with the actual observation date published by ETL. If omitted, the runner uses the current UTC date. A missing exact-date snapshot fails; it does not select an older snapshot silently. Set the date explicitly for reproducible manual reruns.

The current repository cannot successfully execute a production scoring run. Production environment selection alone does not supply an approved model, genuine source data, dated features, or stakeholder acceptance.

## Configuration

All settings are centralized in [src/config.py](../src/config.py); [.env.example](../.env.example) contains placeholders and disabled scheduling defaults.

| Variable | Requirement and behavior |
| --- | --- |
| `DATABASE_URI` | Required MySQL connection URI, including the database. Use the pinned `mysql+mysqlconnector` driver. |
| `APP_ENV` | Defaults to `development`; deployment requires exactly `production`. Development and explicit benchmark scoring are separate from production. |
| `PRODUCTION_FEATURE_SET_VERSION` | Required nonempty version other than `telco_snapshot_v1`; must match the registry, report, preprocessing contract, and dated features. |
| `PRODUCTION_MODEL_ID` | Optional bootstrap selection when no active production state exists. Scheduled runs subsequently follow the active state. |
| `PRODUCTION_LOW_THRESHOLD` | Required accepted lower boundary of the medium tier. No benchmark-derived default. |
| `PRODUCTION_HIGH_THRESHOLD` | Required accepted lower boundary of the high tier; `0 < low < high <= 1`. Both values must exactly match the registry and accepted production report. |
| `PRODUCTION_OUTPUT_TABLE` | Only `model_predictions` is supported. Any other destination fails. |
| `PRODUCTION_PUBLICATION` | Only `atomic_replace` is supported. |
| `PRODUCTION_FEATURE_REFRESH` | Only `external` is supported: consume an already complete dated ETL materialization. |
| `DEPLOY_ALERT_MODE` | Only `logging` is built in. A real provider can implement the injected `AlertSink` interface. |
| `DEPLOY_SCHEDULE_ENABLED` | Defaults to `false`. `--scheduled` requires `true`. |
| `DEPLOY_SCHEDULE_CRON` | Empty by default. A scheduled invocation requires the configured cadence; the operator installs the corresponding cron entry. This variable does not start a scheduler. |

Benchmark scoring requires its explicit `--benchmark` option and rejects a production environment before opening a database connection. Production accepts neither `--benchmark` nor benchmark output. `APP_ENV=benchmark` identifies intentional benchmark use; no environment is inferred from the URI, machine, or model path.

## Preflight and model selection

Preflight rejects invalid/missing configuration, missing required source/operational tables or columns, unsupported publication settings, and missing approved production models. Selection follows this order:

1. Explicit `--model-id ID`, or the rollback target when `--rollback` is requested.
2. The model pinned by the singleton `production_state` row.
3. `PRODUCTION_MODEL_ID` for initial bootstrap, when configured.
4. The newest approved production model compatible with the configured feature version, ordered by `trained_at DESC, model_id DESC`.

Every selected target must have `approved=1`, `evaluation_scope=production`, a compatible non-snapshot feature version, and valid report/model/preprocessing artifacts. A selected target that fails validation stops the run; the runner never tries a benchmark or quietly switches to another artifact directory.

The registry pins `report.json` by SHA-256. The report pins `model.json` and `preprocessor.joblib`; all bytes are checked before deserialization. Production requires exact recorded versions of Python, XGBoost, scikit-learn, pandas, NumPy, and joblib, the expected ordered feature list, and matching model/preprocessing dimensions. Hashes establish integrity against the trusted registry; they are not digital signatures or evidence that arbitrary joblib files are safe. Only approved, trusted training outputs belong in this path.

A production report must agree with the registry's training timestamp, feature/label versions, dataset hash, and held-out metrics. It must declare `approved=true`, `evaluation_scope=production`, `evaluation_design=temporal`, and no promotion blockers. Its `production_acceptance` record must provide temporal evaluation, actual stakeholder approval, a positive `future_window_days`, an `evidence_reference`, and both accepted risk boundaries. This is an evidence contract for a future governed approval workflow; editing JSON flags does not establish evidence. Phase 4 training and Phase 5 registration continue to create unapproved benchmarks, and Phase 6 pending analysis does not satisfy this contract.

Feature validation rejects empty data, missing/duplicate customer rows, incomplete active-customer coverage, missing required predictors, invalid categories, nonfinite/out-of-domain numeric values, and unacceptable nulls. It reuses the existing rule: core fields are non-null, while `total_charges` may be null for at most 5% of rows because preprocessing explicitly imputes it. Extra inactive customers may exist in the dated materialization, but only the intended active cohort is scored.

## Prediction integrity, publication, and reruns

The runner validates every intended customer receives exactly one result, with no additional customers, a finite probability in `[0, 1]`, a valid tier consistent with the configured boundaries, the selected model/version, one populated UTC scoring timestamp, and the expected feature-snapshot hash. Publication reads the saved rows back and validates the batch again inside the transaction.

The singleton `production_state` row is locked first, then the chosen registry row, before source reads in a repeatable-read transaction. This extends Phase 5's registry serialization across model switches: two production deployments cannot publish competing active models. A waiting deployment reads state only after acquiring the lock. Operational/source tables must use transactional InnoDB storage.

The transaction replaces only rows with `scoring_mode='production'`, then commits the complete predictions, active model/run pointer, and successful run status together. Failed validation or publication rolls back all those changes and preserves the previous valid production batch. Benchmark rows remain available separately. Consumers should read production rows using a single query or a consistent-read transaction, never combine unrelated reads across a publication boundary.

Batch identity includes model/report identity, feature version, observation date, the sorted feature-snapshot hash, and accepted risk boundaries. An identical rerun validates and reuses the existing batch, including its original prediction timestamps; it still records a new execution attempt with `reused_batch=1`. A changed date, model, feature content, or policy produces a new batch. The primary key prevents duplicate customer/model/mode rows. Registry records and artifact files are never overwritten by deployment.

`model_predictions` contains the latest published production batch. `deployment_runs` preserves execution and model lineage, counts, timestamps, as-of date, feature/batch hashes, report/model/preprocessing hashes, previous run ID, rollback target, and terminal outcome. Phase 8 additionally retains the validated cohort metadata, all 19 predictors, scores, and tiers in `scoring_batches`/`scoring_history` inside the same publication transaction. Identical identities verify and reuse their first archive; archival failure blocks publication. Rerun `--init-schema` before deploying this upgrade. See [Phase 8 monitoring](phase_8_monitoring.md) for history and outcome contracts.

## Failure handling and alerts

Each attempt gets a run ID and structured logs. A database run row begins as `running`; completed attempts become `succeeded`, `failed`, or `rolled_back`. The selected model ID, feature version, training time, and artifact hashes are recorded. Failures return a nonzero exit code, record the stage and error summary, and invoke the alert sink.

| Stage | Failure examples and effect |
| --- | --- |
| `PREFLIGHT` | Configuration, schema, approval, feature-version, or selected-model prerequisites fail; no scores are published. |
| `DATA_VALIDATION` | Invalid raw sources or active cohort; previous production state is preserved. |
| `FEATURE_REFRESH` | Missing dated materialization, incomplete coverage, duplicate/invalid features; inference does not publish. |
| `MODEL_LOAD` | Missing/tampered report/model/preprocessing, incompatible dependencies, missing production acceptance; previous batch survives. |
| `SCORING` | Preprocessing or inference fails; transaction rolls back. |
| `PREDICTION_VALIDATION` | Missing, duplicate, unexpected, invalid, or mismatched predictions; publication is blocked. |
| `PUBLICATION` | Database write/readback/commit failure; no partial batch is committed. |

`LoggingAlertSink` writes a structured local alert for deployment failures (including preflight, data quality, scoring, and publication) and rollback events. `AlertSink` is the extension point for a real provider; this phase sends no Slack, email, or other external notification. Production operators must connect and monitor that provider or collect the structured logs before enabling unattended execution.

If the database or run table is unavailable, the runner cannot truthfully persist a failed row there: it emits the run ID, failure stage, and error to logs and the alert sink and exits unsuccessfully. An uncatchable process/host termination may leave a durable `running` attempt; the database transaction still protects the previous publication. Investigate stale attempts against scheduler/process logs and the active state before any manual reconciliation. A stale attempt must not be marked successful merely to clear it.

## Model switch and rollback

Switch to an already approved production registry model explicitly:

```powershell
.\venv\Scripts\python.exe -m src.deploy --model-id APPROVED_MODEL_ID --as-of YYYY-MM-DD
```

This runs the full preflight and batch pipeline. It changes the active pointer only after successful publication; it does not grant approval. No deployment command alters historical registry metadata or the benchmark approval CHECK constraint.

Inspect the current pointer and successful lineage:

```sql
SELECT s.model_id, s.run_id, r.previous_run_id, r.observation_date,
       r.status, r.completed_at
FROM production_state s
JOIN deployment_runs r ON r.run_id = s.run_id
WHERE s.state_id = 1;

SELECT run_id, previous_run_id, model_id, status, observation_date,
       completed_at, rollback_target_run_id
FROM deployment_runs
WHERE status IN ('succeeded', 'rolled_back')
ORDER BY completed_at DESC, run_id DESC;
```

Restore the previous distinct model in the successful active-state lineage:

```powershell
.\venv\Scripts\python.exe -m src.deploy --rollback --as-of YYYY-MM-DD
```

`--rollback` and `--model-id` are mutually exclusive. Rollback skips same-model reruns, revalidates the prior model's current approval, compatibility, report and artifacts, checks current dated features, and re-scores the current intended cohort. Configure the accepted thresholds to match the rollback target before invoking it. Missing, revoked, incompatible, or tampered targets fail without changing active state. A successful rollback records a new `rolled_back` event, target run ID, hashes, and predictions, then pins the restored model for subsequent scheduled runs. Old runs retain their original statuses. Rollback does not copy stale customer lists or delete artifacts/history.

## Scheduling

The supplied [ops/deploy-cron.sh](../ops/deploy-cron.sh) is a POSIX cron entry point. It changes to `CHURN_PROJECT_DIR` (default `/opt/churn-analytics`), exports values from `CHURN_DEPLOY_ENV_FILE` (default `/etc/churn/deploy.env`), and executes `./venv/bin/python -m src.deploy --scheduled`. Invoke it with `/bin/sh`, so no executable-bit change is required. Its environment file is trusted shell configuration and must be readable only by the service account and administrators. Use absolute paths; prepare the matching virtual environment and trusted artifacts at their registry locations.

Scheduling is **disabled by default**, with an empty cadence. After production gates and log/alert collection are ready, the accountable owner supplies the cadence and timezone, sets `DEPLOY_SCHEDULE_ENABLED=true`, and sets `DEPLOY_SCHEDULE_CRON` to that approved five-field cron expression. Source the protected environment file in an administrative shell, then print the proposed entry for review:

```sh
printf '%s %s\n' "$DEPLOY_SCHEDULE_CRON" 'CHURN_PROJECT_DIR=/opt/churn-analytics CHURN_DEPLOY_ENV_FILE=/etc/churn/deploy.env /bin/sh /opt/churn-analytics/ops/deploy-cron.sh >> /var/log/churn/deploy.log 2>&1'
```

Install that line with `crontab -e` for the service account. Create the log directory with appropriate ownership and configure log rotation. Keep the cron expression in the installed entry and environment file synchronized; changing the variable alone does not update cron. Confirm the host scheduler's timezone against the chosen business cadence and UTC observation-date semantics. The upstream ETL must complete before the scheduled run for that date. Manual historical/backfill runs supply `--as-of` explicitly.

No cadence, scheduler installation, scheduled production execution, or external alert delivery is claimed by this implementation. The repository's existing Windows workflow may instead use Task Scheduler to run the same production command with a configured working directory/process environment; the included runnable scheduler wrapper targets POSIX cron.

## Benchmark restriction and remaining gates

The Telco snapshot lacks observation dates, dated cancellation events, genuine future churn windows, a verified active-customer source, historical event streams for point-in-time features, temporal evaluation, and production business acceptance. It must remain `evaluation_scope=benchmark`, `approved=0`, and `feature_set_version=telco_snapshot_v1`. The database continues to forbid approving a benchmark row. Renaming a version or setting `APP_ENV=production` does not provide the missing evidence.

Production activation still needs real source integration and temporal ETL/schema, a governed future-window label definition and evaluation/training workflow, stakeholder metric/capacity/cost/tier/cadence decisions and sign-off, an approved compatible registry/artifact package, and operational ownership/alerts. Address the outstanding [foundation review findings](phase_0_3_review.md) before production use. This phase supplies deployment machinery and safeguards, not a production model or fabricated successful deployment.

## Verification

Run the local regression suite without a production database:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Deployment tests cover configuration and benchmark gates, deterministic model selection, artifact/preprocessing integrity, feature domains/coverage, complete prediction batches, atomic replacement and failure rollback, deterministic reruns, rollback target verification/history, failure stages, alert invocation, and explicit database URI routing. Invalid numeric and source-date parsing errors omit the rejected cell values from deployment logs and alerts. Existing Phase 0–6 regressions remain part of the same suite.

The isolated MySQL test requires a configured `DATABASE_URI` and an account allowed to create/drop databases:

```powershell
$env:RUN_DEPLOY_MYSQL_TESTS = '1'
.\venv\Scripts\python.exe -m unittest tests.test_deploy_mysql -v
Remove-Item Env:RUN_DEPLOY_MYSQL_TESTS
```

It creates a unique `churn_deploy_test_<uuid>` database and removes only that fixture database during cleanup; it does not modify the application database named in the URI. It executes the real InnoDB schemas, source reads, publication transactions, reruns, failure rollback, and model rollback. Artifact loading and inference are stubbed with synthetic outputs; the separate artifact tests verify the trusted-artifact boundary. These fixtures are test evidence, not production approval. Concurrent workers, scale/SLA, installed cron, and external alert delivery are not verified. Consult [memory.md](../memory.md) for actual results.
