# Customer Churn Analytics

### From a Telco customer snapshot to validated MySQL labels and features

A Python + MySQL churn project developed against a [spec-driven plan](churn_prediction_spec_driven_plan.md). It includes ingestion, labels, 19 predictors, an evaluated XGBoost snapshot benchmark, and MySQL model registration and benchmark scoring.

**Current scope:** through Phase 6 analysis (benchmark only; business acceptance pending) · **Dataset:** Telco customer snapshot · **Version:** `telco_snapshot_v1`

[Quick start](#quick-start) · [Pipeline](#explore-the-pipeline) · [Data dictionary](#data-dictionary) · [Verification](#verify-results) · [Review](#review-findings) · [Roadmap](#roadmap)

> **Snapshot benchmark:** The CSV contains a supplied `Churn` outcome, but no observation or cancellation dates. These tables cannot establish a future prediction horizon, temporal leakage safety, or a time-based holdout.

## At a glance

Last read-only database verification: **2026-09-24**, MySQL **8.0.46**.

| Customers | Churned | Retained | Predictors | Missing total charges |
| ---: | ---: | ---: | ---: | ---: |
| 7,043 | 1,869 | 5,174 | 19 | 11 |

All 7,043 customers join across raw data, labels, and features. The audit found **zero label mismatches** and **zero mismatches across the 19 copied feature values**. These are dataset checks, not model performance metrics.

<details>
<summary><strong>Choose your path through the repository</strong></summary>

| I want to… | Start here |
| --- | --- |
| Run the current pipeline | [Quick start](#quick-start) |
| Validate the CSV without MySQL | [Local checks](#local-checks) |
| Understand the tables and fields | [Data dictionary](#data-dictionary) |
| Query the current snapshot | [Verify results](#verify-results) |
| Review correctness and remaining gaps | [Phase 0–3 review](docs/phase_0_3_review.md) |
| See implementation history | [Project memory](memory.md) |
| Understand planned modeling and acceptance gates | [SDD plan](churn_prediction_spec_driven_plan.md) |

</details>

## Quick start

These commands use **Windows PowerShell**. The reviewed environment runs **Python 3.13.3**, pandas **3.0.0**, SQLAlchemy **2.0.46**, and mysql-connector-python **9.5.0**.

### 1. Set up Python

From the repository root, create the environment if it does not already exist:

```powershell
py -3.13 -m venv venv
```

Install the pinned dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

<details>
<summary>Already have a working environment?</summary>

Use its Python executable for every command. These examples invoke `venv\Scripts\python.exe` directly, so PowerShell activation is optional. If `py` is unavailable, use your installed Python 3.13 executable to create the environment.

</details>

### 2. Prepare MySQL

Use the existing `customer_churn` database, or create it from a MySQL client with an account authorized to create databases:

```sql
CREATE DATABASE IF NOT EXISTS customer_churn;
```

The pipeline creates its three tables automatically. The configured account needs permission to create those tables and read, delete, and insert their rows.

### 3. Configure the connection

Set the variable in the **same PowerShell session** used to launch Python. Replace the placeholders with your connection details:

```powershell
$env:DATABASE_URI = 'mysql+mysqlconnector://USER:URL_ENCODED_PASSWORD@localhost:3306/customer_churn'
```

The application reads `DATABASE_URI` from the process environment at import time. **It does not automatically load `.env`.** The supported driver in [requirements.txt](requirements.txt) is `mysql+mysqlconnector`.

<details>
<summary>Credentials and special characters</summary>

URL-encode reserved characters in the username or password when constructing a connection URI. Keep real credentials out of issues, screenshots, and committed files. `.env` is excluded by [.gitignore](.gitignore), but the file alone does not configure the Python process.

The helper now supports the PyMySQL timeout keyword too. PyMySQL is installed locally but is not pinned; new installations should use the documented `mysql+mysqlconnector` driver.

</details>

### 4. Validate, then load

Validate the CSV first; this command does not connect to MySQL:

```powershell
.\venv\Scripts\python.exe -m src.ingest --validate-only
```

> **Full snapshot replacement:** A pipeline run deletes and reloads `customers_raw`. Foreign-key cascades remove existing labels and features, including other versions, before later stages rebuild the current version. The stages commit separately; a later failure can leave derived tables empty. See [R1 in the review](docs/phase_0_3_review.md#r1-high-refresh-is-not-atomic-across-stages).

Run against the intended development database:

```powershell
.\venv\Scripts\python.exe run_pipeline.py
```

Expected output for the included CSV:

```text
Loaded 7,043 customer rows
Materialized 7,043 labels (1,869 churned)
Materialized 7,043 feature rows
```

## Explore the pipeline

```mermaid
flowchart LR
    CSV[Telco CSV] --> INGEST[Validate and normalize]
    INGEST --> RAW[(customers_raw)]
    RAW --> LABELS[(churn_labels)]
    RAW --> FEATURES[(feature_store)]
    LABELS --> TRAIN[XGBoost snapshot benchmark]
    FEATURES --> TRAIN
    TRAIN --> REGISTRY[(model_registry)]
    REGISTRY --> SCORE[Explicit benchmark scoring]
    FEATURES --> SCORE
    SCORE --> PREDICTIONS[(model_predictions)]
```

The orchestrator runs **ingestion → labeling → features**. Both derived tables read the raw table. Each materializer uses its own transaction.

<details>
<summary><strong>Phase 1 · Ingest the customer snapshot</strong></summary>

[src/ingest.py](src/ingest.py) checks CSV column order, nonempty input, customer IDs, binary fields, nonnegative numeric values, and core-column blank rates. It strips whitespace, maps Yes/No fields to 1/0, and preserves blank total charges as SQL `NULL`.

[sql/01_create_tables.sql](sql/01_create_tables.sql) defines the 22-column raw table: customer ID, 19 predictors, churn outcome, and ingestion timestamp. The raw delete and insert run in one transaction after the schema statement.

Validation is incomplete for malformed inputs; see the [review findings](docs/phase_0_3_review.md).

</details>

<details>
<summary><strong>Phase 2 · Materialize the supplied outcome</strong></summary>

[src/labels.py](src/labels.py) copies `customers_raw.churned` into [churn_labels](sql/02_create_labels.sql) using label version `telco_snapshot_v1` and source `source_churn_column`.

It checks label count against customer count and requires both retained and churned classes. `observation_date` and `label_window_end` remain `NULL` because the source provides neither date.

</details>

<details>
<summary><strong>Phase 3 · Materialize the predictors</strong></summary>

[src/features.py](src/features.py) explicitly selects 19 predictors into [feature_store](sql/03_create_feature_store.sql), using feature version `telco_snapshot_v1`.

Customer ID is stored for joins but excluded from `FEATURE_COLUMNS`. The target and ingestion timestamp are also excluded. The materializer checks customer coverage and rejects a `total_charges` null rate above 5%. Current features are copies of snapshot attributes; dated rolling-window features have not been implemented.

</details>

## Data dictionary

| Table | Primary key | Purpose | Current rows |
| --- | --- | --- | ---: |
| `customers_raw` | `customer_id` | Validated snapshot plus supplied outcome | 7,043 |
| `churn_labels` | `customer_id`, `label_version` | Versioned snapshot outcome and provenance | 7,043 |
| `feature_store` | `customer_id`, `feature_set_version` | Versioned predictor values | 7,043 |
| `model_registry` | `model_id` | Unapproved benchmark, metrics, artifact identity | 1 |
| `model_predictions` | `customer_id`, `model_id`, `scoring_mode` | Latest benchmark scores per model | 7,043 |
| `business_validation` | `validation_id`; unique analysis hash + N | Pending business-review scenarios | 4 |

<details>
<summary><strong>Expand all 19 predictor definitions</strong></summary>

| Predictor | Source column | Meaning / representation |
| --- | --- | --- |
| `gender` | `gender` | Supplied gender category |
| `senior_citizen` | `SeniorCitizen` | Supplied senior-citizen indicator, 0/1 |
| `partner` | `Partner` | Partner indicator, No/Yes → 0/1 |
| `dependents` | `Dependents` | Dependents indicator, No/Yes → 0/1 |
| `tenure_months` | `tenure` | Supplied tenure in months |
| `phone_service` | `PhoneService` | Phone subscription, No/Yes → 0/1 |
| `multiple_lines` | `MultipleLines` | Multiple-line service category |
| `internet_service` | `InternetService` | Internet service category |
| `online_security` | `OnlineSecurity` | Security service category |
| `online_backup` | `OnlineBackup` | Backup service category |
| `device_protection` | `DeviceProtection` | Device protection category |
| `tech_support` | `TechSupport` | Technical support category |
| `streaming_tv` | `StreamingTV` | TV streaming category |
| `streaming_movies` | `StreamingMovies` | Movie streaming category |
| `contract` | `Contract` | Contract term category |
| `paperless_billing` | `PaperlessBilling` | Paperless billing, No/Yes → 0/1 |
| `payment_method` | `PaymentMethod` | Payment method category |
| `monthly_charges` | `MonthlyCharges` | Monthly charge, `DECIMAL(10,2)` in MySQL |
| `total_charges` | `TotalCharges` | Total charge, nullable `DECIMAL(10,2)` |

The 11 missing total-charge values are preserved in MySQL. Training uses train-only median imputation and one-hot encoding. Snapshot attributes are not proven to precede the churn outcome.

</details>

## Verify results

### Local checks

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m src.ingest --validate-only
```

Current result: **14 local tests passed**; both MySQL tests are opt-in. Phase 6's MySQL integration passed in this iteration; Phase 5's passed in its recorded iteration. Coverage includes business scenario formulas, deterministic ranking, pending approval, stale/mismatched sources, artifact tampering, tier boundaries, production refusal, training-to-registry ordering, and prediction rollback/model isolation. Raw ingestion rollback and all invalid-input cases are not covered.

The live integration check registers and scores the trusted benchmark in the configured development database. It verifies reruns and rollback, and leaves completed benchmark scores available:

```powershell
$env:RUN_MYSQL_TESTS = '1'
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_phase5_mysql.py -v
Remove-Item Env:RUN_MYSQL_TESTS
```

### Database checks

Open `customer_churn` in your MySQL client and expand the checks below. These queries are read-only.

<details>
<summary><strong>Check row counts and class distribution</strong></summary>

```sql
USE customer_churn;

SELECT 'customers_raw' AS table_name, COUNT(*) AS row_count FROM customers_raw
UNION ALL
SELECT 'churn_labels', COUNT(*) FROM churn_labels
UNION ALL
SELECT 'feature_store', COUNT(*) FROM feature_store;

SELECT churned, COUNT(*) AS customers
FROM churn_labels
WHERE label_version = 'telco_snapshot_v1'
GROUP BY churned;
```

Expected: 7,043 rows per table; 5,174 retained (`0`) and 1,869 churned (`1`). Counts describe the current single-version snapshot.

</details>

<details>
<summary><strong>Check versioned joins and missing values</strong></summary>

```sql
SELECT COUNT(*) AS joined_customers
FROM feature_store AS f
JOIN churn_labels AS l ON l.customer_id = f.customer_id
WHERE f.feature_set_version = 'telco_snapshot_v1'
  AND l.label_version = 'telco_snapshot_v1';

SELECT SUM(total_charges IS NULL) AS missing_total_charges
FROM feature_store
WHERE feature_set_version = 'telco_snapshot_v1';

SELECT COUNT(*) AS labels_with_dates
FROM churn_labels
WHERE label_version = 'telco_snapshot_v1'
  AND (observation_date IS NOT NULL OR label_window_end IS NOT NULL);
```

Expected: 7,043 joined customers, 11 missing total charges, and 0 labels with dates.

</details>

## Review findings

The [Phase 0–3 review](docs/phase_0_3_review.md) includes evidence, source locations, impact, and proposed fixes. Existing tests pass, but additional probes reproduced validation and configuration failures.

| Priority | Finding | Status |
| --- | --- | --- |
| High | A refresh commits raw data before rebuilding cascaded labels/features | Open |
| High | Fractional tenure is truncated; infinity and out-of-range values pass validation | Open |
| High | Local PyMySQL configuration fails in the application connection helper | Fixed and tested in Phase 4; `.env` loading remains explicit |
| Medium | Invalid categories, blanks, and overlong IDs can pass CSV validation | Open |
| Medium | Empty feature input is accepted as a successful materialization | Open |
| Medium | The label date constraint permits a partially null date pair | Open |

The earlier statement “label leakage: 0” only established that `churned` was absent from the feature table. It did **not** establish temporal leakage safety.

## Troubleshooting

<details>
<summary><code>DATABASE_URI is required</code></summary>

Set `$env:DATABASE_URI` in the same terminal before starting Python. A `.env` file is not automatically loaded. Restart an already-running Python process after changing the variable.

</details>

<details>
<summary><code>unexpected keyword argument 'connection_timeout'</code></summary>

The current helper chooses the timeout keyword by driver. Update to current code or use the pinned `mysql+mysqlconnector://` scheme in [Quick start](#quick-start).

</details>

<details>
<summary><code>Unknown database 'customer_churn'</code> or access denied</summary>

Check the server, port, database spelling, credentials, and permissions. The application creates tables, not the database. Create the database on the server targeted by the URI before running the pipeline.

</details>

<details>
<summary>A pipeline stage failed after ingestion</summary>

Inspect row counts in all three tables before using their contents. Raw ingestion cascades deletion into both derived tables and commits before later stages run. After addressing the error, rerun the complete pipeline against the intended snapshot. Do not assume a failure preserved the previous full dataset across all three tables.

</details>

## Roadmap

| Phase | Current state | Remaining boundary |
| --- | --- | --- |
| 0 · Setup | Environment verified; modeling dependencies pinned | `.env` loading remains explicit |
| 1 · Data foundation | Included CSV ingested and checked | Review findings; broader event sources unavailable |
| 2 · Labeling | Supplied snapshot outcome materialized | Dated churn events and agreed future-window definition |
| 3 · Features | 19 snapshot predictors materialized | Review findings; temporal features and leakage validation |
| 4 · Modeling | XGBoost snapshot benchmark trained and evaluated | Temporal data and stakeholder acceptance gates |
| 5 · Registry and scoring | Unapproved model registered; 7,043 benchmark scores stored | Approved temporal model, active-customer source, business tiers |
| 6 · Business validation | Reproducible report, ranked customers, scenario analysis, pending MySQL records | Stakeholder thresholds, capacity, costs, cadence, and sign-off remain unresolved |
| 7 · Deployment | Not started | Scheduling, alerting, and rollback validation |
| 8 · Monitoring | Not started | Realized outcomes, drift metrics, retraining policy |

Phase 4 follows the snapshot amendment in SDD Section 5.2. Production temporal requirements remain unmet. See the [evaluation report](docs/phase_4_evaluation.md).

### Run the XGBoost benchmark

With `DATABASE_URI` configured and Phase 1–3 tables populated:

```powershell
.\venv\Scripts\python.exe -m src.train
```

This reads MySQL without refreshing the input tables. Optional `--trials 8` and `--output-dir models/my_run` control candidate count and a new output directory; existing directories are rejected. Add `--register` to register the completed evaluation as an unapproved benchmark.

<details>
<summary><strong>Phase 4 results and artifacts</strong></summary>

Test average precision **0.6588** (prevalence baseline **0.2654**), ROC-AUC **0.8454**, precision **0.5899**, recall **0.7193**, Brier **0.1642**. These describe a non-temporal benchmark, not a future churn forecast.

Runs save `model.json`, `preprocessor.joblib`, `report.json`, and `splits.csv`. Run directories are ignored by Git; the evaluation report is versioned. Only load trusted joblib artifacts. The threshold maximizes validation F1 and is not a business outreach rule. Registration is optional and never approves the benchmark.

</details>

### Review business-validation scenarios

```powershell
.\venv\Scripts\python.exe -m src.business_validation --model-id xgb_9ee0132da5749c7e72124a8c --save
```

This writes a Markdown report, exact JSON, and ranked customer CSV under `models/phase6_business_validation/`, and saves four pending outreach scenarios in `business_validation`. Omit `--save` for database reads only; use repeatable `--top-k N` for additional scenarios and `--report-path PATH.md` for another output location.

The report distinguishes saved held-out metrics from full-snapshot outreach/tier statistics, which include training customers. Candidate business criteria live in `BUSINESS_ACCEPTANCE` in `src/config.py`; unresolved values are `None`. Configured criteria never approve the benchmark. See the [Phase 6 workflow and validation evidence](docs/phase_6_business_validation.md) for formulas, persistence, stakeholder decisions, and blockers before Phase 7.

### Register and score the benchmark

With `DATABASE_URI` configured, register an existing trusted training output:

```powershell
.\venv\Scripts\python.exe -m src.registry --artifact-dir models/phase4_benchmark_verified
```

Use the returned model ID. The verified Phase 4 model's ID is `xgb_9ee0132da5749c7e72124a8c`:

```powershell
.\venv\Scripts\python.exe -m src.score --benchmark --model-id xgb_9ee0132da5749c7e72124a8c
```

This stores 7,043 labeled benchmark scores in MySQL. `python -m src.score` without benchmark mode refuses production scoring. The source has no verified active-customer population, and the model remains `approved=0`. See [Phase 5 validation](docs/phase_5_registry_scoring.md).

<details>
<summary><strong>Inspect the registry and benchmark scores</strong></summary>

```sql
SELECT model_id, trained_at, evaluation_scope, approved, pr_auc,
       low_threshold, high_threshold, artifact_path
FROM model_registry ORDER BY trained_at DESC;

SELECT scoring_mode, risk_tier, COUNT(*) AS customers
FROM model_predictions
WHERE model_id = 'xgb_9ee0132da5749c7e72124a8c'
GROUP BY scoring_mode, risk_tier;

SELECT customer_id, churn_probability, risk_tier, scored_at
FROM model_predictions
WHERE model_id = 'xgb_9ee0132da5749c7e72124a8c' AND scoring_mode = 'benchmark'
ORDER BY churn_probability DESC, customer_id LIMIT 20;
```

Provisional risk policy: low below 0.295687, medium from there to below 0.591374, high at or above 0.591374 (displayed boundaries rounded). These tiers are not approved outreach decisions. Full-snapshot scores include training customers and must not be used as new holdout evaluation.

Reruns replace only that model's benchmark scores atomically. A failure preserves its previous scores. Source-table refreshes do not delete predictions: use score time and `feature_snapshot_sha256` to identify the saved batch, and rerun scoring after a refresh. The application does not retain every historical scoring run.

</details>

<details>
<summary><strong>Repository map and specification precedence</strong></summary>

```text
churn-analytics/
├── data/raw/                       Included Telco CSV
├── sql/
│   ├── 01_create_tables.sql        Raw table
│   ├── 02_create_labels.sql        Snapshot labels
│   ├── 03_create_feature_store.sql Snapshot predictors
│   ├── 04_create_model_registry.sql Model audit records
│   └── 05_create_model_predictions.sql Scored benchmark rows
├── src/
│   ├── config.py                  Paths, versions, environment lookup
│   ├── utils.py                   SQLAlchemy engine helper
│   ├── ingest.py                  CSV validation and replacement load
│   ├── labels.py                  Label materialization
│   ├── features.py                Feature materialization
│   ├── train.py                   XGBoost benchmark and evaluation
│   ├── registry.py                Artifact verification and registration
│   └── score.py                   Gated snapshot batch scoring
├── tests/                         Local regressions and opt-in MySQL integration
├── docs/phase_0_3_review.md         Review and evidence
├── run_pipeline.py                Phase 1–3 orchestration
├── requirements.txt               Pinned runtime dependencies
├── churn_prediction_spec_driven_plan.md
└── memory.md                      Implementation history
```

The user-directed [SDD Markdown plan](churn_prediction_spec_driven_plan.md) guides this work. [Copilot_context](Copilot_context) now follows the implemented SDD table names and PR-AUC metric. The bundled PDF has not been updated alongside the Markdown snapshot amendments.

</details>

[Back to top](#customer-churn-analytics)
