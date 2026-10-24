# Phase 8 — Retained cohorts and monitoring

Phase 8 retains production scoring evidence and reports feature drift, matured-cohort performance, and configured retraining recommendations. The Telco snapshot remains an unapproved benchmark; it has no dated production cohorts or future-window outcomes to monitor.

## History and publication

`src.deploy` archives the validated active cohort in the same transaction as predictions, active state, and successful run metadata. Missing archive tables or an archive failure prevents publication and preserves the previous batch. Schema creation remains a separate operation because MySQL DDL implicitly commits.

| Table | Retained evidence |
| --- | --- |
| `scoring_batches` | Batch identity, first successful run, model/feature/label versions, observation and scoring dates, accepted horizon and tiers, source/report hashes, training date, and held-out baselines |
| `scoring_history` | One row per batch/customer, containing the 19 validated predictors, probability, and tier |
| `monitoring_outcomes` | External, versioned future-window outcomes for the exact retained cohort, with window end, observed-through date, availability time, and evidence reference |

An identical batch reuses its archive. Revisiting an earlier batch identity retains its original history and verifies the features, probabilities, policy, and metadata rather than overwriting them. The latest production prediction table can change without removing prior cohort history. These tables do not reference the replaceable raw snapshot.

Upgrade operational and monitoring schemas before the next production run:

```powershell
.\venv\Scripts\python.exe -m src.deploy --init-schema
```

When operational tables already exist, `python -m src.monitor --init-schema` creates only the three monitoring tables. Neither command approves a model or manufactures production source data. Existing latest-only scores cannot be backfilled into genuine dated history from the undated CSV.

## Reports

With genuine retained production cohorts available:

```powershell
.\venv\Scripts\python.exe -m src.monitor --as-of YYYY-MM-DD
```

The default selects the latest cohort available by that UTC analysis date and its nearest earlier cohort with the same feature version. For a specific comparison, including a week-over-week comparison:

```powershell
.\venv\Scripts\python.exe -m src.monitor --batch-id CURRENT_BATCH_SHA256 --reference-batch-id EARLIER_BATCH_SHA256 --as-of YYYY-MM-DD --output-dir models/phase8_monitoring
```

Replace the placeholders with real retained batch hashes and an analysis date no later than today. The reference must have the same feature version and an earlier observation date. Missing history fails with a nonzero exit status; a missing earlier reference leaves drift pending. Reports read a consistent database snapshot and never join current raw/features, snapshot labels, or the latest prediction table.

The output directory contains deterministic `report.json`, `report.md`, and an escaped, standalone `dashboard.html`. JSON preserves the exact statistics, source identities, policy, pending reasons, and an analysis digest. Open the HTML file locally; no server or new dependency is required. Reports aggregate customer evidence rather than exporting an outreach list.

Numeric drift reports each feature's mean, population standard deviation, null rate, and absolute mean change divided by the reference standard deviation. An unchanged constant reference has shift zero; a changed constant reference is explicitly flagged because its normalized shift has no finite value. Categorical drift reports total variation: half the sum of absolute differences between category proportions. These are descriptive distribution changes, not proof of degradation or causation. Cohort membership may differ between dates.

## Realized outcomes

The upstream label job must publish genuine outcomes into `monitoring_outcomes`. Its primary key is `(batch_sha256, customer_id)` and its foreign key identifies an archived customer. Required fields are:

- `label_version`: exactly the batch's accepted label definition.
- `label_window_end`: observation date plus the archived `future_window_days`.
- `observed_through`: at least the complete window end.
- `available_at`: when the completed observation became available, no earlier than the observed-through date.
- `churned`: verified binary outcome, including explicit zero outcomes.
- `evidence_reference`: a nonblank reference to the real source/label job.

The monitor rejects incompatible labels, windows, invalid values, missing evidence, and unexpected/duplicate customers. Only outcomes available by the analysis date are used. Performance remains pending until the full horizon has matured and every archived customer has a usable outcome. Partial cohorts are not evaluated, and missing outcomes are never assumed to mean retention.

Once complete, the report computes precision/recall at the archived accepted high-risk threshold, average precision, ROC-AUC, Brier score, and deterministic top-decile precision/recall. Undefined metrics for cohorts with no positives or only one class are explicit nulls. Precision/recall differences from the held-out baseline are reported only when the archived evaluation threshold matches the operating threshold. Average precision can be compared independently of the threshold; differences between historical and realized cohorts still require interpretation.

## Policy, alerts, and retraining

`MONITORING` in [src/config.py](../src/config.py) leaves four stakeholder policies unresolved:

| Setting | Meaning when supplied |
| --- | --- |
| `numeric_mean_shift_max` | Maximum accepted normalized numeric mean shift; a changed constant reference also triggers review |
| `categorical_tv_max` | Maximum accepted categorical total variation in `[0, 1]` |
| `precision_min` | Minimum accepted realized precision at the archived high-risk boundary |
| `retrain_interval_days` | Positive interval measured from the archived training date |

All default to `None`. Configured breaches or an elapsed interval produce a retraining recommendation and a structured local alert through the existing injectable alert interface. Operational or data-quality failures also emit structured alerts and return unsuccessfully. Alert-provider failures are logged without concealing the original result. No email or external message is sent by this implementation.

Recommendations do not execute training, change registry approval, or select a production model. The only implemented trainer remains the explicit snapshot benchmark. Genuine temporal retraining requires a dated training/label pipeline and governed acceptance workflow. After those exist, an owner can schedule this monitor after successful cohort publication and route recommendations to that workflow. Cadence, threshold values, scheduler installation, delivery ownership, and automatic temporal retraining are not invented here.

## Verification and remaining work

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
$env:RUN_MONITOR_MYSQL_TESTS = '1'
.\venv\Scripts\python.exe -m unittest tests.test_monitor_mysql -v
Remove-Item Env:RUN_MONITOR_MYSQL_TESTS
```

The opt-in MySQL test requires `DATABASE_URI` with create/drop-database privileges. It creates and removes only its own UUID database, uses synthetic dated customers and outcomes, stubs artifact loading/inference, and executes real schema, source, archive, publication, rollback, and monitoring SQL. `TEST_MONITOR_OUTPUT_DIR` optionally retains clearly labelled synthetic report artifacts. Application data is not refreshed or rescored by this fixture. Actual commands and results are recorded in [memory.md](../memory.md).

Production source integration, stakeholder sign-off, operational scheduling/alerts, scale/concurrency testing, and the outstanding foundation findings remain prerequisites for activation. History growth and access to retained customer features need an owner-approved retention/access policy; this implementation does not delete historical evidence automatically.

Candidate v2 features remain a plan until real event sources and point-in-time definitions exist:

| Candidate | Required source and validation |
| --- | --- |
| 7/30/90-day engagement counts and trends | Dated usage events; prove all contributing events precede the observation cutoff |
| Payment failures and plan changes | Dated billing/subscription history; define event semantics and financial boundaries |
| Open tickets and resolution/CSAT summaries | Dated support records; exclude resolutions or survey answers learned after the cutoff |

Each new feature set needs a separate version, reproducible temporal training/evaluation, and a compatible scoring contract before deployment.
