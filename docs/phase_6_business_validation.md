# Phase 6: business validation

Implemented and validated on 2026-09-25 against the existing MySQL benchmark. The analysis layer is complete; actual business acceptance remains **pending**. Model `xgb_9ee0132da5749c7e72124a8c` remains `approved=0` and `evaluation_scope='benchmark'`.

## Run the analysis

Configure `DATABASE_URI` in the process environment as described in the README. The application does not automatically load `.env`.

```powershell
.\venv\Scripts\python.exe -m src.business_validation --model-id xgb_9ee0132da5749c7e72124a8c --save
```

Default output is `models/phase6_business_validation/report.md`, with adjacent `report.json` and `report.ranked.csv`. These local artifacts are ignored by Git, as are the model run directories. The Markdown report presents the evidence; JSON preserves exact values and source hashes; CSV answers which customers rank highest, with all 7,043 ranks, scores, provisional tiers, and supplied labels. It is not an approved contact list. Retain the three artifacts together for review.

Additional scenarios can be requested without choosing a stakeholder capacity:

```powershell
.\venv\Scripts\python.exe -m src.business_validation --model-id xgb_9ee0132da5749c7e72124a8c --feature-set-version telco_snapshot_v1 --top-k 100 --top-k 500 --report-path models/phase6_custom/report.md
```

`--top-k` is a positive integer no greater than the population and can be repeated. Default scenarios remain top 1%, 5%, 10%, and 20%, rounding customer counts up. Rank is score descending, then customer ID ascending for ties. Selection uses exactly N rows. The last selected score is recorded as the cutoff; a score threshold alone can include extra tied customers. None of these capacities is a stakeholder choice.

Omit `--save` for database reads only. `--save` creates the Phase 6 table separately from the insert transaction, then records pending scenarios. It never updates the registry, scores, labels, features, or raw customers.

## Evidence and interpretation

The pinned Phase 4 report supplies test average precision 0.658753298, ROC-AUC 0.845401586, precision 0.589912281, recall 0.719251337, F1 0.648192771, recall@top-decile 0.294117647, and Brier score 0.164171249. Test prevalence is 0.265436480. Split sizes are 4,225 training, 1,409 validation, and 1,409 test customers. These metrics are reused, not recomputed.

The following analysis uses **all snapshot predictions, including training customers**, matched to supplied labels. Its precision/recall values are descriptive and are not a new holdout evaluation or evidence of future outreach effectiveness.

| Scenario | Selected | Supplied churners captured | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| Top 1% | 71 | 67 | 0.943662 | 0.035848 |
| Top 5% | 353 | 309 | 0.875354 | 0.165329 |
| Top 10% | 705 | 550 | 0.780142 | 0.294275 |
| Top 20% | 1,409 | 969 | 0.687722 | 0.518459 |

Precision is selected churners / selected customers, also the selected prevalence. Recall is selected churners / all 1,869 churners, also cumulative churn capture. Larger lists cannot reduce cumulative captured churners, but precision can rise or fall. Business capacity and outreach costs must inform the eventual choice; neither is inferred here.

### Provisional benchmark tiers

High begins at the validation-F1 threshold; medium begins at half that value. Low has lower bound zero. The Phase 5 registry column `low_threshold` denotes the **medium tier's lower bound**, not a separate additional cutoff.

| Tier | Interval | Customers | % Population | Churners | Churn rate |
| --- | --- | ---: | ---: | ---: | ---: |
| High | [0.59137362241745, 1] | 2,320 | 32.940508% | 1,366 | 0.588793 |
| Medium | [0.295686811208725, 0.59137362241745) | 1,816 | 25.784467% | 389 | 0.214207 |
| Low | [0, 0.295686811208725) | 2,907 | 41.275025% | 114 | 0.039216 |

These categories remain provisional. An empty tier reports undefined churn rate, not a fabricated zero; recall is undefined if there are no positive labels.

## Stakeholder criteria and sign-off

The existing [src/config.py](../src/config.py) configuration now contains `BUSINESS_ACCEPTANCE`. PR-AUC minimum, precision@K minimum, outreach capacity, and refresh cadence are `None` (unresolved). The 0.5 feature-importance maximum is the existing SDD diagnostic guardrail, not a newly invented business threshold. `stakeholder_approved` is false.

When stakeholders provide candidate values, record their names, decision date, intended population/horizon, chosen capacity, metric requirements, cadence, costs, and tier policy in the review record, then update the corresponding configuration constants and rerun. The report exposes configured values and diagnostic comparisons. A configured precision@K check uses full-snapshot descriptive precision and cannot establish held-out acceptance. Passing comparisons never grants approval. Setting `stakeholder_approved` to true fails explicitly; this CLI is not a sign-off mechanism.

Business acceptance and stakeholder status default to pending. The benchmark cannot receive production business sign-off because the necessary temporal evidence and verified active cohort do not exist. Any stakeholder acknowledgment of an exploratory review must be documented separately and must not be presented as model approval.

Open decisions include the future churn definition/horizon, numerical acceptance gates, outreach capacity and costs, refresh cadence, tier policy, and accountable stakeholder sign-off. A production model needs dated features/events, a genuine future-window label, temporal evaluation, and the verified active-customer source. Existing foundation findings also remain open. These are concrete blockers before Phase 7 production deployment; Phase 7 is not implemented here.

## Persistence and reproducibility

[business_validation](../sql/06_create_business_validation.sql) stores one typed record per distinct scenario N and analysis digest, with the model/version, source score timestamp/hash, selected and churner counts, precision/recall, tier counts/boundaries, candidate criteria, pending statuses, and notes. No SQL JSON blobs are used. Separate columns distinguish scenario N from configured stakeholder capacity. `validation_date` is the UTC date of first persistence; `created_at` is the database insertion timestamp. Repeat saves preserve them.

The analysis hash covers model/report identity, input ranking hash, source batch timestamp/hash, exact results, criteria, and analysis version. Identical inputs and arguments produce identical report bytes and validation IDs; reruns reuse records. Changing inputs/configuration/scenarios produces a new analysis. Duplicate Ns within one analysis share one persisted row. Conflicting existing values fail instead of being overwritten. Benchmark approval is also rejected by a database CHECK constraint.

Reads use one repeatable-read transaction. Exact customer-ID coverage is required across predictions, labels, and features. The feature hash must match the scored batch; the feature-plus-label hash must match the evaluated dataset. Stale scores, changed labels, mixed timestamps/versions, missing models, missing predictions, invalid scores, or tier inconsistencies fail loudly. Model artifacts are verified with the existing Phase 5 loader; neither training nor inference runs. Historic records describe their recorded batch, not a guarantee that the current database still contains that batch.

Default analysis SHA-256: `cf44c4f6de4abc889fd72f0c6f4e37e5b8b80df0dc3a3ddf0a39852e8623e526`. Evaluation report SHA-256: `9ee0132da5749c7e72124a8c3a07136ab1fb5796a65b516cabe83b80e9de42b2`. The generated report records dataset, feature-snapshot, and ranked-output hashes for traceability.

## Verification

The local suite passed 14 tests; the two opt-in MySQL tests were skipped by default. Phase 6's opt-in MySQL test passed independently: two identical runs produced four unique pending rows, every persisted analysis value matched the report, all artifacts were byte-identical, and registry approval and source predictions remained unchanged. A benchmark stakeholder-approval update was rejected and rolled back. Source-integrity regressions reject stale score hashes, changed labels, observation dates, mixed score timestamps, and incomplete feature coverage. The CLI also generated a custom N=100 report without saving additional database records.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
$env:RUN_BUSINESS_MYSQL_TESTS = '1'
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_business_validation.py -v
Remove-Item Env:RUN_BUSINESS_MYSQL_TESTS
```

The opt-in test persists pending Phase 6 records in the configured development database; it does not rescore. `TEST_MODEL_ID` can select a different registered benchmark.

All labels still come from the Telco snapshot `Churn` field. There are no observation dates, no genuine future-window churn labels, and no temporal holdout. Scores are not evidence of production-calibrated future churn probability. No stakeholder decisions or production approval were fabricated.
