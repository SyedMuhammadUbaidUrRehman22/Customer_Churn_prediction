# Phase 0–3 implementation review

Reviewed 2026-09-24. Scope: current Python modules, SQL schemas, orchestration, tests, runtime configuration, SDD Markdown, project memory, and repository documentation. Phase 4 remains unstarted. This review records findings; application code and schemas were not changed during the review.

## Assessment

The included snapshot is represented consistently in MySQL: 7,043 raw customers, labels, and feature rows; 1,869 churned; 11 missing total charges. All copied label and predictor values match the raw table. Existing tests pass. However, input validation, configuration, and cross-stage failure handling have unresolved defects. Successful reruns on this CSV do not establish failure safety or temporal correctness.

## Findings

### R1 High: refresh is not atomic across stages

**Locations:** [ingest.py](../src/ingest.py#L112), [pipeline](../run_pipeline.py#L9), [label foreign key](../sql/02_create_labels.sql#L9), [feature schema](../sql/03_create_feature_store.sql).

Ingestion deletes every raw row. Both child tables use `ON DELETE CASCADE`, so this also deletes all existing labels and features, including other versions. Ingestion commits before label and feature materialization start their separate transactions. A later error therefore leaves the old derived data gone; readers may also observe empty derived tables between stages. Running the ingestion CLI alone leaves them empty until rebuilt.

**Evidence:** Control flow inspection and read-only inspection of both live foreign-key delete rules (`CASCADE`). A destructive failure injection was not performed on the user's database.

**Recommended fix:** Separate schema setup from data mutation, then share one transaction across the three data stages, or publish a complete staged snapshot atomically. Decide whether versions identify definitions or retained historical datasets; the present replacement strategy does not retain history. Add an isolated database regression that forces label/feature failure and verifies preservation of the previous complete snapshot.

### R2 High: numeric validation allows corruption and unloadable values

**Locations:** [numeric validation](../src/ingest.py#L79), [integer conversion](../src/ingest.py#L97), [raw schema](../sql/01_create_tables.sql).

The numeric checks require nonnegative, nonmissing values but do not require integral tenure, finite monetary values, or SQL-compatible ranges. `tenure='1.9'` passes and silently becomes `1`. `tenure='65536'` passes despite the unsigned SMALLINT limit. Both charge fields accept `'inf'`. Thus validation can report success for altered or unloadable values.

**Evidence:** Each case was reproduced using a modified copy of the first 100 CSV rows in a temporary directory. No malformed row was inserted into MySQL.

**Recommended fix:** Validate integral tenure and its SQL range before conversion; require finite charges within the decimal column range; define a monetary precision policy. Add boundary and non-finite-value tests.

### R3 High: current local URI fails in the application helper

**Locations:** [connection helper](../src/utils.py#L18), [environment lookup](../src/config.py#L7), [requirements](../requirements.txt).

The local `.env` names `mysql+pymysql`. PyMySQL 1.2.3 is now installed, correcting the earlier session's missing-driver diagnosis, but `get_engine()` passes `connection_timeout`. PyMySQL expects `connect_timeout`; the application connection raises `TypeError`. PyMySQL is also absent from the pinned dependencies.

The application reads only the process environment. Merely creating `.env` does not load it. The diagnostic process initially had no inherited `DATABASE_URI`; the read-only audit explicitly loaded the value and changed the driver to the installed, pinned MySQL Connector for its successful database checks.

**Evidence:** Calling the unmodified `get_engine().connect()` with the local configured URI reproduced `unexpected keyword argument 'connection_timeout'`.

**Recommended fix:** Standardize deployment configuration on the pinned `mysql+mysqlconnector` driver and make the environment-loading contract explicit. If multiple drivers are intended, handle their timeout arguments explicitly and test both. The README now documents the working configuration; the local `.env` was not edited.

### R4 Medium: categorical and string validation is incomplete

**Location:** [CSV validation](../src/ingest.py#L67).

Binary Yes/No fields and `SeniorCitizen` are checked, but other categories and SQL string widths are not. A `gender='Unknown'`, one blank `Contract` in 100 rows, and a 21-character customer ID all passed. The 5% blank-rate rule permits some empty strings; `NOT NULL` does not reject them. Overlong IDs can pass the CLI's validation before MySQL rejects the load.

**Recommended fix:** Specify and enforce category domains, field lengths, and whether blanks are accepted or missing. Test each class of invalid input at the CSV boundary.

### R5 Medium: empty feature input passes the quality gate

**Location:** [feature count validation](../src/features.py#L36).

`_validate_counts(0, 0, None)` succeeds. When feature materialization is called against an empty raw table, it can delete the current feature version and commit an empty result. The full pipeline's ingestion/label checks normally prevent this path, but the callable materializer does not enforce its own nonempty input contract.

**Evidence:** Direct execution of the validator with the values returned by an empty SQL aggregate.

**Recommended fix:** Reject zero customers/features and add an empty-input regression. Verify the transaction's failure behavior in an isolated database.

### R6 Medium: the date constraint permits one missing date

**Location:** [label date CHECK](../sql/02_create_labels.sql#L13).

The intended condition allows either two null dates or an ordered pair. If exactly one date is null, the SQL expression evaluates to `NULL`, which does not fail a MySQL CHECK. A future writer can therefore store an incomplete date pair. Current snapshot rows have both dates null and are unaffected.

**Evidence:** A read-only evaluation of the exact condition with a nonnull start and null end returned `NULL`. Constraint acceptance follows from SQL CHECK semantics; no violating row was inserted into the live table.

**Recommended fix:** Require both dates to be nonnull in the ordered-pair branch. Alter the existing constraint as well as the schema file, because `CREATE TABLE IF NOT EXISTS` does not migrate an existing table. Cover both-null, both-present, reversed, and partially null pairs in an isolated database test.

## Verification evidence

| Check | Result | Limits |
| --- | --- | --- |
| Existing unit suite | 3 tests passed | No SQL execution or transaction failure tests |
| Included CSV validation | 7,043 rows | Edge probes found accepted invalid inputs |
| Raw / label / feature counts | 7,043 / 7,043 / 7,043 | Current database snapshot |
| Version-filtered label/feature join | 7,043 | Both versions `telco_snapshot_v1` |
| Raw class distribution | 5,174 retained; 1,869 churned | Supplied outcome, not independently verified cancellation events |
| Label value mismatches | 0 | Compared every joined label to raw `churned` |
| Feature value mismatches | 0 | Null-safe comparisons across all 19 predictors |
| Missing total charges | 11 | No imputation yet |
| Labels with either date populated | 0 | No temporal evaluation possible |
| Live table engines | All InnoDB | Stage transactions remain separate |
| Two consecutive pipeline runs | Passed in preceding verification | Counts only; not a rollback/concurrency test |
| Original application URI | Connection failed | Driver timeout mismatch, R3 |

Runtime observed: Python 3.13.3, pandas 3.0.0, SQLAlchemy 2.0.46, mysql-connector-python 9.5.0, MySQL 8.0.46. PyMySQL 1.2.3 is locally installed but not pinned.

## SDD and documentation alignment

- Phases 1–3 implement a **snapshot adaptation**, not the full temporal event design. Phase 2 has no verified cancellation dates or agreed future horizon; Phase 3 copies 19 attributes without dated rolling windows.
- Excluding `churned` from predictors is necessary but does not prove absence of temporal leakage. Attribute values may reflect circumstances at or after the supplied outcome; this source cannot establish ordering.
- The SDD requires time-based holdouts and agreed numerical acceptance gates. Neither is available. A random-split benchmark would be a deliberate spec change, not fulfillment of the existing evaluation requirements.
- The SDD requests business outreach thresholds before Phase 3. Those thresholds and stakeholder sign-off are not recorded. Database materialization should not be described as business acceptance.
- `Copilot_context` conflicts with the SDD on table names and the primary evaluation metric. The implementation follows SDD names (`feature_store`) and the documented plan uses PR-AUC. Reconcile the older context before future work.
- Version keys identify the current materialization definition, but raw replacement cascades away all child versions. There is no immutable dataset history.
- The feature schema's NOT NULL columns do not enforce valid categories or binary domains for independent writers. Current materialization relies on the raw-layer contract, which itself has validation gaps.
- The three tests do not cover all stated quality rules, schema constraints, rollback, or full value-level rerun equality. Earlier “all checks passed” statements describe the checks executed, not exhaustive correctness.
- Training and scoring files remain empty placeholders. No model metrics, registry, production scoring, dashboard, scheduled job, access-control audit, or scale/SLA validation has been delivered.

## Suggested order of work

1. Resolve R1–R3 and add failure/boundary regressions through Phase 3.
2. Address categorical validation, empty-feature input, and date constraints.
3. Reconcile specifications and obtain the missing temporal/evaluation decisions before Phase 4.

[Return to README](../README.md)
