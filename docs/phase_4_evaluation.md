# Phase 4: XGBoost snapshot benchmark

Completed 2026-09-25 against the existing `customer_churn` MySQL snapshot. Status: evaluated benchmark, **not approved for production**. Training did not refresh or mutate database tables.

## Design

The SDD Section 5.2 amendment permits this non-temporal benchmark only. No observation dates exist, so temporal leakage cannot be ruled out. Sorted customers, seed 42, stratified 60/20/20 splits yield 4,225 training rows (1,121 churned), 1,409 validation rows (374 churned), and 1,409 test rows (374 churned), without overlap.

Training reads a consistent MySQL transaction, filters both versions to `telco_snapshot_v1`, verifies exact raw/feature/label coverage, and rejects empty data, invalid domains, missing core fields, excessive total-charge nulls, duplicate IDs, and dated labels. IDs and labels are excluded from predictors. Median imputation and one-hot encoding are fitted on training data only.

Eight seeded parameter samples use validation average precision for selection and XGBoost `aucpr` for early stopping (30 rounds). This is one validation holdout, not cross-validation. The final test is evaluated after model and threshold selection. Reported PR-AUC is scikit-learn average precision, not XGBoost's numerical integration.

Selected parameters: depth 3, learning rate 0.05, 400 maximum estimators, subsample 0.7, column subsample 0.8, minimum child weight 1; best iteration 71 (zero-based). Training class weight: 2.768956. Validation-F1 threshold: 0.591374. No post-hoc probability calibration is claimed.

## Held-out results

| Metric | Test result |
| --- | ---: |
| PR-AUC (average precision) | 0.658753 |
| Prevalence baseline AP | 0.265436 |
| ROC-AUC | 0.845402 |
| Precision | 0.589912 |
| Recall | 0.719251 |
| F1 | 0.648193 |
| Recall in top decile | 0.294118 |
| Precision in top decile | 0.780142 |
| Brier score | 0.164171 |
| Constant training-prevalence Brier baseline | 0.194980 |

The top decile contains 141 customers; ties use stable row ordering. Largest aggregated one-hot gain share: contract, 32.96%, below the 50% diagnostic gate. This does not prove absence of leakage. Selected model validation average precision: 0.664085.

## Artifacts and verification

Verified run: `models/phase4_benchmark_verified/`, containing native XGBoost JSON, fitted `preprocessor.joblib`, full `report.json` (parameters, candidates, metrics, versions, hashes), and `splits.csv`. Generated directories are ignored by Git. Only load trusted joblib artifacts. Dataset SHA-256: `826dd1d4d2a69cc84fd206cf71dbce7136aa3aad09a7d9f3451546277b500ea8`.

Five tests pass, covering training-only imputation, reproducible disjoint splits, malformed inputs, saved-model metric equivalence, driver timeouts, and earlier ingestion/label/feature tests. Live eight-candidate training succeeded with the configured PyMySQL driver. The initial failed run at `models/phase4_benchmark/` contains partial artifacts only; use the verified directory.

API references: [XGBoost 3.1 API](https://xgboost.readthedocs.io/en/release_3.1.0/python/python_api.html), [scikit-learn encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

## Remaining boundaries

No stakeholder metric gates or temporal holdout are available: `approved=false` is unconditional. Phase 5 registry/scoring remain unimplemented. In the historical Phase 0–3 review, R3's timeout mismatch is fixed; R1/R2/R4/R5/R6 remain open upstream. Independent training validation does not repair refresh atomicity or schema constraints. Resolve these before productionizing the pipeline.
