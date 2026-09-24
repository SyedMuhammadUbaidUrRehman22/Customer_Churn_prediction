"""Run the implemented churn pipeline stages in order."""

from src.features import materialize_features
from src.ingest import ingest
from src.labels import materialize_labels


if __name__ == "__main__":
    print(f"Loaded {ingest():,} customer rows")
    label_rows, churned_rows = materialize_labels()
    print(f"Materialized {label_rows:,} labels ({churned_rows:,} churned)")
    print(f"Materialized {materialize_features():,} feature rows")
