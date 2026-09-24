"""Run the implemented churn pipeline stages in order."""

from src.ingest import ingest


if __name__ == "__main__":
    print(f"Loaded {ingest():,} customer rows")
