"""Materialize the supplied snapshot churn outcome as a versioned label."""

from sqlalchemy import text

from src.config import LABEL_SCHEMA_PATH, LABEL_VERSION, TABLE_LABELS, TABLE_RAW
from src.utils import get_engine


def _validate_counts(customer_rows, label_rows, churned_rows):
    if label_rows != customer_rows:
        raise ValueError(
            f"Expected one label per customer, found {label_rows} labels for "
            f"{customer_rows} customers"
        )
    if not 0 < churned_rows < label_rows:
        raise ValueError("Labels must contain both churned and retained customers")


def materialize_labels():
    engine = get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            LABEL_SCHEMA_PATH.read_text(encoding="utf-8").rstrip(";\n")
        )
        connection.execute(
            text(f"DELETE FROM {TABLE_LABELS} WHERE label_version = :version"),
            {"version": LABEL_VERSION},
        )
        connection.execute(
            text(
                f"""
                INSERT INTO {TABLE_LABELS}
                    (customer_id, label_version, churned, label_source)
                SELECT customer_id, :version, churned, 'source_churn_column'
                FROM {TABLE_RAW}
                """
            ),
            {"version": LABEL_VERSION},
        )
        customer_rows = connection.execute(
            text(f"SELECT COUNT(*) FROM {TABLE_RAW}")
        ).scalar_one()
        label_rows, churned_rows = connection.execute(
            text(
                f"""
                SELECT COUNT(*), COALESCE(SUM(churned), 0)
                FROM {TABLE_LABELS}
                WHERE label_version = :version
                """
            ),
            {"version": LABEL_VERSION},
        ).one()
        _validate_counts(customer_rows, label_rows, churned_rows)
    return label_rows, churned_rows
