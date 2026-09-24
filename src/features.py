"""Materialize the versioned feature set for the Telco snapshot."""

from sqlalchemy import text

from src.config import (
    FEATURE_SCHEMA_PATH,
    FEATURE_SET_VERSION,
    TABLE_FEATURES,
    TABLE_RAW,
)
from src.utils import get_engine

FEATURE_COLUMNS = (
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "monthly_charges",
    "total_charges",
)


def _validate_counts(customer_rows, feature_rows, total_charge_nulls):
    if feature_rows != customer_rows:
        raise ValueError(
            f"Expected one feature row per customer, found {feature_rows} rows for "
            f"{customer_rows} customers"
        )
    if feature_rows and total_charge_nulls / feature_rows > 0.05:
        raise ValueError("total_charges exceeds the 5% null-value limit")


def materialize_features():
    columns = ", ".join(FEATURE_COLUMNS)
    engine = get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            FEATURE_SCHEMA_PATH.read_text(encoding="utf-8").rstrip(";\n")
        )
        connection.execute(
            text(
                f"DELETE FROM {TABLE_FEATURES} "
                "WHERE feature_set_version = :version"
            ),
            {"version": FEATURE_SET_VERSION},
        )
        connection.execute(
            text(
                f"""
                INSERT INTO {TABLE_FEATURES}
                    (customer_id, feature_set_version, {columns})
                SELECT customer_id, :version, {columns}
                FROM {TABLE_RAW}
                """
            ),
            {"version": FEATURE_SET_VERSION},
        )
        customer_rows = connection.execute(
            text(f"SELECT COUNT(*) FROM {TABLE_RAW}")
        ).scalar_one()
        feature_rows, total_charge_nulls = connection.execute(
            text(
                f"""
                SELECT COUNT(*), SUM(total_charges IS NULL)
                FROM {TABLE_FEATURES}
                WHERE feature_set_version = :version
                """
            ),
            {"version": FEATURE_SET_VERSION},
        ).one()
        _validate_counts(customer_rows, feature_rows, total_charge_nulls)
    return feature_rows
