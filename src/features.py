"""Materialize the versioned feature set for the Telco snapshot."""

import numpy as np
import pandas as pd
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

NUMERIC = ["senior_citizen", "partner", "dependents", "tenure_months",
           "phone_service", "paperless_billing", "monthly_charges", "total_charges"]
CATEGORICAL = [column for column in FEATURE_COLUMNS if column not in NUMERIC]
CATEGORIES = {
    "gender": {"Female", "Male"},
    "multiple_lines": {"Yes", "No", "No phone service"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "contract": {"Month-to-month", "One year", "Two year"},
    "payment_method": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    **{column: {"Yes", "No", "No internet service"} for column in
       ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]},
}


def validate_feature_frame(frame):
    """Shared training/scoring contract; scoring never requires outcome labels."""
    if not {"customer_id", *FEATURE_COLUMNS}.issubset(frame.columns) or frame.empty:
        raise ValueError("Feature data must be nonempty and contain all required columns")
    ids = frame.customer_id.astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.str.len().gt(20).any() or not ids.is_unique:
        raise ValueError("Customer IDs must be nonblank, unique, and at most 20 characters")
    for column in FEATURE_COLUMNS:
        values = frame[column]
        if values.isna().mean() > 0.05 or (column != "total_charges" and values.isna().any()):
            raise ValueError(f"Invalid null rate for {column}")
        if column in CATEGORICAL:
            if not values.isin(CATEGORIES[column]).all():
                raise ValueError(f"Invalid categories in {column}")
        else:
            values = pd.to_numeric(values, errors="raise").dropna().astype(float)
            if not np.isfinite(values).all() or values.lt(0).any():
                raise ValueError(f"Invalid numeric values in {column}")
            if column in ("monthly_charges", "total_charges"):
                valid = values.le(99999999.99).all()
            elif column == "tenure_months":
                valid = values.le(65535).all() and values.mod(1).eq(0).all()
            else:
                valid = values.isin([0, 1]).all()
            if not valid:
                raise ValueError(f"Out-of-domain values in {column}")


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
