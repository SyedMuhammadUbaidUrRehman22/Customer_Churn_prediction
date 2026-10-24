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
            try:
                values = pd.to_numeric(values, errors="raise").dropna().astype(float)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(f"Invalid numeric values in {column}") from None
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


def load_active_customers(connection, observation_date):
    """Production cohort from real subscription records, never snapshot outcomes."""
    customers = pd.read_sql_query(text("SELECT customer_id, signup_date FROM customers"), connection)
    ids = customers.customer_id.astype("string")
    try:
        signup = pd.to_datetime(customers.signup_date, errors="raise")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid production customer signup dates") from None
    if (customers.empty or ids.isna().any() or not ids.is_unique or ids.str.strip().eq("").any()
            or ids.str.len().gt(20).any() or signup.isna().any()):
        raise ValueError("Production customers require unique valid IDs and signup dates")
    subscriptions = pd.read_sql_query(text("""
        SELECT s.customer_id, s.status, s.start_date, s.end_date, c.signup_date
        FROM subscriptions s LEFT JOIN customers c ON c.customer_id = s.customer_id
    """), connection)
    try:
        starts = pd.to_datetime(subscriptions.start_date, errors="raise")
        ends = pd.to_datetime(subscriptions.end_date, errors="raise")
        signup = pd.to_datetime(subscriptions.signup_date, errors="raise")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Invalid production subscription dates") from None
    if (signup.isna().any() or starts.isna().any() or starts.lt(signup).any()
            or not subscriptions.status.isin(["active", "cancelled", "paused"]).all()
            or (ends.notna() & ends.lt(starts)).any()):
        raise ValueError("Invalid production subscription source")
    cutoff = pd.Timestamp(observation_date)
    active = subscriptions.loc[subscriptions.status.eq("active") & starts.le(cutoff)
                               & (ends.isna() | ends.gt(cutoff)), "customer_id"]
    # Multiple active subscriptions still represent one intended customer.
    active = active.drop_duplicates().sort_values().reset_index(drop=True)
    if (active.empty or active.isna().any() or active.astype(str).str.strip().eq("").any()
            or active.astype(str).str.len().gt(20).any()):
        raise ValueError("A nonempty, valid active-customer population is required")
    return active.astype(str).tolist()


def refresh_production_features(connection, feature_version, observation_date, active_ids):
    """Consume an external ETL's complete dated materialization in this read snapshot.

    Dated ETL is a separate job: the undated CSV materializer cannot implement it.
    This boundary never invents dates, reads labels, or writes snapshot features.
    """
    if not feature_version or feature_version == FEATURE_SET_VERSION:
        raise ValueError("Production feature refresh rejects the Telco benchmark")
    frame = pd.read_sql_query(text(f"""
        SELECT customer_id, {', '.join(FEATURE_COLUMNS)} FROM {TABLE_FEATURES}
        WHERE feature_set_version = :version AND observation_date = :observation_date
        ORDER BY customer_id
    """), connection, params={"version": feature_version, "observation_date": observation_date})
    validate_feature_frame(frame)
    frame["customer_id"] = frame.customer_id.astype(str)
    if not set(active_ids).issubset(frame.customer_id):
        raise ValueError("Production feature coverage is incomplete for active customers")
    frame = frame[frame.customer_id.isin(active_ids)].copy().sort_values("customer_id").reset_index(drop=True)
    validate_feature_frame(frame)
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    return frame


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
