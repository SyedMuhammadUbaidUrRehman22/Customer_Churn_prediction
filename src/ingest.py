"""Validate the Telco churn CSV and load it into MySQL."""

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.config import RAW_DATA_PATH, RAW_SCHEMA_PATH, TABLE_RAW
from src.utils import get_engine

SOURCE_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

COLUMN_NAMES = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churned",
}

YES_NO_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_and_validate(csv_path=RAW_DATA_PATH):
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    _require(list(frame.columns) == SOURCE_COLUMNS, "CSV columns do not match the expected Telco schema")
    _require(not frame.empty, "CSV contains no customer rows")

    frame = frame.apply(lambda column: column.str.strip())
    _require(frame["customerID"].ne("").all(), "customerID must not be blank")
    _require(frame["customerID"].is_unique, "customerID must be unique")
    _require(frame["SeniorCitizen"].isin({"0", "1"}).all(), "SeniorCitizen must be 0 or 1")
    for column in YES_NO_COLUMNS:
        _require(frame[column].isin({"Yes", "No"}).all(), f"{column} must be Yes or No")

    tenure = pd.to_numeric(frame["tenure"], errors="coerce")
    monthly = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(frame["TotalCharges"].replace("", pd.NA), errors="coerce")
    _require(tenure.notna().all() and tenure.ge(0).all(), "tenure must be a non-negative number")
    _require(monthly.notna().all() and monthly.ge(0).all(), "MonthlyCharges must be non-negative")
    _require(
        total[frame["TotalCharges"].ne("")].notna().all() and total.dropna().ge(0).all(),
        "TotalCharges must be blank or non-negative",
    )
    _require(
        frame.drop(columns="TotalCharges").ne("").mean().ge(0.95).all(),
        "A core column exceeds the 5% blank-value limit",
    )

    frame = frame.rename(columns=COLUMN_NAMES)
    for column in [COLUMN_NAMES[name] for name in YES_NO_COLUMNS]:
        frame[column] = frame[column].map({"No": 0, "Yes": 1})
    frame["senior_citizen"] = frame["senior_citizen"].astype(int)
    frame["tenure_months"] = tenure.astype(int)
    frame["monthly_charges"] = monthly
    frame["total_charges"] = total
    return frame.astype(object).where(frame.notna(), None)


def ingest(csv_path=RAW_DATA_PATH):
    frame = load_and_validate(csv_path)
    columns = list(frame.columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    statement = text(
        f"INSERT INTO {TABLE_RAW} ({', '.join(columns)}) VALUES ({placeholders})"
    )

    engine = get_engine()
    with engine.begin() as connection:
        connection.exec_driver_sql(RAW_SCHEMA_PATH.read_text(encoding="utf-8").rstrip(";\n"))
        connection.execute(text(f"DELETE FROM {TABLE_RAW}"))
        connection.execute(statement, frame.to_dict(orient="records"))
    return len(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    frame = load_and_validate(args.csv)
    if args.validate_only:
        print(f"Validated {len(frame):,} customer rows")
        return
    print(f"Loaded {ingest(args.csv):,} customer rows into {TABLE_RAW}")


if __name__ == "__main__":
    main()
