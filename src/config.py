"""Configuration for the implemented pipeline stages."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_URI = os.getenv("DATABASE_URI")
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_SCHEMA_PATH = PROJECT_ROOT / "sql" / "01_create_tables.sql"
TABLE_RAW = "customers_raw"
MAX_CONNECTIONS = 20
TIMEOUT_SECONDS = 30

