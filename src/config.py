"""Configuration for the implemented pipeline stages."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_URI = os.getenv("DATABASE_URI")
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RAW_SCHEMA_PATH = PROJECT_ROOT / "sql" / "01_create_tables.sql"
LABEL_SCHEMA_PATH = PROJECT_ROOT / "sql" / "02_create_labels.sql"
FEATURE_SCHEMA_PATH = PROJECT_ROOT / "sql" / "03_create_feature_store.sql"
REGISTRY_SCHEMA_PATH = PROJECT_ROOT / "sql" / "04_create_model_registry.sql"
PREDICTIONS_SCHEMA_PATH = PROJECT_ROOT / "sql" / "05_create_model_predictions.sql"
TABLE_RAW = "customers_raw"
TABLE_LABELS = "churn_labels"
TABLE_FEATURES = "feature_store"
TABLE_REGISTRY = "model_registry"
TABLE_PREDICTIONS = "model_predictions"
LABEL_VERSION = "telco_snapshot_v1"
FEATURE_SET_VERSION = "telco_snapshot_v1"
MAX_CONNECTIONS = 20
TIMEOUT_SECONDS = 30

BUSINESS_SCHEMA_PATH = PROJECT_ROOT / "sql" / "06_create_business_validation.sql"
TABLE_BUSINESS_VALIDATION = "business_validation"
# None means unresolved. Configuring a criterion never grants stakeholder approval.
BUSINESS_ACCEPTANCE = {
    "pr_auc_min": None,
    "precision_at_k_min": None,
    "outreach_capacity": None,
    "refresh_cadence": None,
    "feature_importance_max": 0.5,  # Existing SDD diagnostic guardrail.
    "stakeholder_approved": False,
}

# Deployment is opt-in. No setting grants model or stakeholder approval.
APP_ENV = os.getenv("APP_ENV", "development")
DEPLOYMENT_SCHEMA_PATH = PROJECT_ROOT / "sql" / "07_create_deployment_runs.sql"
PRODUCTION_STATE_SCHEMA_PATH = PROJECT_ROOT / "sql" / "08_create_production_state.sql"
PRODUCTION = {
    "feature_set_version": os.getenv("PRODUCTION_FEATURE_SET_VERSION", ""),
    "model_id": os.getenv("PRODUCTION_MODEL_ID") or None,
    "low_threshold": os.getenv("PRODUCTION_LOW_THRESHOLD", ""),
    "high_threshold": os.getenv("PRODUCTION_HIGH_THRESHOLD", ""),
    "output_table": os.getenv("PRODUCTION_OUTPUT_TABLE", TABLE_PREDICTIONS),
    "publication": os.getenv("PRODUCTION_PUBLICATION", "atomic_replace"),
    "feature_refresh": os.getenv("PRODUCTION_FEATURE_REFRESH", "external"),
    "alert_mode": os.getenv("DEPLOY_ALERT_MODE", "logging"),
    "schedule_enabled": os.getenv("DEPLOY_SCHEDULE_ENABLED", "false"),
    "schedule_cron": os.getenv("DEPLOY_SCHEDULE_CRON", ""),
}

