"""
config.py

Central configuration for the Churn Analytics Project.

Contains:
- Database connection
- File paths
- Table names
- ML parameters
- Logging configuration
- Power BI dashboard location
"""

# =========================
# DATABASE CONFIGURATION
# =========================
# Using SQLAlchemy MySQL connector
DATABASE_URI = "mysql+mysqlconnector://username:password@localhost:3306/churn_db"

# Connection pool settings
MAX_CONNECTIONS = 20
TIMEOUT_SECONDS = 30

# =========================
# PROJECT METADATA
# =========================
PROJECT_NAME = "Churn Analytics"
LOG_LEVEL = "DEBUG"
RANDOM_SEED = 42

# =========================
# DATABASE TABLES
# =========================
TABLE_RAW = "customers_raw"          # Raw ingested CSV data
TABLE_FEATURES = "customer_features" # Feature-engineered table
TABLE_PREDICTIONS = "churn_predictions" # Model scoring output

# =========================
# FILE PATHS
# =========================
RAW_DATA_PATH = "data/raw/telco_churn.csv"
MODEL_PATH = "models/final_model.pkl"
SHAP_FIGURE_PATH = "reports/figures/shap_summary.png"
LOG_FILE = "logs/pipeline.log"

# =========================
# ML PIPELINE PARAMETERS
# =========================
TARGET_COLUMN = "Churn"   # Column to predict
TEST_SIZE = 0.2           # Train/test split ratio
CLASS_IMBALANCE_METHOD = "scale_pos_weight"  # Could be used in XGBoost
FEATURE_SELECTION = True   # Optional feature selection flag

# =========================
# POWER BI CONFIGURATION
# =========================
PBI_DASHBOARD_PATH = "powerbi/churn_dashboard.pbix"

# =========================
# OPTIONAL FEATURE FLAGS
# =========================
ENABLE_SHAP_VISUALS = True
ENABLE_FEATURE_X = False   # Example placeholder for experimental features

