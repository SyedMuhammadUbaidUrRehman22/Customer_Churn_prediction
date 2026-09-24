# Churn Analytics

Phase 1 loads the included Telco churn CSV into an explicit MySQL `customers_raw` table after validating its schema, customer IDs, categorical flags, numeric values, and core-column blank rates.

```powershell
$env:DATABASE_URI = "mysql+mysqlconnector://user:password@localhost:3306/churn_db"
python -m src.ingest --validate-only
python run_pipeline.py
```

The load treats the CSV as a full snapshot: it replaces all rows in `customers_raw` inside one transaction, so reruns are deterministic and failed inserts roll back.
