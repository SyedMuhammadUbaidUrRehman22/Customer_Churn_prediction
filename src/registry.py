"""Register trusted local benchmark artifacts in MySQL without approving them."""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import text
import xgboost

from src.config import (FEATURE_SET_VERSION, LABEL_VERSION, PREDICTIONS_SCHEMA_PATH,
                        REGISTRY_SCHEMA_PATH, TABLE_REGISTRY)
from src.features import FEATURE_COLUMNS
from src.utils import get_engine

METRICS = {
    "pr_auc": "pr_auc_average_precision", "roc_auc": "roc_auc",
    "precision_at_k": "precision_at_top_decile", "recall_at_k": "recall_at_top_decile",
    "precision_at_threshold": "precision", "recall_at_threshold": "recall",
    "f1": "f1", "brier_score": "brier_score",
}


def load_artifacts(path, expected_report_sha256=None):
    """Hashes detect changes after registration; initial artifacts must be trusted."""
    path = Path(path).resolve()
    report_bytes = (path / "report.json").read_bytes()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    if expected_report_sha256 is not None and report_hash != expected_report_sha256:
        raise ValueError("Registered report hash mismatch")
    report = json.loads(report_bytes)
    if (report.get("feature_set_version") != FEATURE_SET_VERSION
            or report.get("label_version") != LABEL_VERSION
            or report.get("features") != list(FEATURE_COLUMNS)):
        raise ValueError("Artifact feature/label version or ordered feature list mismatch")
    if report.get("approved") is not False or not report.get("evaluation_design", "").startswith("non-temporal"):
        raise ValueError("Only unapproved snapshot benchmark artifacts are supported")
    threshold = report.get("threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Benchmark threshold must be finite and in (0, 1]")
    for name in METRICS.values():
        value = report.get("test", {}).get(name)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Missing or invalid held-out metric: {name}")
    if not re.fullmatch(r"[0-9a-f]{64}", report.get("dataset_sha256", "")):
        raise ValueError("Missing or invalid dataset hash")
    trained_at = datetime.fromisoformat(report["trained_at"])
    if trained_at.tzinfo is None:
        raise ValueError("Training timestamp must include a timezone")
    versions = {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                "pandas": pd.__version__, "numpy": np.__version__}
    if any(report.get("versions", {}).get(name) != version for name, version in versions.items()):
        raise ValueError("Artifact dependency versions differ; use the training environment")
    contents = {}
    for name in ("model.json", "preprocessor.joblib"):
        contents[name] = (path / name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != report.get("artifact_sha256", {}).get(name):
            raise ValueError(f"Artifact hash mismatch: {name}")
    # Deserialize the same bytes we verified, not a second read of a mutable file.
    preprocess = joblib.load(BytesIO(contents["preprocessor.joblib"]))
    model = xgboost.XGBClassifier()
    model.load_model(bytearray(contents["model.json"]))
    if list(preprocess.feature_names_in_) != list(FEATURE_COLUMNS):
        raise ValueError("Preprocessor feature order mismatch")
    if model.n_features_in_ != len(preprocess.get_feature_names_out()):
        raise ValueError("Model and preprocessor dimensions differ")
    return report, model, preprocess, report_hash


def ensure_schema(engine):
    # MySQL DDL commits implicitly: never run this inside a prediction refresh.
    with engine.begin() as connection:
        for path in (REGISTRY_SCHEMA_PATH, PREDICTIONS_SCHEMA_PATH):
            connection.exec_driver_sql(path.read_text(encoding="utf-8").rstrip(";\n"))


def register_model(path):
    path = Path(path).resolve()
    report, _, _, report_hash = load_artifacts(path)
    row = {
        "model_id": "xgb_" + report_hash[:24],
        "trained_at": datetime.fromisoformat(report["trained_at"]).astimezone(timezone.utc).replace(tzinfo=None),
        "feature_set_version": report["feature_set_version"], "label_version": report["label_version"],
        "evaluation_scope": "benchmark", "approved": 0,
        **{column: float(report["test"][metric]) for column, metric in METRICS.items()},
        "low_threshold": report["threshold"] / 2, "high_threshold": report["threshold"],
        "artifact_path": str(path), "report_sha256": report_hash,
        "dataset_sha256": report["dataset_sha256"],
    }
    if len(row["artifact_path"]) > 1024:
        raise ValueError("Artifact path exceeds 1024 characters")
    engine = get_engine()
    try:
        ensure_schema(engine)
        with engine.begin() as connection:
            columns = ", ".join(row)
            placeholders = ", ".join(f":{column}" for column in row)
            connection.execute(text(f"INSERT INTO {TABLE_REGISTRY} ({columns}) VALUES ({placeholders}) "
                                    "ON DUPLICATE KEY UPDATE model_id = model_id"), row)
            stored = connection.execute(text(f"SELECT {columns} FROM {TABLE_REGISTRY} "
                                             "WHERE model_id = :model_id FOR UPDATE"), row).mappings().one()
            if any(stored[column] != value for column, value in row.items()):
                raise ValueError("Registration conflicts with immutable model metadata")
        return row["model_id"]
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Trusted local training output directory")
    args = parser.parse_args()
    print(json.dumps({"model_id": register_model(args.artifact_dir), "approved": False}))


if __name__ == "__main__":
    main()
