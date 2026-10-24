"""Register trusted local benchmark artifacts in MySQL without approving them."""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
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


def _validate_production_report(report, registry):
    if registry.get("approved") != 1 or registry.get("evaluation_scope") != "production":
        raise ValueError("Production requires an approved production registry model")
    version = registry.get("feature_set_version")
    if not isinstance(version, str) or not version.strip() or version == FEATURE_SET_VERSION:
        raise ValueError("Production requires a compatible non-benchmark feature version")
    if (report.get("approved") is not True or report.get("evaluation_scope") != "production"
            or report.get("evaluation_design") != "temporal" or report.get("promotion_blockers") != []):
        raise ValueError("Production artifacts require temporal evaluation and no promotion blockers")
    for name in ("feature_set_version", "label_version", "dataset_sha256"):
        if not registry.get(name) or report.get(name) != registry[name]:
            raise ValueError(f"Production report/registry mismatch: {name}")
    if report["label_version"] == LABEL_VERSION:
        raise ValueError("Snapshot outcome labels cannot support a production model")
    trained_at = datetime.fromisoformat(report["trained_at"])
    if (trained_at.tzinfo is None
            or trained_at.astimezone(timezone.utc).replace(tzinfo=None) != registry.get("trained_at")):
        raise ValueError("Production report/registry training timestamp mismatch")
    if any(report.get("test", {}).get(metric) != registry.get(column)
           for column, metric in METRICS.items()):
        raise ValueError("Production report/registry held-out metrics mismatch")
    acceptance = report.get("production_acceptance", {})
    if (not isinstance(acceptance, dict) or acceptance.get("temporal_evaluation") is not True
            or acceptance.get("stakeholder_approved") is not True
            or type(acceptance.get("future_window_days")) is not int
            or acceptance["future_window_days"] <= 0
            or not isinstance(acceptance.get("evidence_reference"), str)
            or not acceptance["evidence_reference"].strip()):
        raise ValueError("Production requires documented temporal, future-window, and stakeholder acceptance")
    low, high = acceptance.get("low_threshold"), acceptance.get("high_threshold")
    if (type(low) not in (int, float) or type(high) not in (int, float)
            or not np.isfinite([low, high]).all() or not 0 < low < high <= 1
            or low != registry.get("low_threshold") or high != registry.get("high_threshold")):
        raise ValueError("Production risk policy differs from accepted registry thresholds")


def load_artifacts(path, expected_report_sha256=None, *, production_registry=None):
    """Hashes detect changes after registration; initial artifacts must be trusted."""
    if production_registry is not None:
        pinned_hash = production_registry.get("report_sha256", "")
        if (not re.fullmatch(r"[0-9a-f]{64}", pinned_hash)
                or expected_report_sha256 not in (None, pinned_hash)):
            raise ValueError("Production artifacts require the registry's pinned report hash")
        expected_report_sha256 = pinned_hash
    path = Path(path).resolve()
    report_bytes = (path / "report.json").read_bytes()
    report_hash = hashlib.sha256(report_bytes).hexdigest()
    if expected_report_sha256 is not None and report_hash != expected_report_sha256:
        raise ValueError("Registered report hash mismatch")
    report = json.loads(report_bytes)
    if report.get("features") != list(FEATURE_COLUMNS):
        raise ValueError("Artifact ordered feature list mismatch")
    if production_registry is None:
        if (report.get("feature_set_version") != FEATURE_SET_VERSION
                or report.get("label_version") != LABEL_VERSION):
            raise ValueError("Artifact feature/label version mismatch")
        if report.get("approved") is not False or not report.get("evaluation_design", "").startswith("non-temporal"):
            raise ValueError("Only unapproved snapshot benchmark artifacts are supported")
        threshold = report.get("threshold")
        if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Benchmark threshold must be finite and in (0, 1]")
    else:
        _validate_production_report(report, production_registry)
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
    if production_registry is not None:
        versions.update(python=platform.python_version(), joblib=joblib.__version__)
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
