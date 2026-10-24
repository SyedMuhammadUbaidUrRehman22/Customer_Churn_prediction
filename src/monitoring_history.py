"""Retain immutable production cohorts inside the caller's publication transaction."""

from datetime import date
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sqlalchemy import text

from src import config
from src.features import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, validate_feature_frame
from src.score import validate_prediction_batch

HISTORY_COLUMNS = ("customer_id", *FEATURE_COLUMNS, "churn_probability", "risk_tier")


def batch_hash(registry, frame, observation_date, low, high):
    source = frame[["customer_id", *FEATURE_COLUMNS]].sort_values("customer_id").reset_index(drop=True)
    snapshot_hash = hashlib.sha256(source.to_csv(index=False).encode()).hexdigest()
    identity = [registry["model_id"], registry["report_sha256"], registry["feature_set_version"],
                observation_date.isoformat(), snapshot_hash, low, high]
    return snapshot_hash, hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def ensure_schema(engine):
    """Run DDL separately: MySQL schema changes implicitly commit."""
    with engine.begin() as connection:
        for path in (config.SCORING_BATCHES_SCHEMA_PATH, config.SCORING_HISTORY_SCHEMA_PATH,
                     config.MONITORING_OUTCOMES_SCHEMA_PATH):
            connection.exec_driver_sql(path.read_text(encoding="utf-8").rstrip(";\n"))


def _canonical_history(frame):
    frame = frame[list(HISTORY_COLUMNS)].sort_values("customer_id").reset_index(drop=True).copy()
    frame[[*NUMERIC, "churn_probability"]] = frame[[*NUMERIC, "churn_probability"]].astype(float)
    frame[["customer_id", *CATEGORICAL, "risk_tier"]] = frame[["customer_id", *CATEGORICAL, "risk_tier"]].astype(str)
    return frame


def _comparable_metadata(metadata):
    result = dict(metadata)
    # Revisiting the same identity retains the first successful archive and scoring time.
    for key in ("run_id", "scored_at"):
        result.pop(key, None)
    result["observation_date"] = pd.Timestamp(result["observation_date"]).date()
    result["trained_at"] = pd.Timestamp(result["trained_at"]).isoformat()
    return result


def archive_batch(connection, run, frame, predictions, registry, report, low, high):
    """Validate and retain one production batch; caller owns locks and transaction."""
    if registry.get("approved") != 1 or registry.get("evaluation_scope") != "production":
        raise ValueError("History requires an approved production model")
    for key in ("feature_set_version", "label_version"):
        value = registry.get(key)
        if (not isinstance(value, str) or not value.strip() or len(value) > 30
                or value == config.FEATURE_SET_VERSION):
            raise ValueError("History requires non-benchmark feature and label versions")
    horizon = report.get("production_acceptance", {}).get("future_window_days")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("History requires a verified positive future-window horizon")
    if type(run.get("observation_date")) is not date:
        raise ValueError("History requires a real observation date")
    for key in ("batch_sha256", "feature_snapshot_sha256", "report_sha256"):
        if not isinstance(run.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", run[key]):
            raise ValueError("History source hashes must be SHA-256 digests")
    if (not isinstance(run.get("run_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", run["run_id"])
            or run.get("model_id") != registry.get("model_id")
            or run.get("feature_set_version") != registry["feature_set_version"]
            or run["report_sha256"] != registry.get("report_sha256")
            or low != registry.get("low_threshold") or high != registry.get("high_threshold")):
        raise ValueError("History metadata differs from the selected production model")
    validate_feature_frame(frame)
    validate_prediction_batch(predictions, frame.customer_id, registry["model_id"],
                              registry["feature_set_version"], run["feature_snapshot_sha256"],
                              low, high, run["scored_at"])
    source = frame[["customer_id", *FEATURE_COLUMNS]].sort_values("customer_id").reset_index(drop=True)
    snapshot_hash, identity_hash = batch_hash(registry, source, run["observation_date"], low, high)
    if snapshot_hash != run["feature_snapshot_sha256"] or identity_hash != run["batch_sha256"]:
        raise ValueError("History source features or batch identity do not match the validated hashes")
    threshold = report.get("threshold")
    metrics = [registry.get(key) for key in ("pr_auc", "precision_at_threshold", "recall_at_threshold")]
    if any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1 for value in metrics):
        raise ValueError("History requires finite held-out metric baselines")
    if threshold is not None and (type(threshold) not in (int, float)
                                  or not np.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("History evaluation threshold is invalid")
    metadata = {key: run[key] for key in ("batch_sha256", "run_id", "model_id", "feature_set_version",
                                         "observation_date", "scored_at", "feature_snapshot_sha256", "report_sha256")}
    metadata.update(label_version=registry["label_version"], customer_count=len(frame),
                    low_threshold=float(low), high_threshold=float(high), future_window_days=horizon,
                    evaluation_threshold=threshold, baseline_pr_auc=metrics[0], baseline_precision=metrics[1],
                    baseline_recall=metrics[2], trained_at=registry["trained_at"])
    history = _canonical_history(source.merge(predictions[["customer_id", "churn_probability", "risk_tier"]],
                                               on="customer_id", validate="one_to_one"))
    existing = connection.execute(text("SELECT " + ", ".join(metadata) + " FROM scoring_batches "
                                        "WHERE batch_sha256 = :batch_sha256"), metadata).mappings().first()
    if existing is not None:
        if _comparable_metadata(existing) != _comparable_metadata(metadata):
            raise ValueError("Immutable scoring batch metadata differs from the retained archive")
        stored = pd.read_sql_query(text("SELECT " + ", ".join(HISTORY_COLUMNS) + " FROM scoring_history "
                                        "WHERE batch_sha256 = :batch_sha256 ORDER BY customer_id"), connection,
                                   params={"batch_sha256": metadata["batch_sha256"]})
        try:
            pd.testing.assert_frame_equal(_canonical_history(stored), history, check_dtype=False, check_exact=True)
        except (AssertionError, TypeError, ValueError, OverflowError):
            raise ValueError("Immutable scoring history differs from the retained archive") from None
        return len(history)
    columns = ", ".join(metadata)
    placeholders = ", ".join(":" + key for key in metadata)
    connection.execute(text(f"INSERT INTO scoring_batches ({columns}) VALUES ({placeholders})"), metadata)
    rows = history.assign(batch_sha256=metadata["batch_sha256"]).astype(object)
    rows = rows.where(pd.notna(rows), None).to_dict(orient="records")
    columns = ", ".join(rows[0])
    placeholders = ", ".join(":" + key for key in rows[0])
    statement = text(f"INSERT INTO scoring_history ({columns}) VALUES ({placeholders})")
    for offset in range(0, len(rows), 1000):
        connection.execute(statement, rows[offset:offset + 1000])
    return len(history)
