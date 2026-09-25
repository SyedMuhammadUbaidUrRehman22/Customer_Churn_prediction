"""Score the feature snapshot in explicit benchmark mode; production stays gated."""

import argparse
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import (FEATURE_SET_VERSION, TABLE_FEATURES, TABLE_PREDICTIONS,
                        TABLE_RAW, TABLE_REGISTRY)
from src.features import FEATURE_COLUMNS, NUMERIC, validate_feature_frame
from src.registry import load_artifacts
from src.utils import get_engine


def risk_tiers(probabilities, low_threshold, high_threshold):
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("Probabilities must be a finite vector in [0, 1]")
    if not np.isfinite([low_threshold, high_threshold]).all() or not 0 < low_threshold < high_threshold <= 1:
        raise ValueError("Risk thresholds must satisfy 0 < low < high <= 1")
    return np.where(probabilities >= high_threshold, "high",
                    np.where(probabilities >= low_threshold, "medium", "low"))


def write_predictions(connection, model_id, frame, probabilities, low_threshold, high_threshold, snapshot_sha256):
    """Caller owns the transaction and registry row lock; replace only this model's benchmark."""
    tiers = risk_tiers(probabilities, low_threshold, high_threshold)
    if frame.empty or len(frame) != len(tiers) or not frame.customer_id.is_unique:
        raise ValueError("Prediction rows must cover unique customers exactly once")
    scored_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = [{"customer_id": customer_id, "model_id": model_id, "scoring_mode": "benchmark",
             "scored_at": scored_at, "feature_set_version": FEATURE_SET_VERSION,
             "feature_snapshot_sha256": snapshot_sha256, "churn_probability": float(probability),
             "risk_tier": tier}
            for customer_id, probability, tier in zip(frame.customer_id, probabilities, tiers)]
    connection.execute(text(f"DELETE FROM {TABLE_PREDICTIONS} "
                            "WHERE model_id = :model_id AND scoring_mode = 'benchmark'"), {"model_id": model_id})
    columns = ", ".join(rows[0])
    placeholders = ", ".join(f":{column}" for column in rows[0])
    statement = text(f"INSERT INTO {TABLE_PREDICTIONS} ({columns}) VALUES ({placeholders})")
    for offset in range(0, len(rows), 1000):
        connection.execute(statement, rows[offset:offset + 1000])
    return len(rows)


def score_customers(model_id=None, benchmark=False):
    if benchmark and not model_id:
        raise ValueError("Benchmark scoring requires an explicit model_id")
    engine = get_engine()
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                condition = "model_id = :model_id" if model_id else "approved = 1 AND evaluation_scope = 'production'"
                registry = connection.execute(text(f"""
                    SELECT model_id, approved, evaluation_scope, feature_set_version, artifact_path,
                           report_sha256, low_threshold, high_threshold
                    FROM {TABLE_REGISTRY} WHERE {condition}
                    ORDER BY trained_at DESC, model_id DESC LIMIT 1 FOR UPDATE
                """), {"model_id": model_id}).mappings().first()
                if registry is None:
                    raise ValueError("Model not found" if model_id else "No approved production model is available")
                if not benchmark:
                    if not registry["approved"] or registry["evaluation_scope"] != "production":
                        raise ValueError("Production scoring requires an approved production model")
                    raise ValueError("Production scoring requires dated features and a verified active-customer source")
                if registry["evaluation_scope"] != "benchmark" or registry["approved"]:
                    raise ValueError("Benchmark mode requires an unapproved benchmark model")
                if registry["feature_set_version"] != FEATURE_SET_VERSION:
                    raise ValueError("Unsupported registry feature version")
                report, model, preprocess, _ = load_artifacts(registry["artifact_path"], registry["report_sha256"])
                low, high = float(registry["low_threshold"]), float(registry["high_threshold"])
                if high != report["threshold"] or low != high / 2:
                    raise ValueError("Registry risk thresholds differ from the benchmark policy")
                columns = ", ".join(FEATURE_COLUMNS)
                frame = pd.read_sql_query(text(f"SELECT customer_id, {columns} FROM {TABLE_FEATURES} "
                                               "WHERE feature_set_version = :version ORDER BY customer_id"),
                                          connection, params={"version": FEATURE_SET_VERSION})
                raw_count = connection.execute(text(f"SELECT COUNT(*) FROM {TABLE_RAW}")).scalar_one()
                if len(frame) != raw_count:
                    raise ValueError("Scoring requires exact raw/feature customer coverage")
                validate_feature_frame(frame)
                frame[NUMERIC] = frame[NUMERIC].astype(float)
                snapshot_hash = hashlib.sha256(frame[["customer_id", *FEATURE_COLUMNS]].to_csv(index=False).encode()).hexdigest()
                # ponytail: in-memory snapshot suits 7k rows; chunk reads/inference before a 1M-customer SLA.
                probabilities = model.predict_proba(preprocess.transform(frame[list(FEATURE_COLUMNS)]))[:, 1]
                count = write_predictions(connection, registry["model_id"], frame, probabilities, low, high, snapshot_hash)
        return {"model_id": registry["model_id"], "scoring_mode": "benchmark", "rows": count,
                "feature_snapshot_sha256": snapshot_hash, "approved": False}
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id")
    parser.add_argument("--benchmark", action="store_true", help="Score all snapshot customers for demonstration only")
    args = parser.parse_args()
    print(json.dumps(score_customers(args.model_id, args.benchmark), indent=2))


if __name__ == "__main__":
    main()
