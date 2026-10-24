"""Immutable cohort retention uses the caller's actual transaction."""

from datetime import date, datetime, timedelta
import hashlib
import json
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src import config
from src.features import FEATURE_COLUMNS, NUMERIC
from src.ingest import load_and_validate
from src.monitoring_history import archive_batch
from src.score import build_prediction_frame


def create_sqlite_history_schema(connection):
    """SQLite fixtures retain the production columns, keys, and relevant checks."""
    for name in ("09_create_scoring_batches.sql", "10_create_scoring_history.sql", "11_create_monitoring_outcomes.sql"):
        ddl = (config.PROJECT_ROOT / "sql" / name).read_text(encoding="utf-8")
        ddl = (ddl.replace(" ENGINE=InnoDB", "").replace(" UNSIGNED", "")
               .replace("ENUM('low', 'medium', 'high')", "TEXT").replace("CHAR_LENGTH", "LENGTH")
               .replace(",\n    INDEX idx_scoring_batch_date (feature_set_version, observation_date)", ""))
        connection.exec_driver_sql(ddl)


class MonitoringHistoryTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        with self.engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.exec_driver_sql("CREATE TABLE deployment_runs (run_id CHAR(32) PRIMARY KEY)")
            connection.exec_driver_sql("CREATE TABLE model_registry (model_id VARCHAR(50) PRIMARY KEY)")
            connection.exec_driver_sql("INSERT INTO model_registry VALUES ('production-model')")
            connection.exec_driver_sql("INSERT INTO deployment_runs VALUES ('" + "1" * 32 + "'), ('" + "2" * 32 + "')")
            create_sqlite_history_schema(connection)
        self.features = load_and_validate().loc[:1, ["customer_id", *FEATURE_COLUMNS]].copy()
        self.features[NUMERIC] = self.features[NUMERIC].astype(float)
        self.features = self.features.sort_values("customer_id").reset_index(drop=True)
        self.registry = {
            "model_id": "production-model", "approved": 1, "evaluation_scope": "production",
            "feature_set_version": "dated_history_test_v1", "label_version": "future_history_test_v1",
            "report_sha256": "a" * 64, "trained_at": datetime(2026, 1, 1),
            "low_threshold": 0.25, "high_threshold": 0.5,
            "pr_auc": 0.65, "precision_at_threshold": 0.6, "recall_at_threshold": 0.7,
        }
        self.report = {"production_acceptance": {"future_window_days": 30}, "threshold": 0.5}
        self.run = {"run_id": "1" * 32, "model_id": "production-model",
                    "feature_set_version": self.registry["feature_set_version"],
                    "observation_date": date(2026, 1, 2), "scored_at": datetime(2026, 1, 2, 12),
                    "report_sha256": "a" * 64}
        self.rehash(self.run, self.features)
        self.predictions = self.predict(self.run)

    def rehash(self, run, frame):
        run["feature_snapshot_sha256"] = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        identity = [self.registry["model_id"], self.registry["report_sha256"], self.registry["feature_set_version"],
                    run["observation_date"].isoformat(), run["feature_snapshot_sha256"], 0.25, 0.5]
        run["batch_sha256"] = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()

    def predict(self, run, probabilities=(0.2, 0.8)):
        return build_prediction_frame("production-model", self.features, probabilities, 0.25, 0.5,
                                      run["feature_snapshot_sha256"], "production",
                                      self.registry["feature_set_version"], run["scored_at"])

    def archive(self, connection, **changes):
        values = {"run": self.run, "frame": self.features, "predictions": self.predictions,
                  "registry": self.registry, "report": self.report, "low": 0.25, "high": 0.5, **changes}
        return archive_batch(connection, **values)

    def counts(self):
        with self.engine.connect() as connection:
            return tuple(connection.execute(text("SELECT COUNT(*) FROM " + name)).scalar_one()
                         for name in ("scoring_batches", "scoring_history"))

    def test_idempotent_archive_retains_first_scoring_time_and_run(self):
        with self.engine.begin() as connection:
            self.assertEqual(self.archive(connection), 2)
        rerun = {**self.run, "run_id": "2" * 32, "scored_at": self.run["scored_at"] + timedelta(days=1)}
        with self.engine.begin() as connection:
            self.assertEqual(self.archive(connection, run=rerun, predictions=self.predict(rerun)), 2)
            row = connection.execute(text("SELECT run_id, scored_at, future_window_days, baseline_precision "
                                          "FROM scoring_batches")).one()
        self.assertEqual(tuple(row), ("1" * 32, str(self.run["scored_at"]), 30, 0.6))
        self.assertEqual(self.counts(), (1, 2))

    def test_failed_publication_rolls_back_new_history_and_preserves_prior_cohort(self):
        with self.engine.begin() as connection:
            self.archive(connection)
        newer = {**self.run, "run_id": "2" * 32, "observation_date": date(2026, 1, 3)}
        self.rehash(newer, self.features)
        with self.assertRaisesRegex(RuntimeError, "publication failed"):
            with self.engine.begin() as connection:
                self.archive(connection, run=newer, predictions=self.predict(newer))
                raise RuntimeError("publication failed")
        self.assertEqual(self.counts(), (1, 2))
        with self.engine.begin() as connection:
            self.archive(connection, run=newer, predictions=self.predict(newer))
            # Latest scores/source features can change independently after retention.
            connection.exec_driver_sql("CREATE TABLE model_predictions (customer_id TEXT)")
            connection.exec_driver_sql("INSERT INTO model_predictions VALUES ('new-latest-customer')")
            connection.exec_driver_sql("DELETE FROM model_predictions")
        self.assertEqual(self.counts(), (2, 4))
        with self.engine.connect() as connection:
            old = connection.execute(text("SELECT churn_probability FROM scoring_history "
                                          "WHERE batch_sha256=:batch ORDER BY customer_id"),
                                     {"batch": self.run["batch_sha256"]}).scalars().all()
        self.assertEqual(old, [0.2, 0.8])

    def test_existing_metadata_scores_or_features_cannot_be_silently_replaced(self):
        with self.engine.begin() as connection:
            self.archive(connection)
        for changes in ({"report": {"production_acceptance": {"future_window_days": 60}, "threshold": 0.5}},
                        {"predictions": self.predict(self.run, (0.1, 0.8))}):
            with self.subTest(changes=list(changes)), self.assertRaisesRegex(ValueError, "Immutable"):
                with self.engine.begin() as connection:
                    self.archive(connection, **changes)
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE scoring_history SET monthly_charges=monthly_charges+1"))
        with self.assertRaisesRegex(ValueError, "Immutable"):
            with self.engine.begin() as connection:
                self.archive(connection)
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE scoring_history SET monthly_charges=:value"),
                               {"value": "private-invalid-source-value"})
        with self.assertRaisesRegex(ValueError, "Immutable") as caught:
            with self.engine.begin() as connection:
                self.archive(connection)
        self.assertNotIn("private-invalid-source-value", str(caught.exception))
        self.assertEqual(self.counts(), (1, 2))

    def test_benchmark_bad_hash_domains_and_missing_horizon_fail_before_writes(self):
        cases = [
            {"registry": {**self.registry, "approved": 0}},
            {"registry": {**self.registry, "evaluation_scope": "benchmark"}},
            {"registry": {**self.registry, "label_version": config.LABEL_VERSION}},
            {"registry": {**self.registry, "feature_set_version": config.FEATURE_SET_VERSION}},
            {"registry": {**self.registry, "pr_auc": np.inf}},
            {"report": {}}, {"report": {"production_acceptance": {"future_window_days": True}}},
            {"run": {**self.run, "feature_snapshot_sha256": "b" * 64}},
            {"run": {**self.run, "batch_sha256": "b" * 64}},
            {"frame": self.features.assign(contract="unknown")},
            {"predictions": self.predictions.assign(risk_tier="low")},
        ]
        for changes in cases:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                with self.engine.begin() as connection:
                    self.archive(connection, **changes)
            self.assertEqual(self.counts(), (0, 0))


if __name__ == "__main__":
    unittest.main()
