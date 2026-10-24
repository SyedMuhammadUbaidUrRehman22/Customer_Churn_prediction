"""Synthetic deployment transactions; no production approval or live database."""

from datetime import date, datetime
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src import config, deploy
from src.features import FEATURE_COLUMNS
from src.ingest import load_and_validate
from src.registry import METRICS
from src.score import write_predictions


class DeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = (load_and_validate().loc[:1, ["customer_id", *FEATURE_COLUMNS]]
                     .sort_values("customer_id").reset_index(drop=True))

    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        self.settings = {
            **config.PRODUCTION, "feature_set_version": "dated_test_v1", "model_id": None,
            "low_threshold": "0.25", "high_threshold": "0.5",
            "output_table": "model_predictions", "publication": "atomic_replace",
            "feature_refresh": "external", "alert_mode": "logging",
            "schedule_enabled": "false", "schedule_cron": "",
        }
        self.sink = MagicMock()
        with self.engine.begin() as connection:
            connection.exec_driver_sql("PRAGMA foreign_keys = ON")
            connection.exec_driver_sql("""
                CREATE TABLE model_registry (
                    model_id TEXT PRIMARY KEY, trained_at DATETIME NOT NULL,
                    approved INTEGER NOT NULL, evaluation_scope TEXT NOT NULL,
                    feature_set_version TEXT NOT NULL, label_version TEXT NOT NULL,
                    artifact_path TEXT NOT NULL, report_sha256 TEXT NOT NULL,
                    dataset_sha256 TEXT NOT NULL, low_threshold REAL, high_threshold REAL,
                    """ + ", ".join(f"{metric} REAL" for metric in METRICS) + """,
                    CHECK (evaluation_scope != 'benchmark' OR approved = 0)
                )
            """)
            connection.exec_driver_sql("""
                CREATE TABLE deployment_runs (
                    run_id TEXT PRIMARY KEY, started_at DATETIME, completed_at DATETIME,
                    environment TEXT, action TEXT, model_id TEXT, trained_at DATETIME,
                    feature_set_version TEXT, observation_date DATE, scoring_mode TEXT,
                    customer_count INTEGER, prediction_count INTEGER, status TEXT,
                    failure_stage TEXT, error_summary TEXT, report_sha256 TEXT,
                    model_sha256 TEXT, preprocessing_sha256 TEXT, feature_snapshot_sha256 TEXT,
                    batch_sha256 TEXT, scored_at DATETIME,
                    previous_run_id TEXT REFERENCES deployment_runs(run_id),
                    rollback_target_run_id TEXT REFERENCES deployment_runs(run_id),
                    reused_batch INTEGER,
                    CHECK (status NOT IN ('succeeded', 'rolled_back') OR
                           (environment = 'production' AND model_id IS NOT NULL
                            AND completed_at IS NOT NULL AND batch_sha256 IS NOT NULL
                            AND customer_count > 0 AND customer_count = prediction_count))
                )
            """)
            connection.exec_driver_sql("""
                CREATE TABLE production_state (
                    state_id INTEGER PRIMARY KEY CHECK (state_id = 1),
                    model_id TEXT REFERENCES model_registry(model_id),
                    run_id TEXT REFERENCES deployment_runs(run_id)
                )
            """)
            connection.exec_driver_sql("INSERT INTO production_state (state_id) VALUES (1)")
            connection.exec_driver_sql("""
                CREATE TABLE model_predictions (
                    customer_id TEXT NOT NULL, model_id TEXT REFERENCES model_registry(model_id),
                    scoring_mode TEXT NOT NULL, scored_at DATETIME NOT NULL,
                    feature_set_version TEXT NOT NULL, feature_snapshot_sha256 TEXT NOT NULL,
                    churn_probability REAL NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
                    risk_tier TEXT NOT NULL, PRIMARY KEY (customer_id, model_id, scoring_mode)
                )
            """)
        self.add_model("model-a")
        self.add_model("model-b")
        self.add_model("benchmark", approved=0, evaluation_scope="benchmark",
                       feature_set_version=config.FEATURE_SET_VERSION)
        with self.engine.begin() as connection:
            write_predictions(connection, "benchmark", self.frame, np.array([0.2, 0.8]),
                              0.25, 0.5, "b" * 64)
        self.model, self.preprocessor = MagicMock(), MagicMock()
        self.model.predict_proba.return_value = np.array([[0.8, 0.2], [0.2, 0.8]])
        self.preprocessor.transform.side_effect = lambda frame: frame
        self.report = {"artifact_sha256": {"model.json": "c" * 64, "preprocessor.joblib": "d" * 64}}
        self.mocks = {}
        for name, options in {
            "validate_database": {},
            "load_artifacts": {"return_value": (self.report, self.model, self.preprocessor, "a" * 64)},
            "load_active_customers": {"return_value": self.frame.customer_id.tolist()},
            "refresh_production_features": {"side_effect": lambda *args: self.frame.copy()},
            "LOG": {},
        }.items():
            patcher = patch(f"src.deploy.{name}", **options)
            self.mocks[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def add_model(self, model_id, **changes):
        row = {
            "model_id": model_id, "trained_at": datetime(2026, 1, 1), "approved": 1,
            "evaluation_scope": "production", "feature_set_version": "dated_test_v1",
            "label_version": "future_test_v1", "artifact_path": "synthetic-fixture",
            "report_sha256": "a" * 64, "dataset_sha256": "b" * 64,
            "low_threshold": 0.25, "high_threshold": 0.5,
            **dict.fromkeys(METRICS, 0.5), **changes,
        }
        with self.engine.begin() as connection:
            connection.execute(text(f"INSERT INTO model_registry ({', '.join(row)}) "
                                    f"VALUES ({', '.join(':' + name for name in row)})"), row)

    def run_deployment(self, **changes):
        return deploy.run_deployment(**{
            "engine": self.engine, "settings": self.settings, "environment": "production",
            "database_uri": "mysql+mysqlconnector://test@localhost/test",
            "observation_date": date(2026, 1, 2), "alert_sink": self.sink, **changes,
        })

    def publication(self):
        with self.engine.connect() as connection:
            rows = connection.execute(text("SELECT " + deploy.PREDICTION_COLUMNS
                                           + " FROM model_predictions ORDER BY scoring_mode, customer_id")).all()
            state = connection.execute(text("SELECT model_id, run_id FROM production_state")).one()
            return rows, state

    def assert_failed(self, stage, **options):
        with self.assertRaises(deploy.DeploymentError) as raised:
            self.run_deployment(**options)
        run = raised.exception.run
        self.assertEqual((run["status"], run["failure_stage"]), ("failed", stage))
        with self.engine.connect() as connection:
            saved = connection.execute(text("SELECT status, failure_stage FROM deployment_runs "
                                             "WHERE run_id = :run_id"), run).one()
        self.assertEqual(tuple(saved), ("failed", stage))
        event = self.sink.emit.call_args.args[0]
        self.assertEqual((event["event_type"], event["run_id"]), ("deployment_failure", run["run_id"]))
        return run

    def test_bootstrap_pinning_rerun_switch_and_rollback(self):
        initial = self.run_deployment()
        self.assertEqual((initial["model_id"], initial["prediction_count"]), ("model-b", 2))
        first_rows, _ = self.publication()
        self.add_model("newer-model", trained_at=datetime(2026, 1, 2))
        repeated = self.run_deployment()
        self.assertEqual((repeated["model_id"], repeated["reused_batch"]), ("model-b", 1))
        self.assertEqual(repeated["scored_at"], initial["scored_at"])
        self.assertEqual(self.publication()[0], first_rows)
        switched = self.run_deployment(model_id="model-a")
        same_model = self.run_deployment()
        restored = self.run_deployment(rollback=True)
        self.assertEqual((restored["status"], restored["model_id"]), ("rolled_back", "model-b"))
        self.assertEqual(restored["rollback_target_run_id"], repeated["run_id"])
        self.assertEqual(restored["previous_run_id"], same_model["run_id"])
        self.assertEqual(tuple(self.publication()[1]), ("model-b", restored["run_id"]))
        self.assertEqual(self.sink.emit.call_args.args[0]["event_type"], "rollback")
        with self.engine.connect() as connection:
            statuses = dict(connection.execute(text("SELECT run_id, status FROM deployment_runs")).all())
            self.assertEqual(statuses[switched["run_id"]], "succeeded")
            self.assertEqual(len(statuses), 5)
            benchmark_count = connection.execute(text("SELECT COUNT(*) FROM model_predictions "
                                                       "WHERE scoring_mode = 'benchmark'")).scalar_one()
        self.assertEqual(benchmark_count, 2)

    def test_failures_at_every_stage_preserve_publication_and_alert(self):
        self.run_deployment()
        original = self.publication()
        failures = {
            "validate_database": "PREFLIGHT", "load_artifacts": "MODEL_LOAD",
            "load_active_customers": "DATA_VALIDATION", "refresh_production_features": "FEATURE_REFRESH",
            "predict_probabilities": "SCORING", "validate_prediction_batch": "PREDICTION_VALIDATION",
        }
        for name, stage in failures.items():
            with self.subTest(stage=stage), patch(f"src.deploy.{name}", side_effect=ValueError("injected failure")):
                self.assert_failed(stage)
            self.assertEqual(self.publication(), original)

        def write_then_fail(*args, **kwargs):
            write_predictions(*args, **kwargs)
            raise ValueError("injected failure after replacement")

        with patch("src.deploy.write_predictions", side_effect=write_then_fail):
            self.assert_failed("PUBLICATION", model_id="model-a")
        self.assertEqual(self.publication(), original)

    def test_revoked_or_tampered_rollback_target_keeps_current_model(self):
        self.run_deployment(model_id="model-b")
        self.run_deployment(model_id="model-a")
        original = self.publication()
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE model_registry SET approved = 0 WHERE model_id = 'model-b'"))
        self.assert_failed("PREFLIGHT", rollback=True)
        self.assertEqual(self.publication(), original)
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE model_registry SET approved = 1 WHERE model_id = 'model-b'"))
        with patch("src.deploy.load_artifacts", side_effect=ValueError("Registered report hash mismatch")):
            self.assert_failed("MODEL_LOAD", rollback=True)
        self.assertEqual(self.publication(), original)

    def test_benchmark_missing_model_and_missing_history_fail_closed(self):
        original = self.publication()
        for options in ({"model_id": "benchmark"}, {"model_id": "missing"}, {"rollback": True},
                        {"model_id": "model-a", "rollback": True}, {"environment": "development"},
                        {"scheduled": True}):
            with self.subTest(options=options):
                self.assert_failed("PREFLIGHT", **options)
            self.assertEqual(self.publication(), original)
        self.mocks["load_artifacts"].assert_not_called()

    def test_configuration_rejects_invalid_boundaries_modes_dates_and_schedule(self):
        for changes in ({"feature_set_version": config.FEATURE_SET_VERSION}, {"low_threshold": "nan"},
                        {"high_threshold": "0.1"}, {"output_table": "other"},
                        {"publication": "append"}, {"feature_refresh": "snapshot"},
                        {"schedule_enabled": "yes"}, {"model_id": "invalid;id"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                deploy.validate_configuration({**self.settings, **changes}, "production",
                                              "mysql+mysqlconnector://test@localhost/test", date(2026, 1, 2))
        for uri in (None, "sqlite://", "mysql://test@localhost/test", "mysql+mysqlconnector://test@localhost"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                deploy.validate_configuration(self.settings, "production", uri, date(2026, 1, 2))
        with self.assertRaises(ValueError):
            deploy.validate_configuration(self.settings, "production", "mysql+mysqlconnector://test@localhost/test",
                                          date(9999, 1, 1))
        accepted = deploy.validate_configuration(
            {**self.settings, "schedule_enabled": "true", "schedule_cron": "0 1 * * 1"},
            "production", "mysql+mysqlconnector://test@localhost/test", date(2026, 1, 2), scheduled=True)
        self.assertEqual(accepted, (0.25, 0.5))

    def test_invalid_inference_and_corrupted_reused_batch_fail_validation(self):
        self.run_deployment()
        original = self.publication()
        for probabilities in (np.array([np.nan, 0.8]), np.array([0.2]), np.array([0.2, 1.2])):
            with self.subTest(probabilities=probabilities), \
                    patch("src.deploy.predict_probabilities", return_value=probabilities):
                self.assert_failed("PREDICTION_VALIDATION")
            self.assertEqual(self.publication(), original)
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE model_predictions SET churn_probability = 0.1 "
                                    "WHERE scoring_mode = 'production' AND churn_probability = 0.2"))
        corrupted = self.publication()
        self.assert_failed("PREDICTION_VALIDATION")
        self.assertEqual(self.publication(), corrupted)

    def test_failed_alert_delivery_does_not_hide_original_failure(self):
        self.sink.emit.side_effect = RuntimeError("alert provider unavailable")
        run = self.assert_failed("PREFLIGHT", model_id="missing")
        self.assertIn("No approved production model", run["error_summary"])
        self.mocks["LOG"].error.assert_called()

    def test_connection_uses_the_database_uri_validated_by_preflight(self):
        uri = "mysql+mysqlconnector://test@localhost/explicit_test_db"
        with patch("src.deploy.get_engine", return_value=self.engine) as get_engine:
            run = self.run_deployment(engine=None, database_uri=uri)
        self.assertEqual(run["status"], "succeeded")
        get_engine.assert_called_once_with(uri)


if __name__ == "__main__":
    unittest.main()
