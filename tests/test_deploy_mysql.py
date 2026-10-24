"""Opt-in real MySQL transactions in a disposable, synthetic-only database."""

from datetime import date, datetime
import os
import re
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from src import config, deploy
from src.features import CATEGORIES, FEATURE_COLUMNS, NUMERIC
from src.registry import METRICS
from src.score import write_predictions


@unittest.skipUnless(os.getenv("RUN_DEPLOY_MYSQL_TESTS") == "1",
                     "Set RUN_DEPLOY_MYSQL_TESTS=1 for isolated MySQL deployment checks")
class DeploymentMySQLTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(config.DATABASE_URI, "DATABASE_URI is required")
        source_url = make_url(config.DATABASE_URI)
        self.assertIn(source_url.drivername, ("mysql+mysqlconnector", "mysql+pymysql"))
        self.database = "churn_deploy_test_" + uuid4().hex
        self.assertRegex(self.database, r"\Achurn_deploy_test_[0-9a-f]{32}\Z")
        self.assertNotEqual(self.database, source_url.database)
        timeout_key = "connect_timeout" if source_url.drivername.endswith("pymysql") else "connection_timeout"
        options = {"connect_args": {timeout_key: config.TIMEOUT_SECONDS}, "pool_pre_ping": True}
        self.admin = create_engine(source_url.set(database=""), **options)
        self.addCleanup(self.admin.dispose)
        self.engine = None
        with self.admin.begin() as connection:
            # No IF NOT EXISTS: only a database this fixture created may be dropped.
            connection.exec_driver_sql(f"CREATE DATABASE `{self.database}`")
        self.addCleanup(self.drop_fixture_database)
        self.url = source_url.set(database=self.database)
        self.engine = create_engine(self.url, **options)
        self.settings = {
            **config.PRODUCTION, "feature_set_version": "dated_mysql_test_v1", "model_id": None,
            "low_threshold": "0.25", "high_threshold": "0.5", "output_table": "model_predictions",
            "publication": "atomic_replace", "feature_refresh": "external", "alert_mode": "logging",
            "schedule_enabled": "false", "schedule_cron": "",
        }
        self.sink = MagicMock()
        deploy.ensure_schema(self.engine)
        with self.engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE customers (customer_id VARCHAR(20) PRIMARY KEY, "
                                       "signup_date DATE NOT NULL) ENGINE=InnoDB")
            connection.exec_driver_sql("CREATE TABLE subscriptions (subscription_id INT PRIMARY KEY AUTO_INCREMENT, "
                                       "customer_id VARCHAR(20) NOT NULL, status VARCHAR(20) NOT NULL, "
                                       "start_date DATE NOT NULL, end_date DATE NULL, "
                                       "FOREIGN KEY (customer_id) REFERENCES customers(customer_id)) ENGINE=InnoDB")
            for table, date_column in (("billing_events", "event_date"), ("usage_events", "event_date"),
                                       ("support_tickets", "opened_at")):
                connection.exec_driver_sql(f"CREATE TABLE {table} (event_id INT PRIMARY KEY AUTO_INCREMENT, "
                                           f"customer_id VARCHAR(20) NOT NULL, {date_column} DATETIME NOT NULL, "
                                           "FOREIGN KEY (customer_id) REFERENCES customers(customer_id)) ENGINE=InnoDB")
            feature_types = ", ".join(f"{column} {'DOUBLE' if column in NUMERIC else 'VARCHAR(50)'}"
                                      for column in FEATURE_COLUMNS)
            connection.exec_driver_sql("CREATE TABLE feature_store (customer_id VARCHAR(20) NOT NULL, "
                                       "observation_date DATE NOT NULL, feature_set_version VARCHAR(30) NOT NULL, "
                                       + feature_types + ", PRIMARY KEY (customer_id, observation_date, "
                                       "feature_set_version)) ENGINE=InnoDB")
            connection.exec_driver_sql("INSERT INTO customers VALUES ('a', '2026-01-01'), ('b', '2026-01-01')")
            connection.exec_driver_sql("INSERT INTO subscriptions (customer_id, status, start_date) "
                                       "VALUES ('a', 'active', '2026-01-01'), ('b', 'active', '2026-01-01')")
            for model_id, report_hash in (("model-a", "a" * 64), ("model-b", "b" * 64)):
                row = {"model_id": model_id, "trained_at": datetime(2026, 1, 1), "approved": 1,
                       "evaluation_scope": "production", "feature_set_version": "dated_mysql_test_v1",
                       "label_version": "future_mysql_test_v1", "artifact_path": "synthetic-fixture-only",
                       "report_sha256": report_hash, "dataset_sha256": "c" * 64,
                       "low_threshold": 0.25, "high_threshold": 0.5, **dict.fromkeys(METRICS, 0.5)}
                connection.execute(text(f"INSERT INTO model_registry ({', '.join(row)}) "
                                        f"VALUES ({', '.join(':' + name for name in row)})"), row)
            feature_row = {column: 1.0 if column in NUMERIC else sorted(CATEGORIES[column])[0]
                           for column in FEATURE_COLUMNS}
            frame = pd.DataFrame([{**feature_row, "customer_id": customer_id,
                                   "observation_date": date(2026, 1, 2),
                                   "feature_set_version": "dated_mysql_test_v1"} for customer_id in ("a", "b")])
            frame.to_sql("feature_store", connection, if_exists="append", index=False)
        report = {"artifact_sha256": {"model.json": "d" * 64, "preprocessor.joblib": "e" * 64},
                  "production_acceptance": {"future_window_days": 30}, "threshold": 0.5}
        for name, result in (("load_artifacts", (report, None, None, "a" * 64)),
                             ("predict_probabilities", np.array([0.2, 0.8]))):
            patcher = patch(f"src.deploy.{name}", return_value=result)
            patcher.start()
            self.addCleanup(patcher.stop)

    def drop_fixture_database(self):
        if self.engine is not None:
            self.engine.dispose()
        if re.fullmatch(r"churn_deploy_test_[0-9a-f]{32}", self.database) is None:
            raise AssertionError("Refusing to drop an unexpected database name")
        with self.admin.begin() as connection:
            connection.exec_driver_sql(f"DROP DATABASE `{self.database}`")

    def run_deployment(self, **changes):
        return deploy.run_deployment(**{
            "engine": self.engine, "settings": self.settings, "environment": "production",
            "database_uri": self.url.render_as_string(hide_password=False),
            "observation_date": date(2026, 1, 2), "alert_sink": self.sink, **changes,
        })

    def publication(self):
        with self.engine.connect() as connection:
            rows = connection.execute(text("SELECT " + deploy.PREDICTION_COLUMNS
                                           + " FROM model_predictions ORDER BY customer_id")).all()
            state = connection.execute(text("SELECT model_id, run_id FROM production_state")).one()
            return rows, state

    def test_real_schema_rerun_failure_rollback_and_model_rollback(self):
        first = self.run_deployment(model_id="model-a")
        self.assertEqual((first["status"], first["prediction_count"]), ("succeeded", 2))
        first_rows, _ = self.publication()
        rerun = self.run_deployment()
        self.assertEqual((rerun["reused_batch"], rerun["scored_at"]), (1, first["scored_at"]))
        self.assertEqual(self.publication()[0], first_rows)
        original = self.publication()

        def fail_after_insert(*args, **kwargs):
            write_predictions(*args, **kwargs)
            raise RuntimeError("synthetic failure after production inserts")

        with patch("src.deploy.write_predictions", side_effect=fail_after_insert):
            with self.assertRaises(deploy.DeploymentError) as failed:
                self.run_deployment(model_id="model-b")
        self.assertEqual(failed.exception.run["failure_stage"], "PUBLICATION")
        self.assertEqual(self.publication(), original)
        self.assertEqual(self.sink.emit.call_args.args[0]["event_type"], "deployment_failure")
        with self.engine.connect() as connection:
            saved = connection.execute(text("SELECT status, failure_stage FROM deployment_runs "
                                             "WHERE run_id = :run_id"), failed.exception.run).one()
        self.assertEqual(tuple(saved), ("failed", "PUBLICATION"))

        switched = self.run_deployment(model_id="model-b")
        restored = self.run_deployment(rollback=True)
        self.assertEqual((restored["status"], restored["model_id"]), ("rolled_back", "model-a"))
        self.assertEqual(restored["rollback_target_run_id"], rerun["run_id"])
        self.assertEqual(restored["previous_run_id"], switched["run_id"])
        self.assertEqual(tuple(self.publication()[1]), ("model-a", restored["run_id"]))
        self.assertEqual(self.sink.emit.call_args.args[0]["event_type"], "rollback")

        # A real missing source column must fail before changing the active batch.
        original = self.publication()
        with self.engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE usage_events DROP COLUMN event_date")
        with self.assertRaises(deploy.DeploymentError) as invalid_schema:
            self.run_deployment()
        self.assertEqual(invalid_schema.exception.run["failure_stage"], "PREFLIGHT")
        self.assertEqual(self.publication(), original)


if __name__ == "__main__":
    unittest.main()
