"""Opt-in monitoring checks reuse the isolated synthetic MySQL deployment fixture."""

from datetime import date, datetime
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sqlalchemy import text

from src import deploy
from src.features import FEATURE_COLUMNS
from tests import test_deploy_mysql as fixtures


@unittest.skipUnless(os.getenv("RUN_MONITOR_MYSQL_TESTS") == "1",
                     "Set RUN_MONITOR_MYSQL_TESTS=1 for isolated MySQL monitoring checks")
class MonitoringMySQLTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DeploymentMySQLTest(
            methodName="test_real_schema_rerun_failure_rollback_and_model_rollback")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def history(self):
        with self.fixture.engine.connect() as connection:
            batches = connection.execute(text(
                "SELECT batch_sha256, model_id, observation_date, scored_at FROM scoring_batches "
                "ORDER BY observation_date, batch_sha256")).all()
            rows = connection.execute(text(
                "SELECT batch_sha256, customer_id, monthly_charges, gender, churn_probability, risk_tier "
                "FROM scoring_history ORDER BY batch_sha256, customer_id")).all()
        return batches, rows

    def test_archive_atomicity_history_survival_and_matured_performance(self):
        from src.monitoring_history import archive_batch
        from src.monitor import run_monitoring, write_report

        with patch("src.deploy.utcnow", return_value=datetime(2026, 1, 4)):
            first = self.fixture.run_deployment(model_id="model-a")
            initial_history = self.history()
            repeated = self.fixture.run_deployment()
            self.assertEqual(repeated["reused_batch"], 1)
            self.assertEqual(self.history(), initial_history)
            self.assertEqual((len(initial_history[0]), len(initial_history[1])), (1, 2))
            initial_publication = self.fixture.publication()

            def archive_then_fail(*args, **kwargs):
                archive_batch(*args, **kwargs)
                raise RuntimeError("synthetic failure after archive inserts")

            with patch("src.deploy.archive_batch", side_effect=archive_then_fail):
                with self.assertRaises(deploy.DeploymentError) as failed:
                    self.fixture.run_deployment(model_id="model-b")
            self.assertEqual(failed.exception.run["failure_stage"], "PUBLICATION")
            self.assertEqual(self.fixture.publication(), initial_publication)
            self.assertEqual(self.history(), initial_history)

            with self.fixture.engine.begin() as connection:
                columns = ", ".join(FEATURE_COLUMNS)
                connection.execute(text(
                    f"INSERT INTO feature_store (customer_id, observation_date, feature_set_version, {columns}) "
                    f"SELECT customer_id, :next_date, feature_set_version, {columns} FROM feature_store "
                    "WHERE observation_date = :first_date"),
                    {"next_date": date(2026, 1, 3), "first_date": date(2026, 1, 2)})
                connection.execute(text(
                    "UPDATE feature_store SET monthly_charges = 2, gender = 'Male' "
                    "WHERE observation_date = :next_date"), {"next_date": date(2026, 1, 3)})
            current = self.fixture.run_deployment(model_id="model-b", observation_date=date(2026, 1, 3))

        retained = self.history()
        self.assertEqual((len(retained[0]), len(retained[1])), (2, 4))
        first_rows = [row for row in retained[1] if row.batch_sha256 == first["batch_sha256"]]
        self.assertEqual(first_rows, initial_history[1])
        self.assertEqual({row.model_id for row in self.fixture.publication()[0]}, {"model-b"})
        # Monitoring must read retained features, never the replaceable source snapshot.
        with self.fixture.engine.begin() as connection:
            connection.execute(text("UPDATE feature_store SET monthly_charges = 999"))
        self.assertEqual(self.history(), retained)

        policy = {"numeric_mean_shift_max": None, "categorical_tv_max": None,
                  "precision_min": None, "retrain_interval_days": None}
        options = {"engine": self.fixture.engine, "batch_id": current["batch_sha256"],
                   "reference_batch_id": first["batch_sha256"], "as_of": date(2026, 2, 10),
                   "policy": policy, "alert_sink": self.fixture.sink}
        with tempfile.TemporaryDirectory() as temporary:
            report = run_monitoring(**options, output_dir=Path(temporary))
            self.assertEqual(report["performance"]["status"], "pending")
            with self.fixture.engine.begin() as connection:
                statement = text(
                    "INSERT INTO monitoring_outcomes (batch_sha256, customer_id, label_version, "
                    "label_window_end, observed_through, available_at, churned, evidence_reference) "
                    "VALUES (:batch, :customer, 'future_mysql_test_v1', :window_end, :window_end, "
                    ":available_at, :churned, 'synthetic-test-only')")
                connection.execute(statement, [
                    {"batch": current["batch_sha256"], "customer": customer,
                     "window_end": date(2026, 2, 2), "available_at": datetime(2026, 2, 3),
                     "churned": churned} for customer, churned in (("a", 0), ("b", 1))])
            report = run_monitoring(**options, output_dir=Path(temporary))
            self.assertEqual(report["performance"]["status"], "complete")
            self.assertEqual(report["performance"]["metrics"]["precision"], 1.0)
            self.assertEqual(report["performance"]["metrics"]["recall"], 1.0)
            for filename in ("report.json", "report.md", "dashboard.html"):
                self.assertTrue((Path(temporary) / filename).is_file(), filename)
            if os.getenv("TEST_MONITOR_OUTPUT_DIR"):
                report["evidence_scope"] = "Synthetic integration-test data; no production observations or approval"
                write_report(report, Path(os.environ["TEST_MONITOR_OUTPUT_DIR"]))


if __name__ == "__main__":
    unittest.main()
