"""Opt-in integration: registers/scores the trusted benchmark in the configured MySQL DB."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.config import PROJECT_ROOT
from src.registry import register_model
from src.score import score_customers, write_predictions
from src.utils import get_engine


@unittest.skipUnless(os.getenv("RUN_MYSQL_TESTS") == "1", "Set RUN_MYSQL_TESTS=1 for live registration/scoring checks")
class Phase5MySQLTest(unittest.TestCase):
    def test_registration_scoring_rerun_and_rollback(self):
        artifact_dir = Path(os.getenv("TEST_MODEL_ARTIFACT_DIR", str(PROJECT_ROOT / "models" / "phase4_benchmark_verified")))
        model_id = register_model(artifact_dir)
        self.assertEqual(register_model(artifact_dir), model_id)
        params = {"model_id": model_id}
        engine = get_engine()
        score_query = text("""
            SELECT customer_id, churn_probability, risk_tier, feature_snapshot_sha256
            FROM model_predictions WHERE model_id = :model_id AND scoring_mode = 'benchmark'
            ORDER BY customer_id
        """)
        timestamp_query = text("""
            SELECT customer_id, scored_at FROM model_predictions
            WHERE model_id = :model_id AND scoring_mode = 'benchmark' ORDER BY customer_id
        """)
        try:
            first = score_customers(model_id, benchmark=True)
            with engine.connect() as connection:
                rows = connection.execute(score_query, params).all()
                raw_rows = connection.execute(text("SELECT COUNT(*) FROM customers_raw")).scalar_one()
                registry = connection.execute(text("""
                    SELECT approved, evaluation_scope FROM model_registry WHERE model_id = :model_id
                """), params).one()
            self.assertEqual(tuple(registry), (0, "benchmark"))
            self.assertEqual(first["rows"], raw_rows)
            self.assertEqual(len(rows), raw_rows)
            self.assertTrue(all(0 <= row.churn_probability <= 1 for row in rows))
            second = score_customers(model_id, benchmark=True)
            self.assertEqual(first, second)
            with engine.connect() as connection:
                self.assertEqual(connection.execute(score_query, params).all(), rows)
                timestamps = connection.execute(timestamp_query, params).all()
            with self.assertRaisesRegex(ValueError, "approved"):
                score_customers(model_id)
            with self.assertRaisesRegex(ValueError, "production|active"):
                score_customers()

            def fail_after_inserts(*args, **kwargs):
                write_predictions(*args, **kwargs)
                raise RuntimeError("injected failure after prediction inserts")

            with patch("src.score.write_predictions", side_effect=fail_after_inserts):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    score_customers(model_id, benchmark=True)
            with engine.connect() as connection:
                self.assertEqual(connection.execute(score_query, params).all(), rows)
                self.assertEqual(connection.execute(timestamp_query, params).all(), timestamps)
            with self.assertRaisesRegex(DBAPIError, "3819|Check constraint"):
                with engine.begin() as connection:
                    connection.execute(text("UPDATE model_registry SET approved = 1 WHERE model_id = :model_id"), params)
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
