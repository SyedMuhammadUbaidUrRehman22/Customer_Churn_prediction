import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch, sentinel

import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
import xgboost

from src.config import FEATURE_SET_VERSION, LABEL_VERSION
from src.features import FEATURE_COLUMNS, validate_feature_frame
from src.ingest import load_and_validate
from src.registry import load_artifacts
from src.score import risk_tiers, score_customers, write_predictions
from src.train import main as train_main


class RegistryScoringTest(unittest.TestCase):
    def test_label_free_features_and_risk_boundaries(self):
        frame = load_and_validate().drop(columns="churned")
        validate_feature_frame(frame)
        for invalid in (frame.iloc[:0], frame.drop(columns="contract"),
                        frame.assign(contract="unknown"),
                        frame.assign(monthly_charges=float("inf")),
                        frame.assign(tenure_months=1.5),
                        frame.assign(total_charges=np.nan),
                        frame.assign(customer_id="duplicate")):
            with self.subTest(columns=list(invalid.columns)), self.assertRaises(ValueError):
                validate_feature_frame(invalid)
        np.testing.assert_array_equal(
            risk_tiers(np.array([0, 0.249, 0.25, 0.499, 0.5, 1]), 0.25, 0.5),
            ["low", "low", "medium", "medium", "high", "high"],
        )
        for scores, low, high in (([np.nan], 0.25, 0.5), ([np.inf], 0.25, 0.5),
                                  ([-0.1], 0.25, 0.5), ([1.1], 0.25, 0.5),
                                  ([0.2], 0.5, 0.25), ([0.2], 0.5, 0.5),
                                  ([0.2], 0, np.nan)):
            with self.subTest(scores=scores, low=low, high=high), self.assertRaises(ValueError):
                risk_tiers(np.array(scores), low, high)

    def test_tampered_artifacts_fail_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifacts = {"model.json": b"{}", "preprocessor.joblib": b"synthetic artifact"}
            report = {
                "trained_at": "2026-09-25T00:00:00+00:00", "approved": False,
                "evaluation_design": "non-temporal stratified snapshot benchmark, 60/20/20; seed 42",
                "feature_set_version": FEATURE_SET_VERSION, "label_version": LABEL_VERSION,
                "features": list(FEATURE_COLUMNS), "threshold": 0.5,
                "dataset_sha256": "a" * 64,
                "promotion_blockers": ["No dated events or temporal holdout"],
                "importance_gate_passed": True,
                "test": {"pr_auc_average_precision": 0.65, "roc_auc": 0.84,
                         "precision_at_top_decile": 0.78, "recall_at_top_decile": 0.29,
                         "precision": 0.59, "recall": 0.72, "f1": 0.65, "brier_score": 0.16},
                "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
                "artifact_sha256": {name: hashlib.sha256(data).hexdigest()
                                    for name, data in artifacts.items()},
            }
            report_bytes = json.dumps(report).encode()
            expected_digest = hashlib.sha256(report_bytes).hexdigest()
            for changed in ("report.json", "model.json", "preprocessor.joblib"):
                for name, data in {**artifacts, "report.json": report_bytes}.items():
                    (output / name).write_bytes(data + (b" " if name == changed else b""))
                with self.subTest(changed=changed), patch("src.registry.joblib.load") as deserialize:
                    with self.assertRaisesRegex(ValueError, "(?i)hash|sha|digest|integrity|checksum"):
                        load_artifacts(output, expected_report_sha256=expected_digest)
                    deserialize.assert_not_called()
            for name, data in artifacts.items():
                (output / name).write_bytes(data)
            for mismatch in ({"features": list(reversed(FEATURE_COLUMNS))},
                             {"feature_set_version": "unsupported"},
                             {"versions": {**report["versions"], "xgboost": "0.0.0"}}):
                (output / "report.json").write_text(json.dumps({**report, **mismatch}), encoding="utf-8")
                with self.subTest(mismatch=mismatch), patch("src.registry.joblib.load") as deserialize:
                    with self.assertRaisesRegex(ValueError, "(?i)feature|version"):
                        load_artifacts(output)
                    deserialize.assert_not_called()

    def test_training_cli_registers_only_after_success(self):
        output = Path("models/cli-test")
        with patch("sys.argv", ["train", "--output-dir", str(output), "--trials", "1", "--register"]), \
                patch("src.train.load_training_frame", return_value=sentinel.frame), \
                patch("src.train.train_benchmark", return_value={"test": {}}) as train, \
                patch("src.registry.register_model", return_value="model-example") as register, \
                patch("builtins.print"):
            sequence = MagicMock()
            sequence.attach_mock(train, "train")
            sequence.attach_mock(register, "register")
            train_main()
            self.assertEqual(sequence.mock_calls, [call.train(sentinel.frame, output, 1),
                                                   call.register(output)])
            register.reset_mock()
            train.side_effect = ValueError("Training validation rejected the run")
            with self.assertRaisesRegex(ValueError, "Training validation"):
                train_main()
            register.assert_not_called()

    def test_production_fails_closed_and_benchmark_requires_explicit_model(self):
        engine, connection = MagicMock(), MagicMock()
        engine.connect.return_value.execution_options.return_value.__enter__.return_value = connection
        connection.execute.return_value.mappings.return_value.first.return_value = None
        with patch("src.score.get_engine", return_value=engine), \
                patch("src.score.load_artifacts") as load, patch("src.score.write_predictions") as write:
            with self.assertRaisesRegex(ValueError, "approved"):
                score_customers()
            for scope in ("benchmark", "production"):
                connection.execute.return_value.mappings.return_value.first.return_value = {
                    "model_id": "example", "approved": 1, "evaluation_scope": scope,
                    "feature_set_version": FEATURE_SET_VERSION,
                }
                with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "(?i)production|active"):
                    score_customers(model_id="example")
            load.assert_not_called()
            write.assert_not_called()
        with patch("src.score.get_engine") as get_engine:
            with self.assertRaises(ValueError):
                score_customers(benchmark=True)
            get_engine.assert_not_called()

    def test_failed_replacement_rolls_back_and_other_models_modes_survive(self):
        engine = create_engine("sqlite://")
        try:
            with engine.begin() as connection:
                connection.exec_driver_sql("""
                    CREATE TABLE model_predictions (
                        customer_id TEXT NOT NULL CHECK (customer_id != 'reject'),
                        model_id TEXT NOT NULL, scoring_mode TEXT NOT NULL,
                        scored_at DATETIME NOT NULL, feature_set_version TEXT NOT NULL,
                        feature_snapshot_sha256 TEXT NOT NULL,
                        churn_probability REAL NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
                        risk_tier TEXT NOT NULL,
                        PRIMARY KEY (customer_id, model_id, scoring_mode)
                    )
                """)
                for model in ("model-a", "model-b"):
                    write_predictions(connection, model, pd.DataFrame({"customer_id": ["old"]}),
                                      np.array([0.1]), 0.25, 0.5, "a" * 64)
                connection.execute(text("""
                    INSERT INTO model_predictions
                    SELECT customer_id, model_id, 'production', scored_at,
                           feature_set_version, feature_snapshot_sha256, churn_probability, risk_tier
                    FROM model_predictions WHERE model_id = 'model-a'
                """))
            query = text("""
                SELECT customer_id, model_id, scoring_mode, scored_at, feature_set_version,
                       feature_snapshot_sha256, churn_probability, risk_tier
                FROM model_predictions ORDER BY model_id, scoring_mode, customer_id
            """)
            with engine.connect() as connection:
                original = connection.execute(query).all()
            with self.assertRaises(IntegrityError):
                with engine.begin() as connection:
                    write_predictions(connection, "model-a",
                                      pd.DataFrame({"customer_id": ["new", "reject"]}),
                                      np.array([0.3, 0.7]), 0.25, 0.5, "b" * 64)
            with engine.connect() as connection:
                self.assertEqual(connection.execute(query).all(), original)
            with engine.begin() as connection:
                write_predictions(connection, "model-a", pd.DataFrame({"customer_id": ["new"]}),
                                  np.array([0.7]), 0.25, 0.5, "b" * 64)
            with engine.connect() as connection:
                rows = connection.execute(query).mappings().all()
            self.assertEqual(len(rows), 3)
            current = next(row for row in rows if row["model_id"] == "model-a"
                           and row["scoring_mode"] == "benchmark")
            self.assertEqual((current["customer_id"], current["risk_tier"],
                              current["feature_snapshot_sha256"]), ("new", "high", "b" * 64))
            self.assertTrue(all(row["customer_id"] == "old" for row in rows if row != current))
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
