"""Synthetic production contracts only; these checks never approve the Telco model."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.config import FEATURE_SET_VERSION
from src.features import FEATURE_COLUMNS
from src.registry import METRICS, load_artifacts
from src.score import build_prediction_frame, validate_prediction_batch, write_predictions


class ProductionArtifactTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = {"model.json": b"synthetic model", "preprocessor.joblib": b"synthetic preprocessor"}
        self.report = {
            "trained_at": "2026-09-25T00:00:00+00:00", "approved": True,
            "evaluation_scope": "production", "evaluation_design": "temporal", "promotion_blockers": [],
            "feature_set_version": "dated_contract_test_v1", "label_version": "future_window_test_v1",
            "features": list(FEATURE_COLUMNS), "dataset_sha256": "a" * 64,
            "test": {name: 0.5 for name in METRICS.values()},
            "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__,
                         "python": platform.python_version(), "joblib": joblib.__version__},
            "artifact_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in self.artifacts.items()},
            "production_acceptance": {"temporal_evaluation": True, "future_window_days": 30,
                                      "stakeholder_approved": True, "evidence_reference": "synthetic test only",
                                      "low_threshold": 0.25, "high_threshold": 0.5},
        }
        self.registry = {"model_id": "synthetic-production", "approved": 1, "evaluation_scope": "production",
                         "trained_at": datetime(2026, 9, 25), "feature_set_version": "dated_contract_test_v1",
                         "label_version": "future_window_test_v1", "dataset_sha256": "a" * 64,
                         "low_threshold": 0.25, "high_threshold": 0.5,
                         **{name: 0.5 for name in METRICS}}

    def write_artifacts(self, directory, report):
        for name, contents in self.artifacts.items():
            (directory / name).write_bytes(contents)
        contents = json.dumps(report).encode()
        (directory / "report.json").write_bytes(contents)
        return {**self.registry, "report_sha256": hashlib.sha256(contents).hexdigest()}

    def test_valid_production_contract_is_explicit_and_benchmark_default_stays_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            registry = self.write_artifacts(directory, self.report)
            preprocess = MagicMock(feature_names_in_=list(FEATURE_COLUMNS))
            preprocess.get_feature_names_out.return_value = list(FEATURE_COLUMNS)
            with patch("src.registry.joblib.load", return_value=preprocess) as deserialize, \
                    patch("src.registry.xgboost.XGBClassifier") as constructor:
                constructor.return_value.n_features_in_ = len(FEATURE_COLUMNS)
                loaded, _, _, digest = load_artifacts(directory, production_registry=registry)
                self.assertEqual(loaded, self.report)
                self.assertEqual(digest, registry["report_sha256"])
                deserialize.assert_called_once()
                constructor.return_value.load_model.assert_called_once_with(bytearray(self.artifacts["model.json"]))
                with self.assertRaises(ValueError):
                    load_artifacts(directory)

    def test_production_metadata_and_acceptance_fail_before_deserialization(self):
        cases = []
        for key, value in (("approved", False), ("evaluation_scope", "benchmark"),
                           ("evaluation_design", "non-temporal"), ("promotion_blockers", ["pending"]),
                           ("feature_set_version", FEATURE_SET_VERSION), ("label_version", FEATURE_SET_VERSION),
                           ("trained_at", "2026-09-24T00:00:00+00:00"), ("dataset_sha256", "b" * 64),
                           ("features", list(reversed(FEATURE_COLUMNS)))):
            cases.append(({**self.report, key: value}, {}))
        for key, value in (("temporal_evaluation", False), ("stakeholder_approved", False),
                           ("future_window_days", 0), ("future_window_days", True),
                           ("evidence_reference", " "), ("low_threshold", 0.1), ("high_threshold", 0.8)):
            cases.append(({**self.report, "production_acceptance": {**self.report["production_acceptance"], key: value}}, {}))
        for dependency in self.report["versions"]:
            cases.append(({**self.report, "versions": {**self.report["versions"], dependency: "incompatible"}}, {}))
        for change in ({"approved": 0}, {"evaluation_scope": "benchmark"},
                       {"feature_set_version": FEATURE_SET_VERSION}, {"pr_auc": 0.8}):
            cases.append((self.report, change))
        with tempfile.TemporaryDirectory() as temporary, patch("src.registry.joblib.load") as deserialize:
            directory = Path(temporary)
            for index, (report, change) in enumerate(cases):
                registry = {**self.write_artifacts(directory, report), **change}
                with self.subTest(case=index), self.assertRaises(ValueError):
                    load_artifacts(directory, production_registry=registry)
            deserialize.assert_not_called()

    def test_missing_or_changed_artifacts_fail_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary, patch("src.registry.joblib.load") as deserialize:
            directory = Path(temporary)
            for name in ("report.json", "model.json", "preprocessor.joblib"):
                for missing in (True, False):
                    registry = self.write_artifacts(directory, self.report)
                    if missing:
                        (directory / name).unlink()
                    else:
                        (directory / name).write_bytes((directory / name).read_bytes() + b"tampered")
                    with self.subTest(name=name, missing=missing), self.assertRaises((ValueError, FileNotFoundError)):
                        load_artifacts(directory, production_registry=registry)
            deserialize.assert_not_called()


class ProductionPredictionTest(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame({"customer_id": ["a", "b", "c"]})
        self.scored_at = datetime(2026, 9, 26)
        self.version = "dated_contract_test_v1"
        self.predictions = build_prediction_frame("production-model", self.source, [0.1, 0.3, 0.8],
                                                   0.25, 0.5, "a" * 64, "production", self.version, self.scored_at)

    def validate(self, frame):
        validate_prediction_batch(frame, self.source.customer_id, "production-model", self.version,
                                  "a" * 64, 0.25, 0.5, self.scored_at)

    def test_complete_batch_and_all_required_prediction_guards(self):
        self.validate(self.predictions)
        self.validate(self.predictions.iloc[::-1])
        invalid = [self.predictions.iloc[:-1], self.predictions.iloc[:0],
                   self.predictions.drop(columns="scored_at"),
                   self.predictions.assign(customer_id=["a", "a", "c"]),
                   self.predictions.assign(customer_id=["a", "b", "unexpected"]),
                   self.predictions.assign(model_id="another-model"),
                   self.predictions.assign(feature_set_version=FEATURE_SET_VERSION),
                   self.predictions.assign(feature_snapshot_sha256="b" * 64),
                   self.predictions.assign(scoring_mode="benchmark"),
                   self.predictions.assign(scored_at=pd.NaT),
                   self.predictions.assign(scored_at=[self.scored_at, self.scored_at, datetime(2026, 9, 25)]),
                   self.predictions.assign(risk_tier="invalid"), self.predictions.assign(risk_tier="low")]
        invalid += [self.predictions.assign(churn_probability=value) for value in (np.nan, np.inf, -0.1, 1.1)]
        for index, frame in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.validate(frame)

    def test_writer_rejects_bad_batch_before_sql_and_unapproved_model_before_delete(self):
        connection = MagicMock()
        with self.assertRaises(ValueError):
            write_predictions(connection, "production-model", self.source, [np.nan] * 3,
                              0.25, 0.5, "a" * 64, scoring_mode="production", feature_version=self.version)
        connection.execute.assert_not_called()
        for registry in (None, {"approved": 0, "evaluation_scope": "production", "feature_set_version": self.version},
                         {"approved": 1, "evaluation_scope": "benchmark", "feature_set_version": self.version},
                         {"approved": 1, "evaluation_scope": "production", "feature_set_version": "wrong"}):
            connection.reset_mock()
            connection.execute.return_value.mappings.return_value.first.return_value = registry
            with self.subTest(registry=registry), self.assertRaises(ValueError):
                write_predictions(connection, "production-model", self.source, [0.1, 0.3, 0.8], 0.25, 0.5,
                                  "a" * 64, scoring_mode="production", feature_version=self.version)
            self.assertEqual(connection.execute.call_count, 1)
            self.assertTrue(str(connection.execute.call_args.args[0]).startswith("SELECT"))


if __name__ == "__main__":
    unittest.main()
