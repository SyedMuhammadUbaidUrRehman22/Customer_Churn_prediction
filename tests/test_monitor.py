"""Monitoring uses retained cohorts, never the snapshot labels or live features."""

from datetime import date, datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text

from src import config
from src.features import CATEGORIES, FEATURE_COLUMNS, NUMERIC
from src.monitor import feature_drift, run_monitoring, validate_policy, write_report
from src.monitoring_history import batch_hash


class MonitoringTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        self.sink = MagicMock()
        self.policy = dict.fromkeys(("numeric_mean_shift_max", "categorical_tv_max", "precision_min",
                                     "retrain_interval_days"))
        self.features = pd.DataFrame([
            {"customer_id": f"c{index:02}", **{column: 1.0 if column in NUMERIC else sorted(CATEGORIES[column])[0]
                                               for column in FEATURE_COLUMNS}}
            for index in range(20)
        ])
        self.features["monthly_charges"] = np.arange(20, dtype=float) + 10
        self.features["gender"] = "Female"
        self.history = self.features.assign(churn_probability=[0.2] * 10 + [0.8] * 10,
                                            risk_tier=["low"] * 10 + ["high"] * 10)
        self.current = self.batch("a" * 64, date(2026, 1, 2), self.features)
        reference_features = self.features.assign(monthly_charges=self.features.monthly_charges - 10, gender="Male")
        self.reference = self.batch("b" * 64, date(2026, 1, 1), reference_features)
        self.reference_history = reference_features.assign(churn_probability=self.history.churn_probability,
                                                            risk_tier=self.history.risk_tier)
        pd.DataFrame([self.current, self.reference]).to_sql("scoring_batches", self.engine, index=False)
        pd.concat([self.history.assign(batch_sha256=self.current["batch_sha256"]),
                   self.reference_history.assign(batch_sha256=self.reference["batch_sha256"])],
                  ignore_index=True).to_sql("scoring_history", self.engine, index=False)
        self.outcomes = pd.DataFrame({
            "batch_sha256": self.current["batch_sha256"], "customer_id": self.history.customer_id,
            "label_version": "future_labels_v1", "label_window_end": date(2026, 2, 1),
            "observed_through": date(2026, 2, 1), "available_at": datetime(2026, 2, 2),
            "churned": [0] * 10 + [1] * 8 + [0] * 2, "evidence_reference": "verified-source/cohort-1",
        })
        self.write_outcomes(self.outcomes.iloc[:0])

    def batch(self, digest, observed, features):
        frame = features[["customer_id", *FEATURE_COLUMNS]].copy()
        frame[NUMERIC] = frame[NUMERIC].astype(float)
        metadata = {"run_id": digest[:32], "model_id": "production-model",
                    "feature_set_version": "dated_features_v1", "label_version": "future_labels_v1",
                    "observation_date": observed, "scored_at": datetime.combine(observed, datetime.min.time()),
                    "customer_count": len(frame), "report_sha256": "c" * 64,
                    "low_threshold": 0.25, "high_threshold": 0.5,
                    "future_window_days": 30, "evaluation_threshold": 0.5,
                    "baseline_pr_auc": 0.85, "baseline_precision": 0.9, "baseline_recall": 0.8,
                    "trained_at": datetime(2025, 12, 1)}
        metadata["feature_snapshot_sha256"], metadata["batch_sha256"] = batch_hash(
            metadata, frame, observed, metadata["low_threshold"], metadata["high_threshold"])
        return metadata

    def write_outcomes(self, frame):
        frame.to_sql("monitoring_outcomes", self.engine, if_exists="replace", index=False)

    def report(self, **changes):
        return run_monitoring(**{"engine": self.engine, "batch_id": self.current["batch_sha256"],
                                 "as_of": date(2026, 2, 10), "policy": self.policy,
                                 "alert_sink": self.sink, **changes})

    def test_retained_drift_and_mature_performance_are_exact_and_repeatable(self):
        self.write_outcomes(self.outcomes)
        report = self.report()
        self.assertEqual(report, self.report(batch_id=None))
        self.assertEqual(report["reference_batch"]["batch_sha256"], self.reference["batch_sha256"])
        drift = {row["feature"]: row for row in report["drift"]["features"]}
        self.assertAlmostEqual(drift["monthly_charges"]["normalized_mean_shift"], 10 / np.std(np.arange(20)))
        self.assertEqual(drift["gender"]["total_variation"], 1)
        self.assertEqual(drift["tenure_months"]["normalized_mean_shift"], 0)
        metrics = report["performance"]["metrics"]
        self.assertEqual((metrics["precision"], metrics["recall"], metrics["true_positives"]), (0.8, 1.0, 8))
        self.assertAlmostEqual(metrics["pr_auc_average_precision"], 0.8)
        self.assertAlmostEqual(metrics["brier_score"], 0.1)
        self.assertAlmostEqual(report["performance"]["baseline_comparison"]["precision_delta"], -0.1)
        self.assertEqual(report["retraining"]["status"], "unconfigured")
        self.assertFalse(report["retraining"]["automatic_retraining"])
        self.sink.emit.assert_not_called()
        self.assertEqual(set(inspect(self.engine).get_table_names()),
                         {"scoring_batches", "scoring_history", "monitoring_outcomes"})

    def test_incomplete_immature_and_unavailable_outcomes_never_become_negatives(self):
        for frame, as_of, reason in ((self.outcomes, date(2026, 1, 31), "future_window_not_mature"),
                                     (self.outcomes.iloc[:-1], date(2026, 2, 10), "outcomes_incomplete_or_unavailable"),
                                     (self.outcomes, date(2026, 2, 1), "outcomes_incomplete_or_unavailable")):
            self.write_outcomes(frame)
            with self.subTest(reason=reason):
                performance = self.report(as_of=as_of)["performance"]
                self.assertEqual((performance["status"], performance["reason"]), ("pending", reason))
                self.assertIsNone(performance["metrics"])
        self.write_outcomes(self.outcomes.assign(churned=0))
        report = self.report()
        self.assertIsNone(report["performance"]["metrics"]["recall"])
        self.assertIsNone(report["performance"]["metrics"]["roc_auc"])
        self.assertIsNone(report["performance"]["metrics"]["pr_auc_average_precision"])

    def test_outcome_version_window_provenance_and_coverage_fail_closed(self):
        invalid = [self.outcomes.assign(label_version="other-labels"),
                   self.outcomes.assign(label_window_end=date(2026, 1, 31)),
                   self.outcomes.assign(observed_through=date(2026, 1, 31)),
                   self.outcomes.assign(available_at=datetime(2026, 1, 1)),
                   self.outcomes.assign(available_at="private-source-invalid-date"),
                   self.outcomes.assign(evidence_reference=" "), self.outcomes.assign(churned=2),
                   self.outcomes.assign(customer_id="unknown"),
                   pd.concat([self.outcomes, self.outcomes.iloc[:1]])]
        for index, frame in enumerate(invalid):
            self.write_outcomes(frame)
            with self.subTest(case=index), self.assertRaises(ValueError) as caught:
                self.report()
            self.assertNotIn("private-source-invalid-date", str(caught.exception))
            self.assertEqual(self.sink.emit.call_args.args[0]["event_type"], "monitoring_failure")

    def test_reference_history_and_as_of_gates(self):
        for changes in ({"reference_batch_id": self.current["batch_sha256"]},
                        {"reference_batch_id": "f" * 64}, {"batch_id": "invalid"},
                        {"as_of": date(2026, 1, 1)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.report(**changes)
        incompatible = {**self.reference, "feature_set_version": "other_v1"}
        _, incompatible_digest = batch_hash(incompatible, self.reference_history[["customer_id", *FEATURE_COLUMNS]],
                                           incompatible["observation_date"], incompatible["low_threshold"],
                                           incompatible["high_threshold"])
        with self.engine.begin() as connection:
            params = {"original": self.reference["batch_sha256"], "changed": incompatible_digest}
            connection.execute(text("UPDATE scoring_batches SET feature_set_version = 'other_v1', "
                                    "batch_sha256 = :changed WHERE batch_sha256 = :original"), params)
            connection.execute(text("UPDATE scoring_history SET batch_sha256 = :changed "
                                    "WHERE batch_sha256 = :original"), params)
        with self.assertRaisesRegex(ValueError, "same feature version"):
            self.report(reference_batch_id=incompatible_digest)
        self.assertEqual(self.report()["drift"]["status"], "pending")
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE scoring_history SET monthly_charges = 999 "
                                    "WHERE batch_sha256 = :digest AND customer_id = 'c00'"),
                               {"digest": self.current["batch_sha256"]})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.report()

    def test_retained_model_and_risk_metadata_tampering_breaks_batch_identity(self):
        for column, value in (("model_id", "another-production-model"), ("low_threshold", 0.3),
                              ("high_threshold", 0.6), ("report_sha256", "d" * 64)):
            with self.engine.begin() as connection:
                connection.execute(text(f"UPDATE scoring_batches SET {column} = :value "
                                        "WHERE batch_sha256 = :digest"),
                                   {"value": value, "digest": self.current["batch_sha256"]})
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "identity"):
                self.report()
            with self.engine.begin() as connection:
                connection.execute(text(f"UPDATE scoring_batches SET {column} = :value "
                                        "WHERE batch_sha256 = :digest"),
                                   {"value": self.current[column], "digest": self.current["batch_sha256"]})

    def test_configured_recommendations_constant_drift_and_threshold_comparison(self):
        changed = self.features.assign(tenure_months=2)
        row = {r["feature"]: r for r in feature_drift(changed, self.features)}["tenure_months"]
        self.assertTrue(row["constant_reference_changed"])
        self.assertIsNone(row["normalized_mean_shift"])
        self.write_outcomes(self.outcomes)
        policy = {"numeric_mean_shift_max": 1, "categorical_tv_max": 0.5,
                  "precision_min": 0.9, "retrain_interval_days": 30}
        report = self.report(policy=policy)
        self.assertEqual(report["retraining"]["status"], "recommended")
        self.assertEqual(len(report["retraining"]["reasons"]), 4)
        self.assertEqual(self.sink.emit.call_args.args[0]["event_type"], "monitoring_retraining_recommended")
        for key, value in (("numeric_mean_shift_max", -1), ("categorical_tv_max", 1.1),
                           ("precision_min", np.nan), ("precision_min", True), ("retrain_interval_days", 0)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_policy({**self.policy, key: value})
        with self.engine.begin() as connection:
            connection.execute(text("UPDATE scoring_batches SET evaluation_threshold = 0.7"))
        comparison = self.report()["performance"]["baseline_comparison"]
        self.assertFalse(comparison["comparable"])
        self.assertIsNone(comparison["precision_delta"])
        self.assertIsNotNone(comparison["pr_auc_delta"])

    def test_artifacts_are_deterministic_and_dashboard_escapes_untrusted_text(self):
        report = self.report()
        report["evidence_scope"] = "Synthetic test data <script>alert(1)</script>"
        report["batch"]["model_id"] = "<img src=x onerror=alert(2)>"
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, directory)
            paths = [Path(directory) / name for name in ("report.json", "report.md", "dashboard.html")]
            original = [path.read_bytes() for path in paths]
            write_report(report, directory)
            self.assertEqual(original, [path.read_bytes() for path in paths])
            self.assertEqual(json.loads(original[0]), report)
            html = paths[2].read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertNotIn("<img", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertIn("Synthetic test data", paths[1].read_text(encoding="utf-8"))
            self.assertIn('lang="en"', html)
        self.sink.emit.side_effect = RuntimeError("provider unavailable")
        self.report(policy={**self.policy, "retrain_interval_days": 1})


if __name__ == "__main__":
    unittest.main()
