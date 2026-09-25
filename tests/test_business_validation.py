import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from src.business_validation import (analyze, build_report, load_inputs, rank_predictions,
                                     run_validation, validation_rows)
from src.config import BUSINESS_ACCEPTANCE, FEATURE_SET_VERSION
from src.features import FEATURE_COLUMNS, NUMERIC
from src.ingest import load_and_validate
from src.registry import METRICS
from src.utils import get_engine


class BusinessValidationTest(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame({
            "customer_id": ["d", "b", "a", "c", "e"],
            "churn_probability": [0.1, 0.8, 0.8, 0.25, 0.5],
            "risk_tier": ["low", "high", "high", "medium", "high"],
            "scored_at": [pd.Timestamp("2026-09-25")] * 5,
            "feature_snapshot_sha256": ["a" * 64] * 5,
        })
        self.labels = pd.DataFrame({"customer_id": ["a", "b", "c", "d", "e"], "churned": [1, 0, 1, 0, 1]})
        self.registry = {"model_id": "example", "feature_set_version": FEATURE_SET_VERSION,
                         "label_version": FEATURE_SET_VERSION, "low_threshold": 0.25,
                         "high_threshold": 0.5, "report_sha256": "b" * 64, "dataset_sha256": "c" * 64}
        self.evaluation = {"trained_at": "2026-09-25T00:00:00+00:00",
                           "test": {"pr_auc_average_precision": 0.6},
                           "test_prevalence_baseline_ap": 0.3,
                           "split_counts": {"train": {"rows": 3, "churned": 1},
                                            "validation": {"rows": 1, "churned": 1},
                                            "test": {"rows": 1, "churned": 1}},
                           "feature_importance": {"contract": 0.4}}

    def test_ranking_precision_recall_tiers_and_repeatability(self):
        ranked = rank_predictions(self.predictions, self.labels, 0.25, 0.5)
        self.assertEqual(ranked.customer_id.tolist(), ["a", "b", "e", "c", "d"])
        result = analyze(ranked, 0.25, 0.5, [2])
        capacity = result["outreach"][-1]
        self.assertEqual((capacity["selected"], capacity["churners"], capacity["precision"], capacity["recall"]), (2, 1, 0.5, 1 / 3))
        self.assertEqual(capacity["selected_prevalence"], capacity["precision"])
        self.assertEqual(capacity["cumulative_churn_capture"], capacity["recall"])
        self.assertEqual([row["customers"] for row in result["tiers"]], [3, 1, 1])
        self.assertEqual(sum(row["customers"] for row in result["tiers"]), len(ranked))
        first = build_report(self.registry, self.evaluation, ranked, [2])
        shuffled = rank_predictions(self.predictions.sample(frac=1, random_state=8), self.labels.iloc[::-1], 0.25, 0.5)
        self.assertEqual(first, build_report(self.registry, self.evaluation, shuffled, [2]))
        rows = validation_rows(first)
        self.assertEqual(len(rows), len({r["outreach_capacity"] for r in rows}))

    def test_criteria_stay_unresolved_and_never_approve(self):
        ranked = rank_predictions(self.predictions, self.labels, 0.25, 0.5)
        result = build_report(self.registry, self.evaluation, ranked)
        for key in ("pr_auc_min", "precision_at_k_min", "outreach_capacity", "refresh_cadence"):
            self.assertIsNone(result["criteria"][key])
        self.assertIsNone(result["diagnostic_criteria_checks"]["pr_auc"])
        configured = {**BUSINESS_ACCEPTANCE, "pr_auc_min": 0.1, "precision_at_k_min": 0.1,
                      "outreach_capacity": 2, "refresh_cadence": "candidate cadence"}
        result = build_report(self.registry, self.evaluation, ranked, criteria=configured)
        self.assertTrue(all(result["diagnostic_criteria_checks"].values()))
        self.assertEqual((result["business_status"], result["stakeholder_status"], result["production_approved"]), ("pending", "pending", False))
        for change in ({"stakeholder_approved": True}, {"outreach_capacity": 0},
                       {"pr_auc_min": float("nan")}, {"precision_at_k_min": True},
                       {"refresh_cadence": ""}, {"outreach_capacity": 6}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                analyze(ranked, 0.25, 0.5, criteria={**BUSINESS_ACCEPTANCE, **change})

    def test_missing_model_coverage_invalid_scores_and_tiers_fail(self):
        connection = MagicMock()
        connection.execute.return_value.mappings.return_value.first.return_value = None
        with self.assertRaisesRegex(ValueError, "Model not found"):
            load_inputs(connection, "missing", FEATURE_SET_VERSION)
        for labels in (self.labels.iloc[:-1], self.labels.assign(customer_id=["z", "b", "c", "d", "e"]),
                       pd.concat([self.labels, self.labels.iloc[:1]])):
            with self.subTest(labels=len(labels)), self.assertRaisesRegex(ValueError, "coverage"):
                rank_predictions(self.predictions, labels, 0.25, 0.5)
        for predictions in (self.predictions.assign(risk_tier="low"),
                            self.predictions.assign(churn_probability=float("nan")), self.predictions.iloc[:0]):
            with self.assertRaises(ValueError):
                rank_predictions(predictions, self.labels, 0.25, 0.5)
        ranked = rank_predictions(self.predictions, self.labels, 0.25, 0.5)
        for n in (0, -1, 6, True):
            with self.subTest(n=n), self.assertRaises(ValueError):
                analyze(ranked, 0.25, 0.5, [n])

    def test_source_hashes_dates_and_batch_consistency(self):
        source = load_and_validate().iloc[:100].sort_values("customer_id").reset_index(drop=True)
        features = source[["customer_id", *FEATURE_COLUMNS]].copy()
        features[NUMERIC] = features[NUMERIC].astype(float)
        labels = source[["customer_id", "churned"]].assign(
            label_source="source_churn_column", observation_date=None, label_window_end=None)
        labels["churned"] = labels.churned.astype(int)
        feature_hash = hashlib.sha256(features.to_csv(index=False).encode()).hexdigest()
        dataset = features.assign(churned=labels.churned)
        dataset_hash = hashlib.sha256(dataset.to_csv(index=False).encode()).hexdigest()
        predictions = pd.DataFrame({"customer_id": features.customer_id,
                                    "churn_probability": 0.5, "risk_tier": "high",
                                    "scored_at": pd.Timestamp("2026-09-25"),
                                    "feature_set_version": FEATURE_SET_VERSION,
                                    "feature_snapshot_sha256": feature_hash})
        evaluation = {**self.evaluation, "label_version": FEATURE_SET_VERSION,
                      "dataset_sha256": dataset_hash, "threshold": 0.5,
                      "test": {metric: 0.5 for metric in METRICS.values()}}
        registry = {**self.registry, "dataset_sha256": dataset_hash, "approved": 0,
                    "evaluation_scope": "benchmark", "artifact_path": "unused",
                    "trained_at": pd.Timestamp("2026-09-25").to_pydatetime(),
                    **{column: 0.5 for column in METRICS}}
        connection = MagicMock()
        connection.execute.return_value.mappings.return_value.first.return_value = registry
        with patch("src.business_validation.load_artifacts", return_value=(evaluation, None, None, None)), \
                patch("src.business_validation.pd.read_sql_query") as read:
            read.side_effect = [predictions, labels, features]
            self.assertEqual(len(load_inputs(connection, "example", FEATURE_SET_VERSION)[2]), 100)
            cases = [
                (predictions.assign(feature_snapshot_sha256="f" * 64), labels, features, "stale"),
                (predictions, labels.assign(churned=1 - labels.churned), features, "dataset"),
                (predictions, labels.assign(observation_date=pd.Timestamp("2026-01-01")), features, "undated"),
                (predictions.assign(scored_at=[pd.Timestamp("2026-09-24")] + [pd.Timestamp("2026-09-25")] * 99), labels, features, "batch"),
                (predictions, labels, features.iloc[:-1], "coverage"),
            ]
            for preds, outcomes, frame, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    read.side_effect = [preds, outcomes, frame]
                    load_inputs(connection, "example", FEATURE_SET_VERSION)


@unittest.skipUnless(os.getenv("RUN_BUSINESS_MYSQL_TESTS") == "1", "Set RUN_BUSINESS_MYSQL_TESTS=1 for Phase 6 MySQL checks")
class BusinessValidationMySQLTest(unittest.TestCase):
    def test_persisted_report_rerun_and_no_model_mutation(self):
        model_id = os.getenv("TEST_MODEL_ID", "xgb_9ee0132da5749c7e72124a8c")
        engine = get_engine()
        params = {"model_id": model_id}
        source = text("SELECT customer_id, churn_probability, risk_tier, scored_at, feature_snapshot_sha256 FROM model_predictions WHERE model_id=:model_id ORDER BY customer_id")
        try:
            with engine.connect() as connection:
                before = connection.execute(source, params).all()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.md"
                first = run_validation(model_id, report_path=path, save=True)
                artifacts = [path, path.with_suffix(".json"), path.with_suffix(".ranked.csv")]
                contents = [p.read_bytes() for p in artifacts]
                self.assertEqual(json.loads(contents[1]), first)
                self.assertEqual(hashlib.sha256(contents[2]).hexdigest(), first["ranking_sha256"])
                self.assertEqual(run_validation(model_id, report_path=path, save=True), first)
                self.assertEqual([p.read_bytes() for p in artifacts], contents)
            expected_rows = validation_rows(first)
            with engine.connect() as connection:
                self.assertEqual(connection.execute(source, params).all(), before)
                self.assertEqual(connection.execute(text("SELECT approved FROM model_registry WHERE model_id=:model_id"), params).scalar_one(), 0)
                self.assertEqual(connection.execute(text("SELECT COUNT(*) FROM business_validation WHERE analysis_sha256=:digest"), {"digest": first["analysis_sha256"]}).scalar_one(), len(expected_rows))
                for row in expected_rows:
                    stored = connection.execute(text(f"SELECT {', '.join(row)} FROM business_validation WHERE validation_id=:validation_id"), row).mappings().one()
                    self.assertEqual(dict(stored), row)
            with self.assertRaisesRegex(DBAPIError, "3819|Check constraint"):
                with engine.begin() as connection:
                    connection.execute(text("UPDATE business_validation SET stakeholder_status='approved' WHERE validation_id=:validation_id"), expected_rows[0])
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
