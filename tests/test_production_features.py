"""Synthetic dated sources exercise production reads without any churn labels."""

from datetime import date
import unittest

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

from src.config import FEATURE_SET_VERSION
from src.features import FEATURE_COLUMNS, load_active_customers, refresh_production_features
from src.ingest import load_and_validate


class ProductionFeatureTest(unittest.TestCase):
    observation_date = date(2026, 9, 25)
    feature_version = "dated_contract_test_v1"

    @classmethod
    def setUpClass(cls):
        cls.features = load_and_validate().loc[:2, ["customer_id", *FEATURE_COLUMNS]].copy()
        cls.features["customer_id"] = ["a", "b", "c"]

    def setUp(self):
        self.engine = create_engine("sqlite://")
        self.addCleanup(self.engine.dispose)
        self.customers = pd.DataFrame({
            "customer_id": ["a", "b", "c", "d", "e"],
            "signup_date": ["2026-01-01"] * 5,
        })
        self.subscriptions = pd.DataFrame([
            ("b", "active", "2026-09-25", None),
            ("a", "active", "2026-02-01", "2026-09-26"),
            ("a", "active", "2026-03-01", None),
            ("c", "active", "2026-02-01", "2026-09-25"),
            ("d", "active", "2026-09-26", None),
            ("e", "cancelled", "2026-02-01", "2026-09-24"),
        ], columns=["customer_id", "status", "start_date", "end_date"])

    def write_sources(self, customers=None, subscriptions=None):
        (self.customers if customers is None else customers).to_sql(
            "customers", self.engine, index=False, if_exists="replace")
        (self.subscriptions if subscriptions is None else subscriptions).to_sql(
            "subscriptions", self.engine, index=False, if_exists="replace")

    def write_features(self, frame):
        frame.to_sql("feature_store", self.engine, index=False, if_exists="replace")

    def dated_features(self):
        return self.features.assign(feature_set_version=self.feature_version,
                                    observation_date=self.observation_date)

    def test_active_cohort_is_distinct_and_uses_subscription_date_boundaries(self):
        self.write_sources()
        with self.engine.connect() as connection:
            self.assertEqual(load_active_customers(connection, self.observation_date), ["a", "b"])
        self.assertNotIn("churn_labels", inspect(self.engine).get_table_names())
        self.write_sources(subscriptions=self.subscriptions.assign(status="paused"))
        with self.engine.connect() as connection, self.assertRaisesRegex(ValueError, "nonempty"):
            load_active_customers(connection, self.observation_date)

    def test_invalid_customer_sources_cannot_hide_outside_active_cohort(self):
        for field, value in (("customer_id", "a"), ("customer_id", " "),
                             ("customer_id", "x" * 21), ("customer_id", None),
                             ("signup_date", None), ("signup_date", "not-a-date")):
            customers = self.customers.copy()
            customers.loc[4, field] = value
            self.write_sources(customers=customers)
            with self.subTest(field=field, value=value), self.engine.connect() as connection:
                with self.assertRaises(ValueError):
                    load_active_customers(connection, self.observation_date)

    def test_invalid_inactive_subscriptions_are_rejected_before_filtering(self):
        for field, value in (("customer_id", "orphan"), ("status", "unknown"),
                             ("start_date", None), ("start_date", "2025-12-31"),
                             ("end_date", "2026-01-01"), ("end_date", "not-a-date")):
            subscriptions = self.subscriptions.copy()
            subscriptions.loc[5, field] = value
            self.write_sources(subscriptions=subscriptions)
            with self.subTest(field=field, value=value), self.engine.connect() as connection:
                with self.assertRaises(ValueError):
                    load_active_customers(connection, self.observation_date)

    def test_feature_refresh_uses_exact_date_version_and_active_ids_without_labels(self):
        dated = self.dated_features()
        older = dated.assign(observation_date=date(2026, 9, 24), contract="invalid")
        other_version = dated.assign(feature_set_version="another_version", contract="invalid")
        self.write_features(pd.concat([dated.iloc[::-1], older, other_version], ignore_index=True))
        with self.engine.connect() as connection:
            actual = refresh_production_features(connection, self.feature_version,
                                                 self.observation_date, ["b", "a"])
            self.assertEqual(actual.customer_id.tolist(), ["a", "b"])
            self.assertEqual(actual.contract.tolist(), self.features.contract.iloc[:2].tolist())
            self.assertEqual(list(actual.columns), ["customer_id", *FEATURE_COLUMNS])
            with self.assertRaisesRegex(ValueError, "nonempty"):
                refresh_production_features(connection, self.feature_version, date(2026, 9, 23), ["a"])
            with self.assertRaisesRegex(ValueError, "benchmark"):
                refresh_production_features(connection, FEATURE_SET_VERSION, self.observation_date, ["a"])
        self.assertEqual(inspect(self.engine).get_table_names(), ["feature_store"])

    def test_feature_refresh_rejects_incomplete_duplicate_and_invalid_snapshots(self):
        dated = self.dated_features()
        invalid_frames = {
            "missing active customer": dated.iloc[1:],
            "duplicate customer": pd.concat([dated, dated.iloc[:1]], ignore_index=True),
            "invalid category": dated.assign(contract="unknown"),
            "nonfinite charges": dated.assign(monthly_charges=np.inf),
            "fractional tenure": dated.assign(tenure_months=1.5),
            "missing core values": dated.assign(partner=None),
        }
        for reason, frame in invalid_frames.items():
            self.write_features(frame)
            with self.subTest(reason=reason), self.engine.connect() as connection:
                with self.assertRaises(ValueError):
                    refresh_production_features(connection, self.feature_version,
                                                self.observation_date, ["a", "b"])

    def test_feature_null_rate_gate_applies_to_selected_active_cohort(self):
        frame = pd.concat([self.dated_features().iloc[:1]] * 20, ignore_index=True)
        frame["customer_id"] = [f"customer-{index}" for index in range(20)]
        frame.loc[0, "total_charges"] = np.nan
        self.write_features(frame)
        with self.engine.connect() as connection:
            complete = refresh_production_features(connection, self.feature_version,
                                                    self.observation_date, frame.customer_id.tolist())
            self.assertEqual(complete.total_charges.isna().sum(), 1)
            with self.assertRaisesRegex(ValueError, "null rate"):
                refresh_production_features(connection, self.feature_version,
                                            self.observation_date, ["customer-0"])

    def test_invalid_source_values_are_not_exposed_in_errors(self):
        private_value = "private-customer-input@example.test"
        self.write_features(self.dated_features().assign(monthly_charges=private_value))
        with self.engine.connect() as connection, self.assertRaises(ValueError) as caught:
            refresh_production_features(connection, self.feature_version,
                                        self.observation_date, ["a", "b"])
        self.assertNotIn(private_value, str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        for field in ("signup_date", "start_date", "end_date"):
            customers, subscriptions = self.customers.copy(), self.subscriptions.copy()
            if field == "signup_date":
                customers.loc[4, field] = private_value
            else:
                subscriptions.loc[5, field] = private_value
            self.write_sources(customers, subscriptions)
            with self.subTest(field=field), self.engine.connect() as connection:
                with self.assertRaises(ValueError) as caught:
                    load_active_customers(connection, self.observation_date)
                self.assertNotIn(private_value, str(caught.exception))
                self.assertTrue(caught.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
