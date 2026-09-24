import unittest

from src.features import FEATURE_COLUMNS, _validate_counts


class FeatureValidationTest(unittest.TestCase):
    def test_features_exclude_label_and_enforce_quality_gates(self):
        self.assertNotIn("churned", FEATURE_COLUMNS)
        _validate_counts(7_043, 7_043, 11)

        with self.assertRaisesRegex(ValueError, "one feature row per customer"):
            _validate_counts(7_043, 7_042, 11)
        with self.assertRaisesRegex(ValueError, "5% null-value limit"):
            _validate_counts(100, 100, 6)


if __name__ == "__main__":
    unittest.main()
