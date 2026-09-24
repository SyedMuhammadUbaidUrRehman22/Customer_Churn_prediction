import unittest

from src.labels import _validate_counts


class LabelValidationTest(unittest.TestCase):
    def test_label_counts_require_full_coverage_and_both_classes(self):
        _validate_counts(7_043, 7_043, 1_869)

        with self.assertRaisesRegex(ValueError, "one label per customer"):
            _validate_counts(7_043, 7_042, 1_869)
        with self.assertRaisesRegex(ValueError, "both churned and retained"):
            _validate_counts(7_043, 7_043, 0)


if __name__ == "__main__":
    unittest.main()
