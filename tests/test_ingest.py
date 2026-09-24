import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH
from src.ingest import load_and_validate


class IngestValidationTest(unittest.TestCase):
    def test_real_csv_and_duplicate_guard(self):
        frame = load_and_validate()
        self.assertEqual(len(frame), 7_043)
        self.assertEqual(frame["customer_id"].nunique(), 7_043)

        source = pd.read_csv(RAW_DATA_PATH)
        source.loc[1, "customerID"] = source.loc[0, "customerID"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "customerID must be unique"):
                load_and_validate(path)


if __name__ == "__main__":
    unittest.main()
