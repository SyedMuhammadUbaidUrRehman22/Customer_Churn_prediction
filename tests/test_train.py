import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS
from src.ingest import load_and_validate
from src.train import NUMERIC, split_data, train_benchmark, validate_training_frame
from src.utils import get_engine


class TrainingTest(unittest.TestCase):
    def test_split_validation_and_saved_model(self):
        frame = load_and_validate()
        validate_training_frame(frame)
        parts = split_data(frame)
        repeated = split_data(frame.sample(frac=1, random_state=7))
        for part, again in zip(parts, repeated):
            self.assertEqual(part.customer_id.tolist(), again.customer_id.tolist())
        sets = [set(part.customer_id) for part in parts]
        self.assertEqual(len(set.union(*sets)), len(frame))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for column, value in (("tenure_months", 1.5), ("monthly_charges", float("inf")),
                              ("contract", "unknown"), ("churned", 2), ("total_charges", 1e10)):
            invalid = frame.copy()
            invalid.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                validate_training_frame(invalid)
        with self.assertRaises(ValueError):
            validate_training_frame(frame.iloc[:0])
        with self.assertRaises(ValueError):
            validate_training_frame(frame.assign(customer_id="duplicate"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            report = train_benchmark(frame, output, trials=1)
            preprocessor = joblib.load(output / "preprocessor.joblib")
            np.testing.assert_allclose(preprocessor.named_transformers_["numeric"].statistics_,
                                       parts[0][NUMERIC].astype(float).median().values)
            model = XGBClassifier()
            model.load_model(output / "model.json")
            probabilities = model.predict_proba(preprocessor.transform(parts[2][list(FEATURE_COLUMNS)]))[:, 1]
            from src.train import evaluate
            self.assertEqual(evaluate(parts[2].churned.astype(int), probabilities, report["threshold"]), report["test"])
            self.assertFalse(report["approved"])
            self.assertEqual(len(report["feature_importance"]), 19)
            self.assertTrue((output / "splits.csv").exists())
            with self.assertRaises(FileExistsError):
                train_benchmark(frame, output, trials=1)

    def test_driver_timeout(self):
        for driver, keyword in (("pymysql", "connect_timeout"), ("mysqlconnector", "connection_timeout")):
            with patch("src.utils.DATABASE_URI", f"mysql+{driver}://localhost/test"), patch("src.utils.create_engine") as create:
                get_engine()
                self.assertEqual(create.call_args.kwargs["connect_args"], {keyword: 30})


if __name__ == "__main__":
    unittest.main()
