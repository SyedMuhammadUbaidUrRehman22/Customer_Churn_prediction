"""Train a reproducible, non-temporal Telco benchmark from MySQL."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import text
import xgboost
from xgboost import XGBClassifier

from src.config import (FEATURE_SET_VERSION, LABEL_VERSION, PROJECT_ROOT,
                        TABLE_FEATURES, TABLE_LABELS, TABLE_RAW)
from src.features import CATEGORICAL, FEATURE_COLUMNS, NUMERIC, validate_feature_frame
from src.utils import get_engine

SEED = 42


def validate_training_frame(frame):
    validate_feature_frame(frame)
    if "churned" not in frame:
        raise ValueError("Training data requires churned labels")
    if not frame.churned.isin([0, 1]).all() or frame.churned.value_counts().reindex([0, 1], fill_value=0).min() < 10:
        raise ValueError("Training requires binary labels and at least 10 customers in each class")


def load_training_frame():
    engine = get_engine()
    params = {"features": FEATURE_SET_VERSION, "labels": LABEL_VERSION}
    columns = ", ".join(f"f.{column}" for column in FEATURE_COLUMNS)
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                frame = pd.read_sql_query(text(f"""
                    SELECT f.customer_id, {columns}, l.churned,
                           l.observation_date, l.label_window_end
                    FROM {TABLE_FEATURES} f
                    JOIN {TABLE_LABELS} l ON f.customer_id = l.customer_id
                    WHERE f.feature_set_version = :features AND l.label_version = :labels
                    ORDER BY f.customer_id
                """), connection, params=params)
                counts = [connection.execute(text(query), params).scalar_one() for query in (
                    f"SELECT COUNT(*) FROM {TABLE_RAW}",
                    f"SELECT COUNT(*) FROM {TABLE_FEATURES} WHERE feature_set_version = :features",
                    f"SELECT COUNT(*) FROM {TABLE_LABELS} WHERE label_version = :labels",
                )]
                if any(count != len(frame) for count in counts):
                    raise ValueError("Training requires exact raw/feature/label customer coverage")
    finally:
        engine.dispose()
    if frame[["observation_date", "label_window_end"]].notna().any().any():
        raise ValueError("Snapshot trainer cannot evaluate dated data; implement temporal splits first")
    validate_training_frame(frame)
    return frame


def split_data(frame):
    frame = frame.sort_values("customer_id").reset_index(drop=True)
    train, remaining = train_test_split(frame, test_size=0.4, stratify=frame.churned, random_state=SEED)
    validation, test = train_test_split(remaining, test_size=0.5, stratify=remaining.churned, random_state=SEED)
    return train, validation, test


def evaluate(labels, probabilities, threshold):
    labels = np.asarray(labels)
    predictions = probabilities >= threshold
    top = np.argsort(-probabilities, kind="stable")[:int(np.ceil(len(labels) * 0.1))]
    return {
        "pr_auc_average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
        "recall_at_top_decile": float(labels[top].sum() / labels.sum()),
        "precision_at_top_decile": float(labels[top].mean()),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def train_benchmark(frame, output_dir, trials=8):
    validate_training_frame(frame)
    frame = frame.copy()
    frame["churned"] = frame.churned.astype(int)
    frame[NUMERIC] = frame[NUMERIC].astype(float)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not 1 <= trials <= 100:
        raise ValueError("trials must be between 1 and 100")
    train, validation, test = split_data(frame)
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    x_train = preprocess.fit_transform(train[list(FEATURE_COLUMNS)])
    x_validation = preprocess.transform(validation[list(FEATURE_COLUMNS)])
    weight = float(train.churned.eq(0).sum() / train.churned.eq(1).sum())
    search_space = dict(n_estimators=[200, 400, 800], max_depth=[3, 4, 5, 6],
                        learning_rate=[0.01, 0.05, 0.1], subsample=[0.7, 0.8, 1.0],
                        colsample_bytree=[0.7, 0.8, 1.0], min_child_weight=[1, 3, 5])
    candidates, best_model, best_score = [], None, -1
    for parameters in ParameterSampler(search_space, n_iter=trials, random_state=SEED):
        model = XGBClassifier(**parameters, objective="binary:logistic", eval_metric="aucpr",
                              tree_method="hist", scale_pos_weight=weight,
                              early_stopping_rounds=30, random_state=SEED, n_jobs=1)
        model.fit(x_train, train.churned, eval_set=[(x_validation, validation.churned)], verbose=False)
        score = float(average_precision_score(validation.churned, model.predict_proba(x_validation)[:, 1]))
        candidates.append({"parameters": parameters, "validation_average_precision": score,
                           "best_iteration": model.best_iteration})
        if score > best_score:
            best_model, best_score = model, score
    probabilities = best_model.predict_proba(x_validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(validation.churned, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f1)])
    test_probabilities = best_model.predict_proba(preprocess.transform(test[list(FEATURE_COLUMNS)]))[:, 1]
    importance = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    # Aggregate one-hot gains back to the original predictors for the 50% gate.
    offset = len(NUMERIC)
    for column, value in zip(NUMERIC, best_model.feature_importances_[:offset]):
        importance[column] = float(value)
    for column, categories in zip(CATEGORICAL, preprocess.named_transformers_["categorical"].categories_):
        importance[column] = float(best_model.feature_importances_[offset:offset + len(categories)].sum())
        offset += len(categories)
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_design": "non-temporal stratified snapshot benchmark, 60/20/20; seed 42",
        "approved": False,
        "promotion_blockers": ["No dated events or temporal holdout", "No stakeholder acceptance thresholds"],
        "feature_set_version": FEATURE_SET_VERSION, "label_version": LABEL_VERSION,
        "features": list(FEATURE_COLUMNS), "scale_pos_weight": weight,
        "dataset_sha256": hashlib.sha256(frame.sort_values("customer_id")[["customer_id", *FEATURE_COLUMNS, "churned"]].to_csv(index=False).encode()).hexdigest(),
        "split_counts": {name: {"rows": len(part), "churned": int(part.churned.sum())}
                         for name, part in (("train", train), ("validation", validation), ("test", test))},
        "threshold": threshold, "threshold_policy": "maximum validation F1; not a business cost threshold",
        "candidates": candidates,
        "selected_parameters": {key: (None if isinstance(value, float) and np.isnan(value) else value)
                                for key, value in best_model.get_params().items()},
        "validation": evaluate(validation.churned, probabilities, threshold),
        "test": evaluate(test.churned, test_probabilities, threshold),
        "test_prevalence_baseline_ap": float(test.churned.mean()),
        "test_constant_train_prevalence_brier": float(brier_score_loss(test.churned, np.full(len(test), train.churned.mean()))),
        "feature_importance": importance,
        "importance_gate_passed": max(importance.values()) <= 0.5,
        "versions": {"xgboost": xgboost.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    if not report["importance_gate_passed"]:
        report["promotion_blockers"].append("A predictor exceeds 50% of aggregate importance")
    output_dir.mkdir(parents=True, exist_ok=False)
    best_model.save_model(output_dir / "model.json")
    joblib.dump(preprocess, output_dir / "preprocessor.joblib")
    report["artifact_sha256"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                                 for name in ("model.json", "preprocessor.joblib")}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat([part[["customer_id"]].assign(split=name) for name, part in
               (("train", train), ("validation", validation), ("test", test))]).to_csv(output_dir / "splits.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / datetime.now(timezone.utc).strftime("benchmark_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--register", action="store_true", help="Register the completed benchmark as unapproved in MySQL")
    args = parser.parse_args()
    report = train_benchmark(load_training_frame(), args.output_dir, args.trials)
    result = {"artifacts": str(args.output_dir), "test": report["test"], "approved": False}
    if args.register:
        from src.registry import register_model
        result["model_id"] = register_model(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
