"""Review persisted snapshot benchmarks; no stakeholder or production approval."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.config import (BUSINESS_ACCEPTANCE, BUSINESS_SCHEMA_PATH, FEATURE_SET_VERSION,
                        PROJECT_ROOT, TABLE_BUSINESS_VALIDATION, TABLE_FEATURES,
                        TABLE_LABELS, TABLE_PREDICTIONS, TABLE_REGISTRY)
from src.features import FEATURE_COLUMNS, NUMERIC, validate_feature_frame
from src.registry import METRICS, load_artifacts
from src.score import risk_tiers
from src.utils import get_engine

LIMITATIONS = [
    "Labels come from the supplied Telco snapshot Churn field.",
    "There are no observation dates and no genuine future-window churn labels.",
    "The saved test evaluation is not a temporal holdout.",
    "Outreach and tier statistics cover the full snapshot, including training customers; they are descriptive, not new held-out performance.",
    "Benchmark predictions are not evidence of production-calibrated future churn probability or a verified active-customer cohort.",
    "Stakeholder approval has not been fabricated. Business acceptance and stakeholder approval remain pending; production approval is false.",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_criteria(criteria):
    if set(criteria) != set(BUSINESS_ACCEPTANCE):
        raise ValueError("Acceptance configuration has missing or unknown keys")
    if criteria["stakeholder_approved"] is not False:
        raise ValueError("Configuration cannot grant stakeholder approval")
    for key in ("pr_auc_min", "precision_at_k_min", "feature_importance_max"):
        value = criteria[key]
        if value is None and key != "feature_importance_max":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid acceptance criterion: {key}")
        if key == "feature_importance_max" and value == 0:
            raise ValueError("Feature importance guardrail must be positive")
    capacity = criteria["outreach_capacity"]
    if capacity is not None and (type(capacity) is not int or capacity < 1):
        raise ValueError("Outreach capacity must be an integer greater than zero or None")
    cadence = criteria["refresh_cadence"]
    if cadence is not None and (not isinstance(cadence, str) or not cadence.strip() or len(cadence) > 100):
        raise ValueError("Refresh cadence must be nonblank text up to 100 characters or None")


def rank_predictions(predictions, labels, medium, high):
    for frame in (predictions, labels):
        if frame.empty or frame.customer_id.isna().any() or not frame.customer_id.is_unique:
            raise ValueError("Prediction/label coverage requires nonempty unique customer IDs")
    if set(predictions.customer_id) != set(labels.customer_id):
        raise ValueError("Prediction/label customer coverage mismatch")
    if not labels.churned.isin([0, 1]).all():
        raise ValueError("Invalid snapshot labels")
    tiers = risk_tiers(predictions.churn_probability, medium, high)
    if not (predictions.risk_tier.to_numpy() == tiers).all():
        raise ValueError("Stored risk tiers differ from Phase 5 thresholds")
    ranked = predictions.merge(labels[["customer_id", "churned"]], on="customer_id", validate="one_to_one")
    ranked = ranked.sort_values(["churn_probability", "customer_id"], ascending=[False, True]).reset_index(drop=True)
    ranked["churned"] = ranked.churned.astype(int)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def analyze(ranked, medium, high, top_k=(), criteria=None):
    criteria = dict(BUSINESS_ACCEPTANCE if criteria is None else criteria)
    validate_criteria(criteria)
    total, churners = len(ranked), int(ranked.churned.sum())
    capacities = [(f"Top {percent}%", math.ceil(total * percent / 100)) for percent in (1, 5, 10, 20)]
    capacities += [(f"Custom N={n}", n) for n in top_k]
    if criteria["outreach_capacity"] is not None:
        capacities.append(("Configured outreach capacity", criteria["outreach_capacity"]))
    outreach = []
    for name, n in capacities:
        if type(n) is not int or not 1 <= n <= total:
            raise ValueError(f"Outreach N must be between 1 and {total}")
        selected = ranked.iloc[:n]
        captured = int(selected.churned.sum())
        outreach.append({"capacity": name, "selected": n, "churners": captured,
                         "precision": captured / n, "recall": captured / churners if churners else None,
                         "selected_prevalence": captured / n,
                         "cumulative_churn_capture": captured / churners if churners else None,
                         "threshold_used": float(selected.churn_probability.iloc[-1])})
    tiers = []
    for tier, lower, upper in (("high", high, 1.0), ("medium", medium, high), ("low", 0.0, medium)):
        rows = ranked[ranked.risk_tier == tier]
        count, captured = len(rows), int(rows.churned.sum())
        tiers.append({"tier": tier, "lower_bound": lower, "upper_bound": upper,
                      "customers": count, "population_fraction": count / total,
                      "churners": captured, "churn_rate": captured / count if count else None})
    return {"population": total, "churners": churners, "prevalence": churners / total,
            "outreach": outreach, "tiers": tiers, "criteria": criteria}


def load_inputs(connection, model_id, feature_version):
    columns = ", ".join(METRICS)
    registry = connection.execute(text(f"""
        SELECT model_id, trained_at, feature_set_version, label_version, evaluation_scope,
               approved, artifact_path, report_sha256, dataset_sha256, low_threshold,
               high_threshold, {columns}
        FROM {TABLE_REGISTRY} WHERE model_id = :model_id
    """), {"model_id": model_id}).mappings().first()
    if registry is None:
        raise ValueError(f"Model not found: {model_id}")
    if registry["evaluation_scope"] != "benchmark" or registry["approved"]:
        raise ValueError("Phase 6 requires an unapproved benchmark model")
    if feature_version != registry["feature_set_version"]:
        raise ValueError("Requested feature-set version differs from registry")
    report, _, _, _ = load_artifacts(registry["artifact_path"], registry["report_sha256"])
    if (registry["label_version"] != report["label_version"]
            or registry["dataset_sha256"] != report["dataset_sha256"]
            or registry["trained_at"] != datetime.fromisoformat(report["trained_at"]).astimezone(timezone.utc).replace(tzinfo=None)
            or any(registry[column] != report["test"][metric] for column, metric in METRICS.items())
            or registry["high_threshold"] != report["threshold"]
            or registry["low_threshold"] != report["threshold"] / 2):
        raise ValueError("Registry metadata differs from the pinned evaluation report")
    params = {"model_id": model_id, "features": feature_version, "labels": registry["label_version"]}
    predictions = pd.read_sql_query(text(f"""
        SELECT customer_id, churn_probability, risk_tier, scored_at, feature_set_version,
               feature_snapshot_sha256 FROM {TABLE_PREDICTIONS}
        WHERE model_id = :model_id AND scoring_mode = 'benchmark'
    """), connection, params=params)
    labels = pd.read_sql_query(text(f"""
        SELECT customer_id, churned, label_source, observation_date, label_window_end
        FROM {TABLE_LABELS} WHERE label_version = :labels
    """), connection, params=params)
    ranked = rank_predictions(predictions, labels, registry["low_threshold"], registry["high_threshold"])
    if (labels[["observation_date", "label_window_end"]].notna().any().any()
            or not labels.label_source.eq("source_churn_column").all()):
        raise ValueError("Phase 6 requires the documented undated snapshot labels")
    for column in ("scored_at", "feature_snapshot_sha256", "feature_set_version"):
        if predictions[column].isna().any() or predictions[column].nunique() != 1:
            raise ValueError(f"Mixed or incomplete prediction batch: {column}")
    if not predictions.feature_set_version.eq(feature_version).all():
        raise ValueError("Prediction feature version mismatch")
    features = pd.read_sql_query(text(f"SELECT customer_id, {', '.join(FEATURE_COLUMNS)} "
                                     f"FROM {TABLE_FEATURES} WHERE feature_set_version = :features "
                                     "ORDER BY customer_id"), connection, params=params)
    validate_feature_frame(features)
    if set(features.customer_id) != set(labels.customer_id):
        raise ValueError("Feature/label customer coverage mismatch")
    features[NUMERIC] = features[NUMERIC].astype(float)
    feature_hash = digest(features[["customer_id", *FEATURE_COLUMNS]].to_csv(index=False).encode())
    if feature_hash != predictions.feature_snapshot_sha256.iloc[0]:
        raise ValueError("Predictions are stale: feature snapshot hash mismatch; rescore first")
    dataset = features.merge(labels[["customer_id", "churned"]], on="customer_id", validate="one_to_one")
    dataset["churned"] = dataset.churned.astype(int)
    dataset_hash = digest(dataset.sort_values("customer_id")[["customer_id", *FEATURE_COLUMNS, "churned"]].to_csv(index=False).encode())
    if dataset_hash != report["dataset_sha256"]:
        raise ValueError("Feature/label snapshot differs from the evaluated benchmark dataset")
    return registry, report, ranked


def build_report(registry, evaluation, ranked, top_k=(), criteria=None):
    medium, high = float(registry["low_threshold"]), float(registry["high_threshold"])
    result = analyze(ranked, medium, high, top_k, criteria)
    result.update({
        "analysis_version": "snapshot_business_v1",
        "model_id": registry["model_id"], "feature_set_version": registry["feature_set_version"],
        "label_version": registry["label_version"], "trained_at": evaluation["trained_at"],
        "evaluation_scope": "benchmark", "production_approved": False,
        "business_status": "pending", "stakeholder_status": "pending", "tier_status": "provisional",
        "scored_at": ranked.scored_at.iloc[0].isoformat(),
        "feature_snapshot_sha256": ranked.feature_snapshot_sha256.iloc[0],
        "evaluation_report_sha256": registry["report_sha256"],
        "dataset_sha256": registry["dataset_sha256"],
        "technical_test_metrics": evaluation["test"],
        "test_prevalence": evaluation["test_prevalence_baseline_ap"],
        "split_counts": evaluation["split_counts"],
        "maximum_feature_importance": max(evaluation["feature_importance"].values()),
        "ranking_sha256": digest(ranking_csv(ranked).encode()),
        "limitations": LIMITATIONS,
    })
    configured = result["criteria"]
    precision = next((row["precision"] for row in result["outreach"]
                      if row["selected"] == configured["outreach_capacity"]), None)
    result["diagnostic_criteria_checks"] = {
        "pr_auc": None if configured["pr_auc_min"] is None else evaluation["test"]["pr_auc_average_precision"] >= configured["pr_auc_min"],
        "full_snapshot_precision_at_k": None if configured["precision_at_k_min"] is None or precision is None else precision >= configured["precision_at_k_min"],
        "feature_importance": result["maximum_feature_importance"] <= configured["feature_importance_max"],
    }
    result["analysis_sha256"] = digest(json.dumps(result, sort_keys=True, allow_nan=False).encode())
    return result


def ranking_csv(ranked):
    return ranked[["rank", "customer_id", "churn_probability", "risk_tier", "churned"]].to_csv(index=False)


def validation_rows(report):
    tiers = {row["tier"]: row for row in report["tiers"]}
    criteria = report["criteria"]
    rows = {}
    for capacity in report["outreach"]:
        n = capacity["selected"]
        rows[n] = {
            "validation_id": digest(f"{report['analysis_sha256']}:{n}".encode()),
            "analysis_sha256": report["analysis_sha256"], "model_id": report["model_id"],
            "feature_set_version": report["feature_set_version"], "label_version": report["label_version"],
            "evaluation_scope": "benchmark", "scored_at": datetime.fromisoformat(report["scored_at"]),
            "feature_snapshot_sha256": report["feature_snapshot_sha256"],
            "outreach_capacity": n, "threshold_used": capacity["threshold_used"],
            "selected_count": n, "selected_churners": capacity["churners"],
            "precision_at_n": capacity["precision"], "recall_at_n": capacity["recall"],
            "population_count": report["population"], "population_churners": report["churners"],
            **{f"{tier}_count": row["customers"] for tier, row in tiers.items()},
            **{f"{tier}_threshold": row["lower_bound"] for tier, row in tiers.items()},
            "pr_auc_min": criteria["pr_auc_min"], "precision_at_k_min": criteria["precision_at_k_min"],
            "stakeholder_outreach_capacity": criteria["outreach_capacity"],
            "refresh_cadence": criteria["refresh_cadence"], "feature_importance_max": criteria["feature_importance_max"],
            "business_status": "pending", "stakeholder_status": "pending",
            "notes": "Full snapshot includes training customers. Capacity is a scenario; tiers are provisional. No stakeholder sign-off or production approval.",
        }
    return list(rows.values())


def save_validation(engine, report):
    rows = validation_rows(report)
    with engine.begin() as connection:
        connection.exec_driver_sql(BUSINESS_SCHEMA_PATH.read_text(encoding="utf-8").rstrip(";\n"))
    with engine.begin() as connection:
        for row in rows:
            columns = ", ".join(row)
            values = ", ".join(f":{column}" for column in row)
            connection.execute(text(f"INSERT INTO {TABLE_BUSINESS_VALIDATION} ({columns}, validation_date) "
                                    f"VALUES ({values}, :validation_date) "
                                    "ON DUPLICATE KEY UPDATE validation_id = validation_id"),
                               {**row, "validation_date": datetime.now(timezone.utc).date()})
            stored = connection.execute(text(f"SELECT {columns} FROM {TABLE_BUSINESS_VALIDATION} "
                                             "WHERE validation_id = :validation_id FOR UPDATE"), row).mappings().one()
            if any(stored[column] != value for column, value in row.items()):
                raise ValueError("Validation conflicts with persisted record")
    return [row["validation_id"] for row in rows]


def render_report(report):
    def value(item):
        return "undefined" if item is None else f"{item:.6f}"

    lines = ["# Phase 6 business validation — snapshot benchmark", "",
             f"Model: `{report['model_id']}`; features: `{report['feature_set_version']}`; labels: `{report['label_version']}`.",
             f"Trained: {report['trained_at']}. Scores: {report['scored_at']} UTC. Scope: benchmark. Production approval: false.",
             f"Analysis SHA-256: `{report['analysis_sha256']}`.", "", "## Technical performance", "",
             "Persisted Phase 4 test metrics; the split is non-temporal. These metrics are not recalculated from the full scoring population.", "",
             "| Metric | Value |", "| --- | ---: |"]
    lines += [f"| {name} | {number:.9f} |" for name, number in report["technical_test_metrics"].items()]
    lines += [f"| Test class prevalence | {report['test_prevalence']:.9f} |", "",
              "Split sizes: " + ", ".join(f"{name}={counts['rows']} ({counts['churned']} churners)" for name, counts in report["split_counts"].items()) + ".",
              "", "## Outreach scenarios — full snapshot", "",
              f"Population: {report['population']}; churners: {report['churners']}; prevalence: {report['prevalence']:.6%}.",
              "Includes training customers. Rank by score descending, then customer ID ascending for ties. Percentage capacities round up. Each row is a scenario, not a chosen business capacity.",
              "Precision equals selected churn prevalence; recall equals cumulative churn capture. The cutoff is the last selected score; ties can mean a threshold alone selects more than N.", "",
              "| Outreach capacity | Customers selected | Churners captured | Precision / prevalence | Recall / cumulative capture | Cutoff |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['capacity']} | {r['selected']} | {r['churners']} | {r['precision']:.6f} | {value(r['recall'])} | {r['threshold_used']:.9f} |" for r in report["outreach"]]
    lines += ["", "## Provisional benchmark tiers", "",
              "High starts at the persisted validation-F1 threshold; medium starts at half that threshold. These are not business-approved categories.",
              "Lower bounds are inclusive; upper bounds are exclusive except high includes 1. Low starts at 0. The registry's low_threshold names the medium tier's lower bound.", "",
              "| Tier | Score interval | Customers | % Population | Churners | Churn rate |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {r['tier']} | [{r['lower_bound']}, {r['upper_bound']}{']' if r['tier'] == 'high' else ')'} | {r['customers']} | {r['population_fraction']:.6%} | {r['churners']} | {value(r['churn_rate'])} |" for r in report["tiers"]]
    lines += ["", "## Business acceptance", "",
              "Business acceptance: pending. Stakeholder approval: pending. Production approval: false. No business sign-off can be issued from this analysis.", ""]
    lines += [f"- {key}: {'unresolved' if item is None else 'configured: ' + str(item)}" for key, item in report["criteria"].items() if key != "stakeholder_approved"]
    lines += ["", f"Maximum original-feature importance: {report['maximum_feature_importance']:.9f}. The 0.5 default guardrail comes from the SDD; it is diagnostic and does not prove temporal safety.",
              "Diagnostic comparisons (null means unresolved; true never grants acceptance): " + json.dumps(report["diagnostic_criteria_checks"]),
              "", "Unresolved decisions: stakeholder criteria/sign-off, business churn definition and future horizon, verified active cohort, outreach costs, and approval of tier policy. Configured candidate criteria still require documented stakeholder review.",
              "", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "## Traceability", "",
              f"Evaluation report SHA-256: `{report['evaluation_report_sha256']}`.",
              f"Evaluated dataset SHA-256: `{report['dataset_sha256']}`.",
              f"Feature snapshot SHA-256: `{report['feature_snapshot_sha256']}`.",
              f"Ranked CSV SHA-256: `{report['ranking_sha256']}`.",
              "The adjacent .json file contains exact values; the adjacent .ranked.csv contains all customer ranks and supplied labels. Neither is an approved contact list.", ""]
    return "\n".join(lines)


def run_validation(model_id, feature_version=FEATURE_SET_VERSION, top_k=(), report_path=None, save=False):
    engine = get_engine()
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            with connection.begin():
                registry, evaluation, ranked = load_inputs(connection, model_id, feature_version)
        report = build_report(registry, evaluation, ranked, top_k)
        if report_path is not None:
            path = Path(report_path).resolve()
            if path.suffix != ".md" or path.is_relative_to(Path(registry["artifact_path"]).resolve()):
                raise ValueError("Report must use .md and be outside the registered model artifact directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render_report(report).encode())
            path.with_suffix(".json").write_bytes(json.dumps(report, indent=2, allow_nan=False).encode())
            path.with_suffix(".ranked.csv").write_bytes(ranking_csv(ranked).encode())
        if save:
            save_validation(engine, report)
        return report
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--feature-set-version", default=FEATURE_SET_VERSION)
    parser.add_argument("--top-k", type=int, action="append", default=[], help="Additional scenario N; repeatable, not a stakeholder decision")
    parser.add_argument("--report-path", type=Path, default=PROJECT_ROOT / "models" / "phase6_business_validation" / "report.md")
    parser.add_argument("--save", action="store_true", help="Persist pending scenario records")
    args = parser.parse_args()
    report = run_validation(args.model_id, args.feature_set_version, args.top_k, args.report_path, args.save)
    print(json.dumps({"model_id": args.model_id, "analysis_sha256": report["analysis_sha256"],
                      "population": report["population"], "report_path": str(args.report_path),
                      "saved": args.save, "stakeholder_status": "pending", "production_approved": False}, indent=2))


if __name__ == "__main__":
    main()
