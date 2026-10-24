"""Monitor retained dated production cohorts; never infer missing outcomes."""

import argparse
from datetime import date, datetime, time, timedelta
import hashlib
from html import escape
import json
import logging
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src import config
from src.deploy import LoggingAlertSink, error_summary, utcnow
from src.features import FEATURE_COLUMNS, NUMERIC, validate_feature_frame
from src.monitoring_history import batch_hash
from src.score import validate_prediction_batch
from src.utils import get_engine

LOG = logging.getLogger(__name__)
BATCH_COLUMNS = ("batch_sha256, run_id, model_id, feature_set_version, label_version, observation_date, "
                 "scored_at, customer_count, feature_snapshot_sha256, report_sha256, low_threshold, "
                 "high_threshold, future_window_days, evaluation_threshold, baseline_pr_auc, "
                 "baseline_precision, baseline_recall, trained_at")
POLICY_KEYS = ("numeric_mean_shift_max", "categorical_tv_max", "precision_min", "retrain_interval_days")


def report_digest(report):
    data = {key: value for key, value in report.items() if key != "analysis_sha256"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_policy(policy):
    if set(policy) != set(POLICY_KEYS):
        raise ValueError("Monitoring policy has missing or unknown keys")
    for key in POLICY_KEYS[:-1]:
        value = policy[key]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                  or value < 0 or (key != "numeric_mean_shift_max" and value > 1)):
            raise ValueError(f"Invalid monitoring policy: {key}")
    interval = policy["retrain_interval_days"]
    if interval is not None and (type(interval) is not int or interval < 1):
        raise ValueError("Retraining interval must be a positive integer or None")


def _timestamp(value, field):
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid retained monitoring date: {field}")
    return parsed


def _validate_history(batch, history, as_of):
    if (batch["feature_set_version"] == config.FEATURE_SET_VERSION
            or batch["label_version"] == config.LABEL_VERSION):
        raise ValueError("Monitoring requires dated production history, not the Telco benchmark")
    for key in ("batch_sha256", "feature_snapshot_sha256", "report_sha256"):
        if not isinstance(batch[key], str) or not re.fullmatch(r"[0-9a-f]{64}", batch[key]):
            raise ValueError("Invalid retained monitoring identity")
    observed = _timestamp(batch["observation_date"], "observation_date").date()
    scored = _timestamp(batch["scored_at"], "scored_at")
    trained = _timestamp(batch["trained_at"], "trained_at")
    if observed > as_of or observed > scored.date() or scored.date() > as_of or trained > scored:
        raise ValueError("Retained batch is unavailable at the requested analysis date")
    if (type(batch["future_window_days"]) is not int or batch["future_window_days"] < 1
            or batch["customer_count"] != len(history)):
        raise ValueError("Invalid retained cohort horizon or customer count")
    validate_feature_frame(history)
    features = history[["customer_id", *FEATURE_COLUMNS]].copy()
    features[NUMERIC] = features[NUMERIC].astype(float)
    snapshot_hash, identity = batch_hash(batch, features, observed, batch["low_threshold"], batch["high_threshold"])
    if snapshot_hash != batch["feature_snapshot_sha256"]:
        raise ValueError("Retained feature snapshot hash mismatch")
    if identity != batch["batch_sha256"]:
        raise ValueError("Retained batch identity mismatch")
    predictions = history[["customer_id", "churn_probability", "risk_tier"]].assign(
        model_id=batch["model_id"], scoring_mode="production", scored_at=batch["scored_at"],
        feature_set_version=batch["feature_set_version"], feature_snapshot_sha256=snapshot_hash)
    validate_prediction_batch(predictions, history.customer_id, batch["model_id"], batch["feature_set_version"],
                              snapshot_hash, batch["low_threshold"], batch["high_threshold"], batch["scored_at"])
    for key in ("baseline_pr_auc", "baseline_precision", "baseline_recall", "evaluation_threshold"):
        value = batch[key]
        if value is None and key == "evaluation_threshold":
            continue
        if value is None or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid retained evaluation baseline")


def load_inputs(connection, batch_id=None, reference_batch_id=None, as_of=None):
    as_of = utcnow().date() if as_of is None else as_of
    if type(as_of) is not date or as_of > utcnow().date():
        raise ValueError("Analysis date must be a date no later than today UTC")
    for value in (batch_id, reference_batch_id):
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Monitoring batch IDs must be SHA-256 digests")
    cutoff = datetime.combine(as_of + timedelta(days=1), time.min)
    condition = "batch_sha256 = :batch_id" if batch_id else "observation_date <= :as_of AND scored_at < :cutoff"
    batch = connection.execute(text(f"SELECT {BATCH_COLUMNS} FROM scoring_batches WHERE {condition} "
                                    "ORDER BY observation_date DESC, scored_at DESC, batch_sha256 DESC LIMIT 1"),
                               {"batch_id": batch_id, "as_of": as_of, "cutoff": cutoff}).mappings().first()
    if batch is None:
        raise ValueError("No retained production scoring cohort is available; production monitoring is pending")
    batch = dict(batch)

    def read_history(metadata):
        # shortcut: in-memory cohorts; aggregate or chunk before validating a 1M-customer SLA.
        frame = pd.read_sql_query(text(f"SELECT customer_id, {', '.join(FEATURE_COLUMNS)}, "
                                       "churn_probability, risk_tier FROM scoring_history "
                                       "WHERE batch_sha256 = :batch_id ORDER BY customer_id"), connection,
                                  params={"batch_id": metadata["batch_sha256"]})
        _validate_history(metadata, frame, as_of)
        return frame

    history = read_history(batch)
    condition = "batch_sha256 = :reference_id" if reference_batch_id else (
        "feature_set_version = :version AND observation_date < :observation_date AND scored_at < :cutoff")
    reference = connection.execute(text(f"SELECT {BATCH_COLUMNS} FROM scoring_batches WHERE {condition} "
                                        "ORDER BY observation_date DESC, scored_at DESC, batch_sha256 DESC LIMIT 1"),
                                   {"reference_id": reference_batch_id, "version": batch["feature_set_version"],
                                    "observation_date": batch["observation_date"], "cutoff": cutoff}).mappings().first()
    reference_history = None
    if reference is None and reference_batch_id:
        raise ValueError("Reference scoring cohort was not found")
    if reference is not None:
        reference = dict(reference)
        if (reference["feature_set_version"] != batch["feature_set_version"]
                or _timestamp(reference["observation_date"], "reference_date").date()
                >= _timestamp(batch["observation_date"], "observation_date").date()):
            raise ValueError("Reference must have the same feature version and an earlier observation date")
        reference_history = read_history(reference)
    outcomes = pd.read_sql_query(text("SELECT customer_id, label_version, label_window_end, observed_through, "
                                      "available_at, churned, evidence_reference FROM monitoring_outcomes "
                                      "WHERE batch_sha256 = :batch_id ORDER BY customer_id"), connection,
                                 params={"batch_id": batch["batch_sha256"]})
    return batch, history, reference, reference_history, outcomes


def feature_drift(current, reference):
    rows = []
    for column in FEATURE_COLUMNS:
        now, previous = current[column], reference[column]
        row = {"feature": column, "current_null_rate": float(now.isna().mean()),
               "reference_null_rate": float(previous.isna().mean())}
        if column in NUMERIC:
            now, previous = now.astype(float), previous.astype(float)
            current_mean, reference_mean = float(now.mean()), float(previous.mean())
            current_std, reference_std = float(now.std(ddof=0)), float(previous.std(ddof=0))
            difference = abs(current_mean - reference_mean)
            row.update(kind="numeric", current_mean=current_mean, reference_mean=reference_mean,
                       current_std=current_std, reference_std=reference_std,
                       normalized_mean_shift=difference / reference_std if reference_std else (0.0 if not difference else None),
                       constant_reference_changed=bool(not reference_std and difference))
        else:
            now, previous = now.value_counts(normalize=True), previous.value_counts(normalize=True)
            categories = sorted(set(now.index) | set(previous.index))
            row.update(kind="categorical", total_variation=float(sum(abs(now.get(c, 0) - previous.get(c, 0))
                                                                      for c in categories) / 2),
                       current_distribution={c: float(now.get(c, 0)) for c in categories},
                       reference_distribution={c: float(previous.get(c, 0)) for c in categories})
        rows.append(row)
    return rows


def realized_performance(batch, history, outcomes, as_of):
    window_end = _timestamp(batch["observation_date"], "observation_date").date() + timedelta(days=batch["future_window_days"])
    result = {"status": "pending", "reason": None, "window_end": window_end.isoformat(),
              "customers": len(history), "recorded_outcomes": len(outcomes), "available_outcomes": 0,
              "metrics": None, "baseline_comparison": {"comparable": False, "precision_delta": None,
                                                        "recall_delta": None, "pr_auc_delta": None}}
    if not outcomes.empty:
        if (outcomes.customer_id.isna().any() or not outcomes.customer_id.is_unique
                or not set(outcomes.customer_id).issubset(history.customer_id)
                or not outcomes.label_version.eq(batch["label_version"]).all()
                or not outcomes.churned.isin([0, 1]).all()
                or outcomes.evidence_reference.isna().any()
                or outcomes.evidence_reference.astype(str).str.strip().eq("").any()
                or outcomes.evidence_reference.astype(str).str.len().gt(255).any()):
            raise ValueError("Invalid monitoring outcome coverage, label version, values, or evidence")
        ends = pd.to_datetime(outcomes.label_window_end, errors="coerce", utc=True)
        through = pd.to_datetime(outcomes.observed_through, errors="coerce", utc=True)
        available = pd.to_datetime(outcomes.available_at, errors="coerce", utc=True)
        expected_end = pd.Timestamp(window_end, tz="UTC")
        if (ends.isna().any() or through.isna().any() or available.isna().any()
                or not ends.eq(expected_end).all() or through.lt(ends).any()
                or available.dt.normalize().lt(through).any()):
            raise ValueError("Monitoring outcomes require exact mature windows and observation provenance")
        cutoff = pd.Timestamp(as_of + timedelta(days=1), tz="UTC")
        usable = outcomes.loc[available.lt(cutoff)]
        result["available_outcomes"] = len(usable)
    else:
        usable = outcomes
    if as_of < window_end:
        result["reason"] = "future_window_not_mature"
        return result
    if len(usable) != len(history):
        result["reason"] = "outcomes_incomplete_or_unavailable"
        return result
    evaluated = history.merge(usable[["customer_id", "churned"]], on="customer_id", validate="one_to_one")
    labels = evaluated.churned.to_numpy(dtype=int)
    scores = evaluated.churn_probability.to_numpy(dtype=float)
    predicted = scores >= batch["high_threshold"]
    positives, true_positive = int(labels.sum()), int(labels[predicted].sum())
    order = np.lexsort((evaluated.customer_id.to_numpy(), -scores))
    top = labels[order[:math.ceil(len(labels) / 10)]]
    metrics = {"threshold": float(batch["high_threshold"]), "churners": positives,
               "selected": int(predicted.sum()), "true_positives": true_positive,
               "precision": true_positive / int(predicted.sum()) if predicted.any() else 0.0,
               "recall": true_positive / positives if positives else None,
               "pr_auc_average_precision": float(average_precision_score(labels, scores)) if positives else None,
               "roc_auc": float(roc_auc_score(labels, scores)) if 0 < positives < len(labels) else None,
               "brier_score": float(np.mean((scores - labels) ** 2)),
               "precision_at_top_decile": float(top.mean()),
               "recall_at_top_decile": int(top.sum()) / positives if positives else None}
    comparable = batch["evaluation_threshold"] == batch["high_threshold"]
    result.update(status="complete", metrics=metrics, evidence_references=sorted(usable.evidence_reference.unique().tolist()))
    result["baseline_comparison"] = {
        "comparable": comparable,
        "precision_delta": metrics["precision"] - batch["baseline_precision"] if comparable else None,
        "recall_delta": metrics["recall"] - batch["baseline_recall"] if comparable and positives else None,
        "pr_auc_delta": metrics["pr_auc_average_precision"] - batch["baseline_pr_auc"] if positives else None,
    }
    return result


def build_report(batch, history, reference, reference_history, outcomes, as_of, policy=None):
    policy = dict(config.MONITORING if policy is None else policy)
    validate_policy(policy)
    drift = feature_drift(history, reference_history) if reference is not None else []
    performance = realized_performance(batch, history, outcomes, as_of)
    reasons = []
    for row in drift:
        if (row["kind"] == "numeric" and policy["numeric_mean_shift_max"] is not None
                and (row["constant_reference_changed"]
                     or row["normalized_mean_shift"] > policy["numeric_mean_shift_max"])):
            reasons.append(f"Numeric mean drift: {row['feature']}")
        if (row["kind"] == "categorical" and policy["categorical_tv_max"] is not None
                and row["total_variation"] > policy["categorical_tv_max"]):
            reasons.append(f"Categorical drift: {row['feature']}")
    if (performance["status"] == "complete" and policy["precision_min"] is not None
            and performance["metrics"]["precision"] < policy["precision_min"]):
        reasons.append("Realized precision below configured floor")
    age = (as_of - _timestamp(batch["trained_at"], "trained_at").date()).days
    if policy["retrain_interval_days"] is not None and age >= policy["retrain_interval_days"]:
        reasons.append("Configured retraining interval elapsed")
    report = {
        "analysis_date": as_of.isoformat(), "scope": "retained_dated_production_cohort",
        "evidence_scope": "Retained dated production cohort; no benchmark outcomes are inferred.",
        "batch": json.loads(json.dumps(batch, default=str)),
        "reference_batch": json.loads(json.dumps(reference, default=str)) if reference else None,
        "drift": {"status": "complete" if reference else "pending", "features": drift,
                  "reason": None if reference else "no_earlier_compatible_reference"},
        "performance": performance,
        "held_out_baseline": {"precision": batch["baseline_precision"], "recall": batch["baseline_recall"],
                              "pr_auc_average_precision": batch["baseline_pr_auc"],
                              "evaluation_threshold": batch["evaluation_threshold"]},
        "retraining": {"status": "recommended" if reasons else ("unconfigured" if all(v is None for v in policy.values()) else "no_trigger"),
                       "reasons": reasons, "model_age_days": age, "automatic_retraining": False,
                       "policy": policy, "unresolved": [key for key, value in policy.items() if value is None]},
        "limitations": ["Feature drift is descriptive; it does not establish model degradation or causation.",
                        "Precision/recall use the archived accepted high-risk boundary; threshold baseline comparisons require an identical evaluation threshold.",
                        "Missing or unavailable outcomes are never inferred as retained customers.",
                        "Recommendations do not train or promote a model; temporal retraining and operational policy remain governed prerequisites."],
    }
    report["analysis_sha256"] = report_digest(report)
    return report


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["analysis_sha256"] = report_digest(report)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    performance, retraining = report["performance"], report["retraining"]
    lines = ["# Production cohort monitoring", "", report["evidence_scope"], "", f"Analysis date: {report['analysis_date']}",
             f"Batch: `{report['batch']['batch_sha256']}`", f"Model: {report['batch']['model_id']}", "",
             f"Feature drift: {report['drift']['status']}", f"Realized performance: {performance['status']}",
             f"Retraining recommendation: {retraining['status']}", "", "## Realized metrics", "",
             "| Metric | Value |", "| --- | --- |"]
    lines.extend(f"| {key} | {value} |" for key, value in (performance["metrics"] or {}).items())
    if performance["reason"]:
        lines.extend(["", f"Pending reason: {performance['reason']}"])
    comparison_rows = [("precision", "precision_delta"), ("recall", "recall_delta"),
                       ("pr_auc_average_precision", "pr_auc_delta")]
    comparisons = [(metric, (performance["metrics"] or {}).get(metric), report["held_out_baseline"][metric],
                    performance["baseline_comparison"][delta]) for metric, delta in comparison_rows]
    comparison_note = ("Precision/recall thresholds match the held-out evaluation." if performance["baseline_comparison"]["comparable"]
                       else "Precision/recall comparison is pending or uses a different evaluation threshold. AP is independent of threshold.")
    lines.extend(["", "## Held-out baseline comparison", "", comparison_note, "",
                  "| Metric | Realized | Held-out | Delta |", "| --- | --- | --- | --- |"])
    lines.extend(f"| {metric} | {current} | {baseline} | {delta} |" for metric, current, baseline, delta in comparisons)
    lines.extend(["", "## Feature drift", "", "| Feature | Kind | Mean shift / total variation |", "| --- | --- | --- |"])
    lines.extend(f"| {row['feature']} | {row['kind']} | {row.get('normalized_mean_shift', row.get('total_variation'))} |"
                 for row in report["drift"]["features"])
    lines.extend(["", "## Retraining policy", "", "Recommendations never trigger training or model approval.", ""])
    lines.extend(f"- {reason}" for reason in (retraining["reasons"] or ["No configured trigger fired."]))
    lines.extend(["", "## Limits", "", *[f"- {value}" for value in report["limitations"]], ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    metrics = "".join(f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>"
                      for key, value in (performance["metrics"] or {}).items())
    drift = "".join(f"<tr><th>{escape(row['feature'])}</th><td>{escape(row['kind'])}</td>"
                    f"<td>{escape(str(row.get('normalized_mean_shift', row.get('total_variation'))))}</td></tr>"
                    for row in report["drift"]["features"])
    reasons = "".join(f"<li>{escape(value)}</li>" for value in (retraining["reasons"] or ["No configured trigger fired."]))
    limits = "".join(f"<li>{escape(value)}</li>" for value in report["limitations"])
    baseline = "".join(f"<tr><th>{escape(metric)}</th><td>{escape(str(current))}</td>"
                       f"<td>{escape(str(saved))}</td><td>{escape(str(delta))}</td></tr>"
                       for metric, current, saved, delta in comparisons)
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Production cohort monitoring</title><style>body{{font:16px system-ui;margin:3rem auto;padding:0 1rem;max-width:1100px;color:#17243a;background:#f5f7fb}}h1{{font-size:2rem}}.cards{{display:flex;gap:1rem;flex-wrap:wrap}}.card,section{{background:white;padding:1.2rem;border:1px solid #d7deea;border-radius:12px;margin:1rem 0}}.card{{flex:1}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #e2e7f0}}code{{overflow-wrap:anywhere}}small{{color:#42546c}}</style>
<h1>Production cohort monitoring</h1><p>{escape(report['evidence_scope'])}</p><p>{escape(report['analysis_date'])} · Model {escape(report['batch']['model_id'])}</p><small>Batch <code>{escape(report['batch']['batch_sha256'])}</code></small>
<div class="cards"><div class="card">Customers<br><strong>{report['batch']['customer_count']}</strong></div><div class="card">Feature drift<br><strong>{escape(report['drift']['status'])}</strong></div><div class="card">Realized performance<br><strong>{escape(performance['status'])}</strong></div><div class="card">Retraining<br><strong>{escape(retraining['status'])}</strong></div></div>
<section><h2>Realized metrics</h2><p>{escape(performance['reason'] or 'Complete matured cohort; no missing outcomes inferred.')}</p><table><thead><tr><th scope="col">Metric</th><th scope="col">Value</th></tr></thead><tbody>{metrics}</tbody></table></section>
<section><h2>Held-out baseline comparison</h2><p>{escape(comparison_note)}</p><table><thead><tr><th scope="col">Metric</th><th scope="col">Realized</th><th scope="col">Held-out</th><th scope="col">Delta</th></tr></thead><tbody>{baseline}</tbody></table></section>
<section><h2>Feature drift</h2><p>Numeric values show normalized mean shift. Categorical values show total variation. A changed constant reference has no finite normalized shift; exact values are in report.json.</p><table><thead><tr><th scope="col">Feature</th><th scope="col">Kind</th><th scope="col">Shift</th></tr></thead><tbody>{drift}</tbody></table></section>
<section><h2>Retraining recommendation</h2><ul>{reasons}</ul><p>Automatic retraining is disabled. Policy: <code>{escape(json.dumps(retraining['policy'], sort_keys=True))}</code></p></section><section><h2>Interpretation and limits</h2><ul>{limits}</ul></section></html>"""
    (output_dir / "dashboard.html").write_text(html, encoding="utf-8")


def run_monitoring(*, batch_id=None, reference_batch_id=None, as_of=None, output_dir=None,
                   policy=None, engine=None, alert_sink=None):
    sink = LoggingAlertSink() if alert_sink is None else alert_sink
    as_of = utcnow().date() if as_of is None else as_of
    owned_engine = engine is None

    def emit(event):
        try:
            sink.emit(event)
        except Exception:
            LOG.error("Monitoring alert delivery failed; batch=%s", batch_id)

    try:
        if engine is None:
            engine = get_engine()
        isolation = "REPEATABLE READ" if engine.dialect.name == "mysql" else "SERIALIZABLE"
        with engine.connect().execution_options(isolation_level=isolation) as connection, connection.begin():
            inputs = load_inputs(connection, batch_id, reference_batch_id, as_of)
            report = build_report(*inputs, as_of, policy)
        if output_dir is not None:
            write_report(report, output_dir)
        if report["retraining"]["reasons"]:
            emit({"event_type": "monitoring_retraining_recommended", "batch_sha256": report["batch"]["batch_sha256"],
                  "analysis_sha256": report["analysis_sha256"], "reasons": report["retraining"]["reasons"]})
        return report
    except Exception as exc:
        emit({"event_type": "monitoring_failure", "batch_sha256": batch_id, "error_summary": error_summary(exc)})
        raise
    finally:
        if owned_engine and engine is not None:
            engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id")
    parser.add_argument("--reference-batch-id")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--output-dir", type=Path, default=config.PROJECT_ROOT / "models" / "phase8_monitoring")
    parser.add_argument("--init-schema", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.init_schema:
            if args.batch_id or args.reference_batch_id or args.as_of:
                raise ValueError("Run --init-schema separately from monitoring")
            from src.monitoring_history import ensure_schema
            engine = get_engine()
            try:
                ensure_schema(engine)
            finally:
                engine.dispose()
            print(json.dumps({"monitoring_schema_initialized": True}))
        else:
            result = run_monitoring(batch_id=args.batch_id, reference_batch_id=args.reference_batch_id,
                                    as_of=args.as_of, output_dir=args.output_dir)
            print(json.dumps({"analysis_sha256": result["analysis_sha256"],
                              "performance_status": result["performance"]["status"],
                              "retraining_status": result["retraining"]["status"],
                              "output_dir": str(args.output_dir)}, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, SQLAlchemyError, OSError) as exc:
        LOG.error(error_summary(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
