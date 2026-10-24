"""Fail-closed production batch deployment; the Telco benchmark cannot deploy."""

import argparse
from datetime import date, datetime, timezone
import json
import logging
import re
from typing import Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from src import config
from src.features import FEATURE_COLUMNS, load_active_customers, refresh_production_features
from src.monitoring_history import archive_batch, batch_hash, ensure_schema as ensure_monitoring_schema
from src.registry import METRICS, ensure_schema as ensure_model_schema, load_artifacts
from src.score import (build_prediction_frame, predict_probabilities, risk_tiers,
                       validate_prediction_batch, write_predictions)
from src.utils import get_engine

LOG = logging.getLogger(__name__)
STAGES = ("PREFLIGHT", "DATA_VALIDATION", "FEATURE_REFRESH", "MODEL_LOAD", "SCORING",
          "PREDICTION_VALIDATION", "PUBLICATION")
REGISTRY_COLUMNS = ("model_id, trained_at, approved, evaluation_scope, feature_set_version, "
                    "label_version, artifact_path, report_sha256, dataset_sha256, "
                    "low_threshold, high_threshold, " + ", ".join(METRICS))
PREDICTION_COLUMNS = ("customer_id, model_id, scoring_mode, scored_at, feature_set_version, "
                      "feature_snapshot_sha256, churn_probability, risk_tier")
SOURCE_COLUMNS = {
    "customers": {"customer_id", "signup_date"},
    "subscriptions": {"customer_id", "status", "start_date", "end_date"},
    "billing_events": {"customer_id", "event_date"},
    "usage_events": {"customer_id", "event_date"},
    "support_tickets": {"customer_id", "opened_at"},
    config.TABLE_FEATURES: {"customer_id", "feature_set_version", "observation_date", *FEATURE_COLUMNS},
}


class AlertSink(Protocol):
    def emit(self, event: dict) -> None: ...


class LoggingAlertSink:
    def emit(self, event):
        LOG.error(json.dumps({"event": "alert", **event}, default=str, sort_keys=True))


class DeploymentError(RuntimeError):
    def __init__(self, run):
        self.run = run
        super().__init__(f"{run['failure_stage']}: {run['error_summary']} (run {run['run_id']})")


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def log_event(event, run):
    LOG.info(json.dumps({"event": event, **run}, default=str, sort_keys=True))


def alert(sink, event, run):
    try:
        sink.emit({"event_type": event, **run})
    except Exception as exc:
        # Alert delivery cannot undo a committed batch or hide the original failure.
        LOG.error("Alert delivery failed (%s); run_id=%s", type(exc).__name__, run["run_id"])


def validate_configuration(settings, environment, database_uri, observation_date, scheduled=False):
    if not database_uri:
        raise ValueError("DATABASE_URI is required; production deployment is blocked")
    try:
        url = make_url(database_uri)
        valid_uri = url.drivername in ("mysql+mysqlconnector", "mysql+pymysql") and url.host and url.database and url.username
    except Exception:
        valid_uri = False
    if not valid_uri:
        raise ValueError("DATABASE_URI must explicitly name a MySQL driver, host, database and user")
    if environment != "production":
        raise ValueError("Production deployment requires explicit APP_ENV=production")
    version = settings["feature_set_version"]
    if (not re.fullmatch(r"[A-Za-z0-9_]{1,30}", version)
            or version == config.FEATURE_SET_VERSION):
        raise ValueError("Production requires a non-benchmark feature version; telco_snapshot_v1 cannot deploy")
    if settings["model_id"] is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", settings["model_id"]):
        raise ValueError("Invalid production model ID")
    try:
        low, high = float(settings["low_threshold"]), float(settings["high_threshold"])
        risk_tiers([], low, high)
    except (TypeError, ValueError):
        raise ValueError("Set valid PRODUCTION_LOW_THRESHOLD and PRODUCTION_HIGH_THRESHOLD") from None
    for key, required in (("output_table", config.TABLE_PREDICTIONS), ("publication", "atomic_replace"),
                          ("feature_refresh", "external"), ("alert_mode", "logging")):
        if settings[key] != required:
            raise ValueError(f"Unsupported production configuration: {key}; expected {required}")
    if settings["schedule_enabled"] not in ("true", "false"):
        raise ValueError("DEPLOY_SCHEDULE_ENABLED must be true or false")
    if scheduled and (settings["schedule_enabled"] != "true" or not settings["schedule_cron"].strip()):
        raise ValueError("Scheduling is disabled or DEPLOY_SCHEDULE_CRON is undefined")
    if type(observation_date) is not date or observation_date > utcnow().date():
        raise ValueError("The observation date must be a real date no later than today (UTC)")
    return low, high


def ensure_schema(engine):
    """One-time setup only; MySQL DDL must never run inside deployment publication."""
    ensure_model_schema(engine)
    with engine.begin() as connection:
        for path in (config.DEPLOYMENT_SCHEMA_PATH, config.PRODUCTION_STATE_SCHEMA_PATH):
            connection.exec_driver_sql(path.read_text(encoding="utf-8").rstrip(";\n"))
        connection.execute(text("INSERT INTO production_state (state_id) VALUES (1) "
                                "ON DUPLICATE KEY UPDATE state_id = state_id"))
    ensure_monitoring_schema(engine)


def validate_database(connection):
    if connection.dialect.name != "mysql":
        raise ValueError("Production publication requires MySQL/InnoDB")
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    required = {*SOURCE_COLUMNS, config.TABLE_REGISTRY, config.TABLE_PREDICTIONS,
                "deployment_runs", "production_state", config.TABLE_SCORING_BATCHES,
                config.TABLE_SCORING_HISTORY, config.TABLE_MONITORING_OUTCOMES}
    missing = required - tables
    if missing:
        raise ValueError("Required production source/output tables missing: " + ", ".join(sorted(missing)))
    for table, columns in SOURCE_COLUMNS.items():
        missing = columns - {column["name"] for column in inspector.get_columns(table)}
        if missing:
            raise ValueError(f"Production columns missing from {table}: " + ", ".join(sorted(missing)))
    engines = dict(connection.execute(text("SELECT TABLE_NAME, ENGINE FROM information_schema.TABLES "
                                           "WHERE TABLE_SCHEMA = DATABASE()" )).all())
    if any(engines.get(table, "").lower() != "innodb" for table in required):
        raise ValueError("All production source, feature, registry and publication tables must use InnoDB")


def locking_clause(connection):
    # SQLite is used only in isolated transaction tests; validate_database rejects it in production.
    return " FOR UPDATE" if connection.dialect.name == "mysql" else ""


def lock_state(connection):
    state = connection.execute(text("SELECT model_id, run_id FROM production_state "
                                    "WHERE state_id = 1" + locking_clause(connection))).mappings().first()
    if state is None:
        raise ValueError("Production state is not initialized; run --init-schema separately")
    return state


def select_model(connection, feature_version, model_id=None):
    condition = "model_id = :model_id" if model_id else (
        "approved = 1 AND evaluation_scope = 'production' AND feature_set_version = :version")
    row = connection.execute(text(f"SELECT {REGISTRY_COLUMNS} FROM {config.TABLE_REGISTRY} WHERE {condition} "
                                  "ORDER BY trained_at DESC, model_id DESC LIMIT 1" + locking_clause(connection)),
                             {"model_id": model_id, "version": feature_version}).mappings().first()
    if row is None:
        raise ValueError("No approved production model is available; production deployment is blocked")
    if row["approved"] != 1 or row["evaluation_scope"] != "production":
        raise ValueError("Only approved production models can deploy; benchmark models are rejected")
    if row["feature_set_version"] != feature_version or row["feature_set_version"] == config.FEATURE_SET_VERSION:
        raise ValueError("Production model feature-set version is incompatible")
    return row


def previous_model_run(connection, state):
    run_id, visited = state["run_id"], set()
    while run_id:
        if run_id in visited:
            raise ValueError("Invalid production run lineage")
        visited.add(run_id)
        row = connection.execute(text("SELECT run_id, model_id, status, previous_run_id FROM deployment_runs "
                                      "WHERE run_id = :run_id"), {"run_id": run_id}).mappings().one()
        if row["status"] not in ("succeeded", "rolled_back"):
            raise ValueError("Rollback lineage contains an unsuccessful run")
        if row["model_id"] != state["model_id"]:
            return row
        run_id = row["previous_run_id"]
    raise ValueError("No previous valid, distinct production model exists for rollback")


def save_run(connection, run, insert=False):
    if insert:
        columns = ", ".join(run)
        values = ", ".join(f":{name}" for name in run)
        connection.execute(text(f"INSERT INTO deployment_runs ({columns}) VALUES ({values})"), run)
    else:
        values = ", ".join(f"{name} = :{name}" for name in run if name != "run_id")
        result = connection.execute(text(f"UPDATE deployment_runs SET {values} "
                                         "WHERE run_id = :run_id AND status = 'running'"), run)
        if result.rowcount != 1:
            raise ValueError("Deployment run is missing or already terminal; history is immutable")


def read_predictions(connection):
    return pd.read_sql_query(text(f"SELECT {PREDICTION_COLUMNS} FROM {config.TABLE_PREDICTIONS} "
                                  "WHERE scoring_mode = 'production' ORDER BY customer_id"), connection,
                             parse_dates=["scored_at"])


def error_summary(exc):
    # DBAPI messages can contain credentials, SQL parameters or customer values.
    if isinstance(exc, SQLAlchemyError):
        return f"Database operation failed ({type(exc).__name__}); inspect protected database logs"
    if isinstance(exc, FileNotFoundError):
        return "Required model/preprocessing/report artifact is missing"
    if isinstance(exc, ValueError):
        summary = str(exc)
        if config.DATABASE_URI:
            summary = summary.replace(config.DATABASE_URI, "[redacted database URI]")
        return summary[:500]
    return f"{type(exc).__name__} during deployment; inspect the failing stage"


def run_deployment(*, model_id=None, rollback=False, observation_date=None, scheduled=False,
                   settings=None, environment=None, database_uri=None, engine=None, alert_sink=None):
    settings = dict(config.PRODUCTION if settings is None else settings)
    environment = config.APP_ENV if environment is None else environment
    database_uri = config.DATABASE_URI if database_uri is None else database_uri
    observation_date = utcnow().date() if observation_date is None else observation_date
    sink = LoggingAlertSink() if alert_sink is None else alert_sink
    owned_engine, recorded, stage = engine is None, False, "PREFLIGHT"
    run = {"run_id": uuid4().hex, "started_at": utcnow(), "completed_at": None,
           "environment": environment[:20], "action": "rollback" if rollback else "deploy",
           "model_id": None, "trained_at": None, "feature_set_version": None,
           "observation_date": None, "scoring_mode": "production", "customer_count": 0,
           "prediction_count": 0, "status": "running", "failure_stage": None, "error_summary": None,
           "report_sha256": None, "model_sha256": None, "preprocessing_sha256": None,
           "feature_snapshot_sha256": None, "batch_sha256": None, "scored_at": None,
           "previous_run_id": None, "rollback_target_run_id": None, "reused_batch": 0}
    log_event("deployment_started", run)
    try:
        low, high = validate_configuration(settings, environment, database_uri, observation_date, scheduled)
        if model_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", model_id):
            raise ValueError("Invalid requested model ID")
        if rollback and model_id:
            raise ValueError("Choose either --rollback or --model-id")
        run.update(feature_set_version=settings["feature_set_version"], observation_date=observation_date)
        if engine is None:
            engine = get_engine(database_uri)
        with engine.begin() as connection:
            save_run(connection, run, insert=True)
        recorded = True
        with engine.connect().execution_options(isolation_level="REPEATABLE READ" if engine.dialect.name == "mysql"
                                                else "SERIALIZABLE") as connection:
            with connection.begin():
                # All production writers lock the singleton before their model registry row.
                # shortcut: one global production lock; partition only if independent tenant cohorts exist.
                state = lock_state(connection)
                target = previous_model_run(connection, state) if rollback else None
                requested = target["model_id"] if target else model_id or state["model_id"] or settings["model_id"]
                registry = select_model(connection, settings["feature_set_version"], requested)
                run.update(model_id=registry["model_id"], trained_at=registry["trained_at"],
                           report_sha256=registry["report_sha256"], previous_run_id=state["run_id"],
                           rollback_target_run_id=target["run_id"] if target else None)
                validate_database(connection)
                stage = "MODEL_LOAD"
                report, model, preprocess, _ = load_artifacts(registry["artifact_path"], registry["report_sha256"],
                                                            production_registry=registry)
                if low != registry["low_threshold"] or high != registry["high_threshold"]:
                    raise ValueError("Configured production risk policy differs from accepted model policy")
                run.update(model_sha256=report["artifact_sha256"]["model.json"],
                           preprocessing_sha256=report["artifact_sha256"]["preprocessor.joblib"])
                log_event("production_model_selected", run)
                stage = "DATA_VALIDATION"
                active_ids = load_active_customers(connection, observation_date)
                run["customer_count"] = len(active_ids)
                stage = "FEATURE_REFRESH"
                frame = refresh_production_features(connection, settings["feature_set_version"], observation_date, active_ids)
                snapshot_hash, identity = batch_hash(registry, frame, observation_date, low, high)
                run.update(feature_snapshot_sha256=snapshot_hash, batch_sha256=identity)
                current = None
                if state["run_id"]:
                    current = connection.execute(text("SELECT batch_sha256, scored_at FROM deployment_runs "
                                                       "WHERE run_id = :run_id"), {"run_id": state["run_id"]}).mappings().one()
                reuse = current is not None and current["batch_sha256"] == identity
                scored_at = pd.Timestamp(current["scored_at"]).to_pydatetime() if reuse else utcnow()
                run.update(scored_at=scored_at, reused_batch=int(reuse))
                stage = "SCORING"
                probabilities = predict_probabilities(model, preprocess, frame)
                stage = "PREDICTION_VALIDATION"
                expected = build_prediction_frame(registry["model_id"], frame, probabilities, low, high, snapshot_hash,
                                                  "production", settings["feature_set_version"], scored_at)
                validation_args = (active_ids, registry["model_id"], settings["feature_set_version"], snapshot_hash,
                                   low, high, scored_at)
                validate_prediction_batch(expected, *validation_args)
                stage = "PUBLICATION"
                if not reuse:
                    write_predictions(connection, registry["model_id"], frame, probabilities, low, high, snapshot_hash,
                                      scoring_mode="production", feature_version=settings["feature_set_version"],
                                      scored_at=scored_at)
                stored = read_predictions(connection)
                stage = "PREDICTION_VALIDATION"
                validate_prediction_batch(stored, *validation_args)
                if not np.array_equal(stored.churn_probability.to_numpy(),
                                      expected.sort_values("customer_id").churn_probability.to_numpy()):
                    raise ValueError("Published probabilities differ from the validated scoring result")
                stage = "PUBLICATION"
                run.update(prediction_count=len(stored), completed_at=utcnow(),
                           status="rolled_back" if rollback else "succeeded")
                archive_batch(connection, run, frame, stored, registry, report, low, high)
                save_run(connection, run)
                connection.execute(text("UPDATE production_state SET model_id = :model_id, run_id = :run_id "
                                        "WHERE state_id = 1"), run)
        log_event("deployment_completed", run)
        if rollback:
            alert(sink, "rollback", run)
        return run
    except Exception as exc:
        run.update(status="failed", completed_at=utcnow(), failure_stage=stage, error_summary=error_summary(exc))
        # Publication transaction has rolled back. Persist failure in a separate transaction.
        try:
            if engine is None and database_uri:
                engine = get_engine(database_uri)
            if engine is not None:
                with engine.begin() as connection:
                    # A lost COMMIT acknowledgement may conceal a successful commit.
                    committed = connection.execute(text("SELECT status FROM deployment_runs WHERE run_id = :run_id"),
                                                   run).scalar_one_or_none()
                    if committed in ("succeeded", "rolled_back"):
                        run["error_summary"] = "Commit acknowledgement lost; successful run exists; inspect run before retry"
                        log_event("commit_outcome_recovered", run)
                    else:
                        save_run(connection, run, insert=not recorded)
        except Exception as audit_exc:
            LOG.error("Database audit unavailable (%s); structured failure log is the fallback; run_id=%s",
                      type(audit_exc).__name__, run["run_id"])
        log_event("deployment_failed", run)
        alert(sink, "deployment_failure", run)
        raise DeploymentError(run) from None
    finally:
        if owned_engine and engine is not None:
            engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--model-id", help="Activate this already-approved production model")
    selection.add_argument("--rollback", action="store_true", help="Re-score with the previous distinct valid model")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="External feature date (default: today UTC)")
    parser.add_argument("--scheduled", action="store_true", help="Require explicitly enabled schedule configuration")
    parser.add_argument("--init-schema", action="store_true", help="Create deployment tables only; no scoring")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.init_schema:
            if args.model_id or args.rollback or args.scheduled or args.as_of:
                raise ValueError("Run --init-schema separately from deployment")
            engine = get_engine()
            try:
                ensure_schema(engine)
            finally:
                engine.dispose()
            print(json.dumps({"schema_initialized": True, "production_deployed": False}))
        else:
            run = run_deployment(model_id=args.model_id, rollback=args.rollback, observation_date=args.as_of,
                                 scheduled=args.scheduled)
            print(json.dumps(run, default=str, sort_keys=True))
        return 0
    except (DeploymentError, ValueError, SQLAlchemyError, RuntimeError) as exc:
        LOG.error(str(exc) if isinstance(exc, DeploymentError) else error_summary(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
