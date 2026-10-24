#!/bin/sh
# Install an owner-approved cron cadence only after the production gates pass.
set -eu
cd "${CHURN_PROJECT_DIR:-/opt/churn-analytics}"
set -a
. "${CHURN_DEPLOY_ENV_FILE:-/etc/churn/deploy.env}"
set +a
exec ./venv/bin/python -m src.deploy --scheduled
