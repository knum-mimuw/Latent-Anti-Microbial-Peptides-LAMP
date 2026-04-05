#!/usr/bin/env bash
set -euo pipefail

TRACKING_URI="${MLFLOW_TRACKING_URI:-file:./mlflow-store}"

zenml init 2>/dev/null || true
zenml artifact-store register local-store --flavor=local --path=./zenml-artifacts 2>/dev/null || true
zenml experiment-tracker register mlflow-local --flavor=mlflow --tracking_uri="$TRACKING_URI" 2>/dev/null || true
zenml orchestrator register local-orch --flavor=local 2>/dev/null || true
zenml stack register lamp-local \
    --artifact-store=local-store \
    --experiment-tracker=mlflow-local \
    --orchestrator=local-orch \
    --set 2>/dev/null || true

echo "Stack 'lamp-local' is active (tracking_uri=$TRACKING_URI)"
