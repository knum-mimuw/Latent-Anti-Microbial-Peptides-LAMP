#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TRACKING_URI="${MLFLOW_TRACKING_URI:-file:./mlflow-store}"
# Absolute path so ZenML resolves the store correctly when the pipeline runs
# from a cwd other than the repo root (e.g. src/modelling).
ARTIFACT_PATH="$REPO_ROOT/zenml-artifacts"

ZENML=(uv run zenml)

"${ZENML[@]}" init 2>/dev/null || true
"${ZENML[@]}" artifact-store register local-store --flavor=local --path="$ARTIFACT_PATH" 2>/dev/null || true
"${ZENML[@]}" artifact-store update local-store --path="$ARTIFACT_PATH" 2>/dev/null || true
"${ZENML[@]}" experiment-tracker register mlflow-local --flavor=mlflow --tracking_uri="$TRACKING_URI" 2>/dev/null || true
"${ZENML[@]}" orchestrator register local-orch --flavor=local 2>/dev/null || true
"${ZENML[@]}" stack register lamp-local \
    --artifact-store=local-store \
    --experiment_tracker=mlflow-local \
    --orchestrator=local-orch \
    --set 2>/dev/null || true

echo "Stack 'lamp-local' is active (tracking_uri=$TRACKING_URI)"
