#!/usr/bin/env bash
set -euo pipefail

mkdir -p mlflow-store

echo "Starting MLflow server at http://127.0.0.1:5000"
echo "Set MLFLOW_TRACKING_URI=http://127.0.0.1:5000 in your .env to use it."

mlflow server \
    --backend-store-uri "sqlite:///mlflow-store/mlflow.db" \
    --default-artifact-root ./mlflow-store/artifacts \
    --host 127.0.0.1 \
    --port 5000
