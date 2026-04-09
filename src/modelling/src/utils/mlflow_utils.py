"""MLflow utility functions for checkpoint and artifact management.

Works standalone -- no ZenML dependency. Reads MLFLOW_TRACKING_URI from the
environment when no explicit tracking_uri is provided.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mlflow import MlflowClient


def get_mlflow_client(tracking_uri: str | None = None) -> MlflowClient:
    """Return an MlflowClient configured from *tracking_uri* or the environment.

    Falls back to ``MLFLOW_TRACKING_URI`` env var, then to MLflow's own default
    (``./mlruns``).
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    return MlflowClient(tracking_uri=uri)


def download_artifact(
    run_id: str,
    artifact_path: str,
    dest_dir: str | Path = ".mlflow-cache",
    tracking_uri: str | None = None,
) -> Path:
    """Download any artifact from an MLflow run and return its local path."""
    client = get_mlflow_client(tracking_uri)
    dest = Path(dest_dir) / run_id
    dest.mkdir(parents=True, exist_ok=True)

    local_path = client.download_artifacts(run_id, artifact_path, str(dest))
    result = Path(local_path)
    print(f"Downloaded {artifact_path} from run {run_id} -> {result}")
    return result


def download_config(
    run_id: str,
    dest_dir: str | Path = ".mlflow-cache",
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Download and parse a logged config YAML from an MLflow run.

    Expects the config to have been logged as ``config/model_config.yaml``.

    Args:
        run_id: MLflow run ID.
        dest_dir: Local directory to save the config into.
        tracking_uri: Optional tracking URI override.

    Returns:
        Parsed config dictionary.
    """
    import yaml

    client = get_mlflow_client(tracking_uri)
    dest = Path(dest_dir) / run_id
    dest.mkdir(parents=True, exist_ok=True)

    local_path = client.download_artifacts(run_id, "config/model_config.yaml", str(dest))
    config_path = Path(local_path)

    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    print(f"Downloaded config from run {run_id} -> {config_path}")
    return config


def log_checkpoint_artifact(
    run_id: str,
    checkpoint_path: str | Path,
    artifact_path: str = "checkpoints",
    tracking_uri: str | None = None,
) -> None:
    """Log a checkpoint file as an artifact to an existing MLflow run.

    Args:
        run_id: MLflow run ID.
        checkpoint_path: Local path to the ``.ckpt`` file.
        artifact_path: Destination folder inside the run's artifact store.
        tracking_uri: Optional tracking URI override.
    """
    client = get_mlflow_client(tracking_uri)
    client.log_artifact(run_id, str(checkpoint_path), artifact_path)
    print(f"Logged {checkpoint_path} -> run {run_id}/{artifact_path}")


def log_artifact_directory(
    run_id: str,
    local_dir: str | Path,
    artifact_path: str,
    tracking_uri: str | None = None,
) -> None:
    """Log an entire local directory as an MLflow artifact subtree."""
    client = get_mlflow_client(tracking_uri)
    client.log_artifacts(run_id, str(local_dir), artifact_path)
    print(f"Logged {local_dir} -> run {run_id}/{artifact_path}")


def list_experiments(tracking_uri: str | None = None) -> list[dict[str, Any]]:
    """List all MLflow experiments.

    Returns:
        List of experiment metadata dicts.
    """
    client = get_mlflow_client(tracking_uri)
    experiments = client.search_experiments()
    return [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
        }
        for exp in experiments
    ]


def list_runs(
    experiment_name: str,
    tracking_uri: str | None = None,
    max_results: int = 100,
) -> list[dict[str, Any]]:
    """List runs for a given experiment name.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Optional tracking URI override.
        max_results: Maximum number of runs to return.

    Returns:
        List of run metadata dicts.
    """
    client = get_mlflow_client(tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=max_results,
        order_by=["start_time DESC"],
    )
    return [
        {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
        }
        for run in runs
    ]


def get_run_summary(run_id: str, tracking_uri: str | None = None) -> dict[str, Any]:
    """Return a structured summary for a single MLflow run."""
    client = get_mlflow_client(tracking_uri)
    run = client.get_run(run_id)
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "experiment_id": run.info.experiment_id,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }
