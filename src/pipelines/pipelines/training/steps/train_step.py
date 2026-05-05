"""Training step — Hugging Face Trainer + Hydra via ``uv run modelling``."""

import os
import subprocess
from pathlib import Path
from typing import NamedTuple

import mlflow
from mlflow import MlflowClient
from zenml import step

from modelling.src.training.hydra_overrides import flatten_yaml_file

from ...utils.pipeline_utils import lamp_repo_root, load_run_config


class TrainResult(NamedTuple):
    """Structured training outputs for downstream pipeline composition."""

    checkpoint_artifact_path: str
    mlflow_run_id: str
    experiment_name: str | None


@step(enable_cache=False)
def train(config_paths: list[str], run_config_path: str) -> TrainResult:
    """Run modelling training and return MLflow run outputs."""
    repo_root = lamp_repo_root()
    run_config = load_run_config(run_config_path)
    env = _build_training_env(run_config.mlflow.experiment_name, repo_root=repo_root)

    cmd: list[str] = ["uv", "run", "modelling"]
    for cfg_path in config_paths:
        cmd.extend(flatten_yaml_file(Path(cfg_path)))

    subprocess.run(
        cmd,
        check=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    run_id = _find_latest_run_id(run_config.mlflow.experiment_name)
    checkpoint_artifact_path = _find_checkpoint_artifact(run_id)

    return TrainResult(
        checkpoint_artifact_path=checkpoint_artifact_path,
        mlflow_run_id=run_id,
        experiment_name=run_config.mlflow.experiment_name,
    )


def _build_training_env(experiment_name: str, *, repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    env["LAMP_REPO_ROOT"] = str(repo_root.resolve())
    return env


def _find_latest_run_id(experiment_name: str) -> str:
    """Get the most recent completed run in the experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"No finished runs in experiment '{experiment_name}'. Training may have failed."
        )
    return runs[0].info.run_id


def _find_checkpoint_artifact(run_id: str) -> str:
    """Find the best checkpoint artifact path in the run."""
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id, path="checkpoints")
    if not artifacts:
        artifacts = client.list_artifacts(run_id)
        checkpoint_dirs = [a.path for a in artifacts if "checkpoint" in a.path]
        if not checkpoint_dirs:
            raise RuntimeError(
                f"No checkpoint artifacts found in MLflow run '{run_id}'. "
                "Ensure save_strategy is configured and MLflow artifact logging is active."
            )
        return checkpoint_dirs[-1]

    checkpoint_dirs = sorted(a.path for a in artifacts if a.is_dir)
    if not checkpoint_dirs:
        raise RuntimeError(f"No checkpoint directories in 'checkpoints/' for run '{run_id}'.")
    return checkpoint_dirs[-1]
