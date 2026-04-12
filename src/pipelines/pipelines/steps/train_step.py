"""Training step -- thin wrapper around the standalone Lightning CLI."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import yaml
from zenml import step

from .._pipeline_utils import lamp_repo_root, load_run_config


class TrainResult(NamedTuple):
    """Structured training outputs for downstream pipeline composition."""

    checkpoint_path: str
    checkpoint_artifact_path: str
    manifest_artifact_path: str
    mlflow_run_id: str
    experiment_name: str | None


@step(enable_cache=False)
def train(config_paths: list[str], run_config_path: str) -> TrainResult:
    """Run Lightning training via CLI and return manifest-derived outputs.

    This executes the same command you would run manually (explicit ``--config``
    paths for trainer, model, data, logger, callbacks as needed)::

        uv run modelling fit --config <trainer.yaml> --config <model.yaml> ...

    Args:
        config_paths: One or more YAML config file paths to pass to
            ``modelling fit --config ...``.
        run_config_path: YAML file defining run-scoped MLflow / HF / ZenML identities.

    Returns:
        Structured checkpoint and MLflow coordinates written by training.
    """
    repo_root = lamp_repo_root()
    run_config = load_run_config(run_config_path)
    with tempfile.TemporaryDirectory(prefix="lamp-train-manifest-") as temp_dir:
        manifest_path = Path(temp_dir) / "training_manifest.json"
        env = _build_training_env(manifest_path)
        logger_override_path = Path(temp_dir) / "logger_override.yaml"
        _write_logger_override(
            path=logger_override_path,
            experiment_name=run_config.mlflow.experiment_name,
        )

        cmd: list[str] = ["uv", "run", "modelling", "fit"]
        for cfg in [*config_paths, str(logger_override_path)]:
            cmd.extend(["--config", cfg])

        subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=repo_root,
            env=env,
        )

        manifest = _read_manifest(manifest_path)

    return TrainResult(
        checkpoint_path=manifest["best_checkpoint_path"],
        checkpoint_artifact_path=manifest["best_checkpoint_artifact_path"],
        manifest_artifact_path=manifest["manifest_artifact_path"],
        mlflow_run_id=manifest["run_id"],
        experiment_name=manifest.get("experiment_name"),
    )


def _build_training_env(manifest_path: Path) -> dict[str, str]:
    """Build the explicit environment used by the training subprocess."""
    env = dict(os.environ)
    env["TRAINING_MANIFEST_PATH"] = str(manifest_path)
    env.setdefault("MLFLOW_CHECKPOINT_ARTIFACT_PATH", "checkpoints")
    env.setdefault("MLFLOW_MANIFEST_ARTIFACT_PATH", "metadata")
    return env


def _read_manifest(manifest_path: Path) -> dict[str, str]:
    """Load and validate the training manifest written by the subprocess."""
    if not manifest_path.exists():
        raise RuntimeError(
            "Training did not produce a manifest. Ensure the checkpoint callback config is active."
        )

    manifest = json.loads(manifest_path.read_text())
    required_fields = [
        "run_id",
        "best_checkpoint_path",
        "best_checkpoint_artifact_path",
        "manifest_artifact_path",
    ]
    missing = [field for field in required_fields if not manifest.get(field)]
    if missing:
        raise RuntimeError(f"Training manifest is missing required fields: {', '.join(missing)}")

    return manifest


def _write_logger_override(path: Path, experiment_name: str) -> None:
    """Write a run-scoped MLflow logger override config."""
    payload = {
        "trainer": {
            "logger": [
                {
                    "class_path": "pytorch_lightning.loggers.mlflow.MLFlowLogger",
                    "init_args": {
                        "experiment_name": experiment_name,
                        "log_model": False,
                    },
                }
            ]
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
