"""Training step — Hugging Face Trainer + Hydra via ``uv run modelling``."""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

from zenml import step

from modelling.src.training.hydra_overrides import flatten_yaml_file

from ...utils.pipeline_utils import lamp_repo_root, load_run_config


class TrainResult(NamedTuple):
    """Structured training outputs for downstream pipeline composition."""

    checkpoint_path: str
    checkpoint_artifact_path: str
    manifest_artifact_path: str
    mlflow_run_id: str
    experiment_name: str | None


@step(enable_cache=False)
def train(config_paths: list[str], run_config_path: str) -> TrainResult:
    """Run modelling training and return manifest-derived outputs."""
    repo_root = lamp_repo_root()
    run_config = load_run_config(run_config_path)
    with tempfile.TemporaryDirectory(prefix="lamp-train-manifest-") as temp_dir:
        manifest_path = Path(temp_dir) / "training_manifest.json"
        env = _build_training_env(manifest_path, run_config.mlflow.experiment_name)

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

        manifest = _read_manifest(manifest_path)

    return TrainResult(
        checkpoint_path=manifest["best_checkpoint_path"],
        checkpoint_artifact_path=manifest["best_checkpoint_artifact_path"],
        manifest_artifact_path=manifest["manifest_artifact_path"],
        mlflow_run_id=manifest["run_id"],
        experiment_name=manifest.get("experiment_name"),
    )


def _build_training_env(manifest_path: Path, experiment_name: str) -> dict[str, str]:
    env = dict(os.environ)
    env["TRAINING_MANIFEST_PATH"] = str(manifest_path)
    env["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    env.setdefault("MLFLOW_CHECKPOINT_ARTIFACT_PATH", "checkpoints")
    env.setdefault("MLFLOW_MANIFEST_ARTIFACT_PATH", "metadata")
    return env


def _read_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        raise RuntimeError(
            "Training did not produce a manifest. Ensure MLflow reporting and ManifestCallback are active."
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
