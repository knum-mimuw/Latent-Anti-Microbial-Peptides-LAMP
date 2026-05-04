"""Deterministic training manifest + MLflow artifact logging (HF Trainer)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from transformers import TrainerCallback, TrainerState, TrainingArguments


class ManifestCallback(TrainerCallback):
    """Write ``training_manifest.json`` and log checkpoints + manifest to MLflow."""

    def __init__(
        self,
        manifest_path: str | None = None,
        checkpoint_artifact_path: str | None = None,
        manifest_artifact_path: str | None = None,
    ) -> None:
        self.manifest_path = manifest_path or os.environ.get(
            "TRAINING_MANIFEST_PATH", "training_manifest.json"
        )
        self.checkpoint_artifact_path = checkpoint_artifact_path or os.environ.get(
            "MLFLOW_CHECKPOINT_ARTIFACT_PATH", "checkpoints"
        )
        self.manifest_artifact_path = manifest_artifact_path or os.environ.get(
            "MLFLOW_MANIFEST_ARTIFACT_PATH", "metadata"
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control,
        **kwargs,
    ):
        del kwargs
        if not state.is_world_process_zero:
            return control
        if mlflow.active_run() is None:
            raise RuntimeError(
                "ManifestCallback requires an active MLflow run "
                "(set TrainingArguments.report_to to include 'mlflow')."
            )
        if not args.load_best_model_at_end:
            raise RuntimeError(
                "ManifestCallback requires load_best_model_at_end=True "
                "so that best_model_checkpoint is populated."
            )
        return control

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control,
        **kwargs,
    ):
        del kwargs
        if not state.is_world_process_zero:
            return control

        active = mlflow.active_run()
        if active is None:
            return control

        run_id = active.info.run_id
        experiment_id = active.info.experiment_id
        experiment = mlflow.get_experiment(experiment_id)
        experiment_name = experiment.name if experiment is not None else None

        best = state.best_model_checkpoint
        if not best:
            raise RuntimeError(
                "No best_model_checkpoint on TrainerState; "
                "ensure evaluation ran and metric_for_best_model is configured."
            )

        best_path = Path(best).resolve()
        checkpoint_dir = best_path if best_path.is_dir() else best_path.parent

        manifest_path = Path(self.manifest_path).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_filename = manifest_path.name
        manifest = _build_manifest(
            run_id=run_id,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir,
            best_checkpoint_path=str(best_path),
            checkpoint_artifact_path=self.checkpoint_artifact_path,
            manifest_artifact_path=self.manifest_artifact_path,
            manifest_filename=manifest_filename,
            metrics=_metrics_from_state(state),
        )

        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        client = MlflowClient()
        client.log_artifacts(run_id, str(checkpoint_dir), self.checkpoint_artifact_path)
        client.log_artifact(run_id, str(manifest_path), self.manifest_artifact_path)

        return control


def _metrics_from_state(state: TrainerState) -> dict[str, Any]:
    """Flatten ``log_history`` into JSON-serializable scalars (last value wins)."""
    merged: dict[str, Any] = {}
    for row in state.log_history:
        for key, value in row.items():
            if key in ("step", "epoch"):
                continue
            if hasattr(value, "item"):
                merged[key] = value.item()
            elif isinstance(value, (float, int, str, bool)) or value is None:
                merged[key] = value
    return merged


def _join_artifact_path(prefix: str | None, suffix: str) -> str:
    if not prefix:
        return suffix
    return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"


def _artifact_path_for_checkpoint_dir(
    artifact_root: str,
    checkpoint_dir: Path,
    root_output: Path,
) -> str | None:
    """Relative artifact path for the logged checkpoint directory."""
    try:
        relative = checkpoint_dir.relative_to(root_output.resolve())
    except ValueError:
        relative = Path(checkpoint_dir.name)
    return _join_artifact_path(artifact_root, str(relative))


def _build_manifest(
    *,
    run_id: str,
    experiment_id: str,
    experiment_name: str | None,
    checkpoint_dir: Path,
    best_checkpoint_path: str,
    checkpoint_artifact_path: str,
    manifest_artifact_path: str,
    manifest_filename: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    root_output = checkpoint_dir.parent
    best_art = _artifact_path_for_checkpoint_dir(
        checkpoint_artifact_path, Path(best_checkpoint_path).resolve(), root_output
    )

    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_artifact_path": checkpoint_artifact_path,
        "best_checkpoint_path": best_checkpoint_path,
        "best_checkpoint_artifact_path": best_art,
        "last_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_artifact_path": best_art,
        "manifest_artifact_path": _join_artifact_path(manifest_artifact_path, manifest_filename),
        "metrics": metrics,
    }
