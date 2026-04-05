"""Write a deterministic training manifest and log canonical MLflow artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mlflow import MlflowClient
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger


class TrainingManifestCallback(Callback):
    """Persist training outputs and canonical checkpoint artifacts for one run."""

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

    def on_fit_start(self, trainer: Trainer, pl_module: Any) -> None:
        """Place checkpoints in a run-specific directory before training starts."""
        del pl_module
        mlflow_logger = _get_mlflow_logger(trainer)
        checkpoint_callback = _get_checkpoint_callback(trainer)

        base_dir = Path(checkpoint_callback.dirpath or "checkpoints")
        if not base_dir.is_absolute():
            base_dir = (Path(trainer.default_root_dir or ".") / base_dir).resolve()

        run_dir = base_dir / mlflow_logger.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback.dirpath = str(run_dir)

    def on_fit_end(self, trainer: Trainer, pl_module: Any) -> None:
        """Write the manifest and upload checkpoint artifacts to MLflow."""
        del pl_module
        mlflow_logger = _get_mlflow_logger(trainer)
        checkpoint_callback = _get_checkpoint_callback(trainer)

        manifest_path = Path(self.manifest_path).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = Path(checkpoint_callback.dirpath).resolve()
        manifest = _build_manifest(
            trainer=trainer,
            logger=mlflow_logger,
            checkpoint_callback=checkpoint_callback,
            checkpoint_dir=checkpoint_dir,
            checkpoint_artifact_path=self.checkpoint_artifact_path,
            manifest_artifact_path=self.manifest_artifact_path,
            manifest_filename=manifest_path.name,
        )

        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        client = MlflowClient(tracking_uri=getattr(mlflow_logger, "_tracking_uri", None))
        if checkpoint_dir.exists():
            client.log_artifacts(
                run_id=mlflow_logger.run_id,
                local_dir=str(checkpoint_dir),
                artifact_path=self.checkpoint_artifact_path,
            )
        client.log_artifact(
            run_id=mlflow_logger.run_id,
            local_path=str(manifest_path),
            artifact_path=self.manifest_artifact_path,
        )


def _build_manifest(
    trainer: Trainer,
    logger: MLFlowLogger,
    checkpoint_callback: ModelCheckpoint,
    checkpoint_dir: Path,
    checkpoint_artifact_path: str,
    manifest_artifact_path: str,
    manifest_filename: str,
) -> dict[str, Any]:
    """Assemble a machine-readable manifest for downstream steps."""
    best_checkpoint_path = _normalize_optional_path(checkpoint_callback.best_model_path)
    last_checkpoint_path = _normalize_optional_path(checkpoint_callback.last_model_path)

    return {
        "run_id": logger.run_id,
        "experiment_id": logger.experiment_id,
        "experiment_name": getattr(logger, "_experiment_name", None),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_artifact_path": checkpoint_artifact_path,
        "best_checkpoint_path": best_checkpoint_path,
        "best_checkpoint_artifact_path": _artifact_path_for_file(
            checkpoint_artifact_path, best_checkpoint_path
        ),
        "last_checkpoint_path": last_checkpoint_path,
        "last_checkpoint_artifact_path": _artifact_path_for_file(
            checkpoint_artifact_path, last_checkpoint_path
        ),
        "manifest_artifact_path": _join_artifact_path(
            manifest_artifact_path, manifest_filename
        ),
        "metrics": _serialize_metrics(trainer.callback_metrics),
    }


def _get_mlflow_logger(trainer: Trainer) -> MLFlowLogger:
    """Return the active MLflow logger or fail loudly."""
    for logger in trainer.loggers:
        if isinstance(logger, MLFlowLogger):
            return logger
    raise RuntimeError("TrainingManifestCallback requires an MLFlowLogger to be configured.")


def _get_checkpoint_callback(trainer: Trainer) -> ModelCheckpoint:
    """Return the single checkpoint callback used for canonical artifacts."""
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    raise RuntimeError("TrainingManifestCallback requires a ModelCheckpoint callback.")


def _serialize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert callback metrics to JSON-serializable Python values."""
    serialized: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def _normalize_optional_path(path: str | None) -> str | None:
    """Resolve a possibly empty path to an absolute path."""
    if not path:
        return None
    return str(Path(path).resolve())


def _artifact_path_for_file(artifact_root: str, local_path: str | None) -> str | None:
    """Map a local file path into the configured MLflow artifact namespace."""
    if local_path is None:
        return None
    return _join_artifact_path(artifact_root, Path(local_path).name)


def _join_artifact_path(prefix: str | None, suffix: str) -> str:
    """Join MLflow artifact path components without filesystem semantics."""
    if not prefix:
        return suffix
    return f"{prefix.rstrip('/')}/{suffix.lstrip('/')}"


