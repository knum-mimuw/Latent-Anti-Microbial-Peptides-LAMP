"""ZenML step for publishing MLflow-managed checkpoints to Hugging Face Hub."""

from pathlib import Path
from typing import Any

from zenml import get_step_context, log_metadata, step


@step
def publish_hf(
    repo_id: str,
    run_id: str,
    artifact_path: str,
    revision: str | None = None,
    tag: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
    model_card_title: str | None = None,
    experiment_name: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    source_stage: str = "publish",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a checkpoint identified by explicit MLflow coordinates."""
    from modelling.src.utils.export_to_hf import export_to_huggingface

    checkpoint = _download_checkpoint_from_mlflow(run_id=run_id, artifact_path=artifact_path)

    publish_metadata = _build_publish_metadata(
        checkpoint_path=checkpoint,
        repo_id=repo_id,
        artifact_path=artifact_path,
        revision=revision,
        tag=tag,
        mlflow_run_id=run_id,
        experiment_name=experiment_name,
        source_stage=source_stage,
        extra_metadata=extra_metadata,
    )
    result = export_to_huggingface(
        checkpoint_path=checkpoint,
        repo_id=repo_id,
        revision=revision,
        tag=tag,
        private=private,
        commit_message=commit_message,
        token=token,
        model_card_title=model_card_title,
        metadata=publish_metadata,
    )

    if model_name is not None:
        try:
            log_metadata(
                metadata={
                    "huggingface": {
                        "repo_id": result["repo_id"],
                        "url": result["hub_url"],
                        "revision": result["revision"],
                        "tag": result["tag"],
                    },
                    "publish": publish_metadata,
                },
                model_name=model_name,
                model_version=model_version,
            )
        except KeyError:
            print(
                f"ZenML model version '{model_version}' for '{model_name}' not found; "
                "skipping metadata logging."
            )

    return {**result, "mlflow_run_id": run_id, "source_stage": source_stage}


def _download_checkpoint_from_mlflow(run_id: str, artifact_path: str) -> Path:
    """Download the target checkpoint from MLflow."""
    from modelling.src.utils.mlflow_utils import download_artifact

    checkpoint = download_artifact(run_id=run_id, artifact_path=artifact_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"MLflow artifact did not produce a checkpoint file: {run_id}/{artifact_path}"
        )
    return checkpoint


def _build_publish_metadata(
    checkpoint_path: Path,
    repo_id: str,
    artifact_path: str,
    revision: str | None,
    tag: str | None,
    mlflow_run_id: str,
    experiment_name: str | None,
    source_stage: str,
    extra_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collect metadata that should travel with the HF export."""
    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "mlflow_artifact_path": artifact_path,
        "repo_id": repo_id,
        "source_stage": source_stage,
    }

    if experiment_name:
        metadata["experiment_name"] = experiment_name
    if revision:
        metadata["hf_revision"] = revision
        metadata["hf_load_revision"] = revision
    if tag:
        metadata["hf_tag"] = tag
        metadata["model_version"] = tag
        metadata["hf_load_revision"] = tag
    if mlflow_run_id:
        metadata["mlflow"] = _load_mlflow_summary(mlflow_run_id)

    zenml_run_id = _get_zenml_run_id()
    if zenml_run_id:
        metadata["zenml_run_id"] = zenml_run_id

    if extra_metadata:
        metadata.update(extra_metadata)

    return metadata


def _load_mlflow_summary(run_id: str) -> dict[str, Any]:
    """Fetch metrics and provenance from MLflow when available."""
    from modelling.src.utils.mlflow_utils import get_run_summary

    summary = get_run_summary(run_id)
    return {
        "run_id": summary["run_id"],
        "status": summary["status"],
        "experiment_id": summary["experiment_id"],
        "artifact_uri": summary["artifact_uri"],
        "metrics": summary["metrics"],
        "params": summary["params"],
        "tags": summary["tags"],
    }


def _get_zenml_run_id() -> str | None:
    """Best-effort extraction of the current ZenML run ID."""
    step_context = get_step_context()
    pipeline_run = getattr(step_context, "pipeline_run", None)
    if pipeline_run is None:
        return None

    run_id = getattr(pipeline_run, "id", None)
    return None if run_id is None else str(run_id)
