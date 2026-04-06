"""Publish an MLflow-managed checkpoint to Hugging Face via ZenML."""

from __future__ import annotations

import argparse

from ._zenml_sqlalchemy_uuid_compat import apply as _apply_zenml_uuid_compat

_apply_zenml_uuid_compat()

from zenml import pipeline

from ._pipeline_utils import (
    configured_model_target,
    load_run_config,
    resolve_model_card_title,
    resolve_private,
    resolve_repo_id,
    resolve_revision,
    resolve_tag,
)
from .steps.publish_hf_step import publish_hf


@pipeline
def publish_hf_pipeline(
    run_config_path: str,
    repo_id: str | None = None,
    run_id: str | None = None,
    artifact_path: str | None = None,
    manifest_path: str | None = None,
    manifest_artifact_path: str | None = None,
    revision: str | None = None,
    tag: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
    model_card_title: str | None = None,
    experiment_name: str | None = None,
    source_stage: str = "publish",
):
    """Publish any MLflow-managed checkpoint as a reusable standalone pipeline."""
    run_config = load_run_config(run_config_path)
    model_name, model_version = configured_model_target(run_config)
    publish_hf(
        repo_id=resolve_repo_id(repo_id, run_config),
        run_id=run_id,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        manifest_artifact_path=manifest_artifact_path,
        revision=resolve_revision(revision, run_config),
        tag=resolve_tag(tag, run_config),
        private=resolve_private(private, run_config),
        commit_message=commit_message,
        token=token,
        model_card_title=resolve_model_card_title(model_card_title, run_config),
        experiment_name=experiment_name or run_config.mlflow.experiment_name,
        model_name=model_name,
        model_version=model_version,
        source_stage=source_stage,
    )


def main() -> None:
    """CLI entry point for the standalone HF publish pipeline."""
    parser = argparse.ArgumentParser(description="Publish an MLflow checkpoint to Hugging Face via ZenML")
    parser.add_argument("run_config_path", help="Per-run YAML config for MLflow / HF / ZenML identities")
    parser.add_argument("--repo-id", default=None, help="Target HF repo ID")
    parser.add_argument("--run-id", default=None, help="MLflow run ID that owns the checkpoint")
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="Artifact path inside the MLflow run, e.g. checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Local training manifest path containing run_id and checkpoint artifact coordinates",
    )
    parser.add_argument(
        "--manifest-artifact-path",
        default=None,
        help="Optional MLflow artifact path for the training manifest",
    )
    parser.add_argument("--revision", default=None, help="Optional HF branch or revision")
    parser.add_argument("--tag", default=None, help="Optional immutable HF tag to create")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--commit-message", default=None, help="Optional HF commit message")
    parser.add_argument("--token", default=None, help="Optional HF token override")
    parser.add_argument("--model-card-title", default=None, help="Optional model card title")
    parser.add_argument("--experiment-name", default=None, help="Optional MLflow experiment name override")
    parser.add_argument(
        "--source-stage",
        default="publish",
        help="Provenance label stored with the uploaded metadata",
    )
    args = parser.parse_args()

    publish_hf_pipeline(
        run_config_path=args.run_config_path,
        repo_id=args.repo_id,
        run_id=args.run_id,
        artifact_path=args.artifact_path,
        manifest_path=args.manifest_path,
        manifest_artifact_path=args.manifest_artifact_path,
        revision=args.revision,
        tag=args.tag,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
        model_card_title=args.model_card_title,
        experiment_name=args.experiment_name,
        source_stage=args.source_stage,
    )


if __name__ == "__main__":
    main()
