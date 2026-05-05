"""Composable ZenML pipeline for training and optional HF publishing."""

from __future__ import annotations

import argparse

from zenml import pipeline

from ..utils.pipeline_utils import (
    configured_model_target,
    load_run_config,
    resolve_model_card_title,
    resolve_private,
    resolve_repo_id,
    resolve_revision,
    resolve_tag,
)
from ..publish.steps.publish_hf_step import publish_hf
from .steps.train_step import train


@pipeline
def train_and_optional_publish_pipeline(
    config_paths: list[str],
    run_config_path: str,
    upload_to_hf: bool = False,
    repo_id: str | None = None,
    revision: str | None = None,
    tag: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
    model_card_title: str | None = None,
):
    """Train a model and optionally publish the canonical MLflow checkpoint."""
    run_config = load_run_config(run_config_path)
    model_name, model_version = configured_model_target(run_config)
    (
        checkpoint_artifact_path,
        mlflow_run_id,
        experiment_name,
    ) = train(config_paths, run_config_path)

    if upload_to_hf:
        publish_hf(
            run_id=mlflow_run_id,
            artifact_path=checkpoint_artifact_path,
            repo_id=resolve_repo_id(repo_id, run_config),
            revision=resolve_revision(revision, run_config),
            tag=resolve_tag(tag, run_config),
            private=resolve_private(private, run_config),
            commit_message=commit_message,
            token=token,
            model_card_title=resolve_model_card_title(model_card_title, run_config),
            experiment_name=experiment_name or run_config.mlflow.experiment_name,
            model_name=model_name,
            model_version=model_version,
            source_stage="training",
        )


def main() -> None:
    """CLI entry point for the composed train-then-publish pipeline."""
    parser = argparse.ArgumentParser(
        description="Train with ZenML and optionally publish the best checkpoint to Hugging Face"
    )
    parser.add_argument(
        "config_paths",
        nargs="+",
        help="One or more config files passed through to `modelling fit`",
    )
    parser.add_argument(
        "--run-config",
        required=True,
        help="Per-run YAML config for MLflow / HF / ZenML identities",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Publish the best checkpoint to Hugging Face after training",
    )
    parser.add_argument("--repo-id", default=None, help="Target HF repo ID")
    parser.add_argument("--revision", default=None, help="Optional HF branch or revision")
    parser.add_argument("--tag", default=None, help="Optional immutable HF tag to create")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--commit-message", default=None, help="Optional HF commit message")
    parser.add_argument("--token", default=None, help="Optional HF token override")
    parser.add_argument("--model-card-title", default=None, help="Optional model card title")
    args = parser.parse_args()

    train_and_optional_publish_pipeline(
        config_paths=args.config_paths,
        run_config_path=args.run_config,
        upload_to_hf=args.upload_to_hf,
        repo_id=args.repo_id,
        revision=args.revision,
        tag=args.tag,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
        model_card_title=args.model_card_title,
    )


if __name__ == "__main__":
    main()
