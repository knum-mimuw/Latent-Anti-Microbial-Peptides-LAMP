"""Shared helpers for run-scoped pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def lamp_repo_root() -> Path:
    """Return the LAMP workspace git root (directory containing ``src/modelling``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "modelling" / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "Could not locate LAMP repository root (expected src/modelling/pyproject.toml)."
    )


@dataclass(frozen=True)
class MlflowRunConfig:
    """Run-scoped MLflow identity."""

    experiment_name: str


@dataclass(frozen=True)
class ZenmlRunConfig:
    """Optional ZenML model identity for metadata linkage."""

    model_name: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class HuggingFaceRunConfig:
    """Run-scoped HF release identity."""

    repo_id: str | None = None
    revision: str | None = None
    tag: str | None = None
    model_card_title: str | None = None
    private: bool = False


@dataclass(frozen=True)
class RunConfig:
    """Top-level run-scoped configuration loaded from YAML."""

    mlflow: MlflowRunConfig
    zenml: ZenmlRunConfig
    huggingface: HuggingFaceRunConfig


def load_run_config(run_config_path: str) -> RunConfig:
    """Load the per-run YAML config used by pipeline entrypoints."""
    config_path = Path(run_config_path)
    with config_path.open() as handle:
        payload = yaml.safe_load(handle) or {}

    mlflow_payload = payload.get("mlflow", {})
    experiment_name = mlflow_payload.get("experiment_name")
    if not experiment_name:
        raise ValueError(
            f"Run config {config_path} must define mlflow.experiment_name."
        )

    zenml_payload = payload.get("zenml", {})
    huggingface_payload = payload.get("huggingface", {})

    return RunConfig(
        mlflow=MlflowRunConfig(experiment_name=str(experiment_name)),
        zenml=ZenmlRunConfig(
            model_name=_optional_str(zenml_payload.get("model_name")),
            model_version=_optional_str(zenml_payload.get("model_version")),
        ),
        huggingface=HuggingFaceRunConfig(
            repo_id=_optional_str(huggingface_payload.get("repo_id")),
            revision=_optional_str(huggingface_payload.get("revision")),
            tag=_optional_str(huggingface_payload.get("tag")),
            model_card_title=_optional_str(huggingface_payload.get("model_card_title")),
            private=bool(huggingface_payload.get("private", False)),
        ),
    )


def resolve_repo_id(repo_id: str | None, run_config: RunConfig) -> str:
    """Resolve the target HF repo ID from explicit arguments or run config."""
    resolved = repo_id or run_config.huggingface.repo_id
    if not resolved:
        raise ValueError(
            "A Hugging Face repo ID is required. Pass repo_id explicitly or set huggingface.repo_id in the run config."
        )
    return resolved


def resolve_revision(revision: str | None, run_config: RunConfig) -> str | None:
    """Resolve the HF revision from explicit arguments or run config."""
    return revision or run_config.huggingface.revision


def resolve_tag(tag: str | None, run_config: RunConfig) -> str | None:
    """Resolve the HF tag from explicit arguments or run config."""
    return tag or run_config.huggingface.tag


def resolve_model_card_title(model_card_title: str | None, run_config: RunConfig) -> str | None:
    """Resolve the model card title from explicit arguments or run config."""
    return model_card_title or run_config.huggingface.model_card_title


def resolve_private(private: bool, run_config: RunConfig) -> bool:
    """Resolve the repo visibility flag from explicit arguments or run config."""
    return private or run_config.huggingface.private


def configured_model_target(run_config: RunConfig) -> tuple[str | None, str | None]:
    """Return the optional ZenML model name/version pair from run config."""
    return run_config.zenml.model_name, run_config.zenml.model_version


def _optional_str(value: object) -> str | None:
    """Normalize optional string values from YAML payloads."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
