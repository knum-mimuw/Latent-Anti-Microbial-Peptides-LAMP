import os
from pathlib import Path
from typing import Optional, Any, Dict
from huggingface_hub import HfApi, create_repo
import typer
from pydantic import BaseModel, Field
from .utils import load_config_file


class CreateHuggingFaceDatasetRepoConfig(BaseModel):
    repo_name: str = Field(..., description="Dataset repository name")
    token: Optional[str] = Field(
        None, description="HF token; falls back to HF_TOKEN env"
    )
    private: bool = Field(False, description="Whether the repo should be private")
    organization: Optional[str] = Field(None, description="Organization name, if any")


def create_huggingface_dataset_repo(
    config: CreateHuggingFaceDatasetRepoConfig,
) -> str:
    """Create or ensure a dataset-type repository exists on Hugging Face Hub.

    Returns full repo id (e.g., "username/repo" or "org/repo").
    """
    token = config.token
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Hugging Face token not provided. Set HF_TOKEN or pass --token."
            )

    api = HfApi(token=token)

    if config.organization:
        repo_id = f"{config.organization}/{config.repo_name}"
    else:
        who = api.whoami()
        repo_id = f"{who['name']}/{config.repo_name}"

    create_repo(
        repo_id=repo_id,
        token=token,
        private=config.private,
        repo_type="dataset",
        exist_ok=True,
    )
    return repo_id


def create_huggingface_dataset_repo_command(
    config_file: Path = typer.Option(
        ...,  # required
        "--config",
        "-c",
        help="Path to YAML/JSON config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Typer command: create HF dataset repo from a YAML/JSON config file."""
    if not config_file.exists():
        typer.echo(f"❌ Config not found: {config_file}", err=True)
        raise typer.Exit(1)
    try:
        raw = load_config_file(config_file)
        cfg = CreateHuggingFaceDatasetRepoConfig(**raw)
        repo_id = create_huggingface_dataset_repo(cfg)
        typer.echo(f"✅ Dataset repository ready: {repo_id}")
    except Exception as e:
        typer.echo(f"❌ Failed to create dataset repo: {e}", err=True)
        raise typer.Exit(1)
