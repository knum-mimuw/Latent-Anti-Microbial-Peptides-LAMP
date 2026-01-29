# prepare_tokenizer/upload_tokenizer.py

import os
import shutil
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import HfApi
from pydantic import BaseModel, Field

from setup.prepare_data.utils import load_config_file

from .build_tokenizer import DEFAULT_OUTPUT_DIR


class UploadTokenizerConfig(BaseModel):
    """Configuration for uploading the tokenizer to Hugging Face Hub."""

    repo_id: str = Field(..., description="Hugging Face repo ID (e.g., pszmk/protein-aa-fast-tokenizer)")
    private: bool = Field(False, description="Whether the repo should be private")
    commit_message: str = Field("Upload tokenizer", description="Commit message for the upload")
    token: Optional[str] = Field(None, description="HF token; falls back to HF_TOKEN env")
    hub_readme: Optional[str] = Field(None, description="Path to README to upload to the hub")
    delete_after: bool = Field(False, description="Delete local tokenizer directory after upload")


def upload_tokenizer(
    tokenizer_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload tokenizer",
    token: Optional[str] = None,
    hub_readme: Optional[Path] = None,
    delete_after: bool = False,
) -> None:
    """Upload tokenizer directory to Hugging Face Hub."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        typer.echo("❌ Hugging Face token not provided. Set HF_TOKEN or pass --token.", err=True)
        raise typer.Exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    typer.echo(f"✓ Repository ready: {repo_id}")

    # Copy hub README to tokenizer directory if provided
    if hub_readme and hub_readme.exists():
        readme_dest = Path(tokenizer_dir) / "README.md"
        shutil.copy(hub_readme, readme_dest)
        typer.echo(f"✓ Copied {hub_readme} → {readme_dest}")

    # Upload the folder
    api.upload_folder(
        folder_path=tokenizer_dir,
        repo_id=repo_id,
        commit_message=commit_message,
    )

    typer.echo(f"✓ Tokenizer uploaded to https://huggingface.co/{repo_id}")

    # Delete local directory if requested
    if delete_after:
        shutil.rmtree(tokenizer_dir)
        typer.echo(f"✓ Deleted local directory: {tokenizer_dir}")


def upload_tokenizer_command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    tokenizer_dir: str = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--tokenizer-dir",
        "-d",
        help="Tokenizer directory to upload",
    ),
    repo_id: Optional[str] = typer.Option(
        None,
        "--repo-id",
        "-r",
        help="Hugging Face repo ID (e.g., pszmk/protein-aa-fast-tokenizer)",
    ),
    private: bool = typer.Option(False, "--private", help="Make repo private"),
    commit_message: str = typer.Option(
        "Upload tokenizer",
        "--commit-message",
        "-m",
        help="Commit message",
    ),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="HF API token"),
    delete_after: bool = typer.Option(
        False,
        "--delete-after",
        help="Delete local tokenizer directory after upload",
    ),
) -> None:
    """Upload tokenizer to Hugging Face Hub."""
    hub_readme: Optional[Path] = None

    if config:
        raw = load_config_file(config)
        build_cfg = raw.get("build", {})
        upload_cfg = raw.get("upload", {})

        tokenizer_dir = build_cfg.get("output_dir", tokenizer_dir)

        upload_config = UploadTokenizerConfig(**upload_cfg)
        repo_id = upload_config.repo_id
        private = upload_config.private
        commit_message = upload_config.commit_message
        token = upload_config.token or token
        # CLI flag overrides config
        delete_after = delete_after or upload_config.delete_after

        if upload_config.hub_readme:
            hub_readme = Path(upload_config.hub_readme)

    if not repo_id:
        typer.echo("❌ --repo-id is required (or set in config)", err=True)
        raise typer.Exit(1)

    upload_tokenizer(
        tokenizer_dir=tokenizer_dir,
        repo_id=repo_id,
        private=private,
        commit_message=commit_message,
        token=token,
        hub_readme=hub_readme,
        delete_after=delete_after,
    )


if __name__ == "__main__":
    upload_tokenizer_command()
