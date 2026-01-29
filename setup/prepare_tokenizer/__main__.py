#!/usr/bin/env python3
import shutil
from pathlib import Path
from typing import Optional

import typer

from setup.prepare_data.utils import load_config_file

from .build_tokenizer import (
    DEFAULT_OUTPUT_DIR,
    BuildTokenizerConfig,
    build_tokenizer,
    build_tokenizer_command,
)
from .upload_tokenizer import (
    UploadTokenizerConfig,
    upload_tokenizer,
    upload_tokenizer_command,
)

app = typer.Typer(
    name="lamp-tokenizer-prep",
    help=(
        "Build and upload protein amino-acid tokenizers to Hugging Face. "
        "Creates a fast Rust-backed tokenizer with 1 token = 1 amino acid."
    ),
    no_args_is_help=True,
)

app.command("build")(build_tokenizer_command)
app.command("upload")(upload_tokenizer_command)


@app.command("run")
def run_command(
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
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Output directory for tokenizer files",
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
    skip_build: bool = typer.Option(
        False, "--skip-build", help="Skip building (use existing directory)"
    ),
    skip_upload: bool = typer.Option(
        False, "--skip-upload", help="Skip uploading to HF Hub"
    ),
) -> None:
    """Build, upload, and optionally delete tokenizer in one command."""
    hub_readme: Optional[Path] = None

    if config:
        raw = load_config_file(config)
        build_cfg = BuildTokenizerConfig(**raw.get("build", {}))
        output_dir = build_cfg.output_dir

        upload_cfg = raw.get("upload", {})
        if upload_cfg:
            upload_config = UploadTokenizerConfig(**upload_cfg)
            repo_id = repo_id or upload_config.repo_id
            private = private or upload_config.private
            commit_message = (
                upload_config.commit_message
                if commit_message == "Upload tokenizer"
                else commit_message
            )
            token = token or upload_config.token
            delete_after = delete_after or upload_config.delete_after

            if upload_config.hub_readme:
                hub_readme = Path(upload_config.hub_readme)

    # Build
    if not skip_build:
        typer.echo("🔨 Building tokenizer...")
        tokenizer = build_tokenizer(output_dir)

        # Sanity check
        out = tokenizer(
            ["MKTLLILAVAVCSAA", "ACDEFGHIK"],
            padding=True,
            return_tensors="pt",
        )
        typer.echo("Sanity check:")
        typer.echo(str(out))
    else:
        typer.echo(f"⏭️  Skipping build (using existing: {output_dir})")

    # Upload
    if not skip_upload:
        if not repo_id:
            typer.echo(
                "❌ --repo-id is required for upload (or set in config)", err=True
            )
            raise typer.Exit(1)

        typer.echo("📤 Uploading tokenizer...")
        upload_tokenizer(
            tokenizer_dir=output_dir,
            repo_id=repo_id,
            private=private,
            commit_message=commit_message,
            token=token,
            hub_readme=hub_readme,
            delete_after=delete_after,
        )
    else:
        typer.echo("⏭️  Skipping upload")

        # Handle delete_after even if skipping upload
        if delete_after:
            shutil.rmtree(output_dir)
            typer.echo(f"✓ Deleted local directory: {output_dir}")

    typer.echo("✅ Done!")


if __name__ == "__main__":
    app()
