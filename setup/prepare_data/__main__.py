#!/usr/bin/env python3
import typer
from .create_huggingface_dataset_repo import (
    create_huggingface_dataset_repo_command,
)
from .prepare_and_upload_esm2_uniref import prepare_and_upload_esm2_uniref_command

app = typer.Typer(
    name="lamp-data-prep",
    help=(
        "Hugging Face dataset utilities: create dataset repos and upload prepared data. "
        "All uploads require an explicit columns mapping; no schema inference."
    ),
    no_args_is_help=True,
)

app.command("create_huggingface_dataset_repo")(create_huggingface_dataset_repo_command)
app.command("prepare_and_upload_esm2_uniref")(prepare_and_upload_esm2_uniref_command)

if __name__ == "__main__":
    app()
