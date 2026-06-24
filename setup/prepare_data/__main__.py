#!/usr/bin/env python3
import typer

from .create_huggingface_dataset_repo import (
    create_huggingface_dataset_repo_command,
)
from .prepare_and_upload_apex_predictions import prepare_and_upload_apex_predictions_command
from .prepare_and_upload_dbaasp import prepare_and_upload_dbaasp_command
from .prepare_and_upload_dbamp import prepare_and_upload_dbamp_command
from .prepare_and_upload_esm2_uniref import prepare_and_upload_esm2_uniref_command
from .prepare_and_upload_physicochemical_properties import (
    prepare_and_upload_physicochemical_properties_command,
)
from .prepare_and_upload_strain_conditions import prepare_and_upload_strain_conditions_command

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
app.command("prepare_and_upload_dbaasp")(prepare_and_upload_dbaasp_command)
app.command("prepare_and_upload_dbamp")(prepare_and_upload_dbamp_command)
app.command("prepare_and_upload_apex_predictions")(prepare_and_upload_apex_predictions_command)
app.command("prepare_and_upload_strain_conditions")(prepare_and_upload_strain_conditions_command)
app.command("prepare_and_upload_physicochemical_properties")(
    prepare_and_upload_physicochemical_properties_command
)

if __name__ == "__main__":
    app()
