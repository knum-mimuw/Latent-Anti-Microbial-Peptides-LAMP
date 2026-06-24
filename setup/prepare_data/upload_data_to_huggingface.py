import os
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

import typer
from datasets import Dataset as HFDataset, DatasetDict, Features, Value
from huggingface_hub import HfApi
from pydantic import BaseModel, Extra, Field


class UploadConfig(BaseModel):
    """Configuration for data upload."""

    repo_id: str = Field(..., description="Hugging Face dataset repository ID")
    commit_message: str = Field(..., description="Commit message for the upload")
    token: str | None = Field(
        None, description="Hugging Face token (uses HF_TOKEN env var if None)"
    )
    columns: dict[str, str] = Field(..., description="Column mapping for the dataset")
    # Subset configuration
    subset_name: str = Field(..., description="Subset name for subset upload")

    class Config:
        extra = Extra.allow  # Allow future extensions without breaking


def upload_dataset_splits_to_huggingface(
    dsd: DatasetDict,
    *,
    repo_id: str,
    commit_message: str,
    token: str | None = None,
) -> None:
    """Upload a DatasetDict (multiple splits) to a Hugging Face dataset repository.

    This is useful for prepared split datasets (e.g., train/validation) produced by
    preparation scripts, decoupling preparation from upload.
    """
    if token is None:
        token_env = os.getenv("HF_TOKEN")
        if token_env is None:
            typer.echo(
                "❌ Hugging Face token not provided. Set HF_TOKEN or pass --token.",
                err=True,
            )
            raise typer.Exit(1)
        token = token_env

    try:
        dsd.push_to_hub(repo_id=repo_id, token=token, commit_message=commit_message)
    except Exception as e:
        typer.echo(f"❌ Upload failed: {e}", err=True)
        raise typer.Exit(1) from e


def _generator_from_stream(items: list[dict[str, Any]], columns: dict[str, str]):
    """Yield rows mapped according to columns spec from a list of items."""
    for item in items:
        row: dict[str, Any] = {}
        for col, source_field in columns.items():
            row[col] = item.get(source_field)
        yield row


def _value_feature_from_python(value: Any) -> Value:
    """Infer a HF Value feature from a Python value."""
    if isinstance(value, bool):
        return Value("bool")
    if isinstance(value, Integral):
        return Value("int64")
    if isinstance(value, Real):
        return Value("float64")
    return Value("string")


def _check_repository_exists(repo_id: str, token: str | None = None) -> bool:
    """Check if a Hugging Face dataset repository exists."""
    try:
        api = HfApi(token=token)
        api.repo_info(repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def build_datasetdict_from_streams(
    streams: Mapping[str, Iterable[dict[str, Any]]],
    *,
    columns: dict[str, str],
    features: Features | None = None,
) -> DatasetDict:
    """Build a DatasetDict from split-name -> item stream using Dataset.from_generator."""
    out: dict[str, HFDataset] = {}

    for split, items_iter in streams.items():
        # Materialize the generator to a list (picklable)
        items_list = list(items_iter)
        if not items_list:
            raise ValueError(f"Split '{split}' has no rows to upload.")

        inferred_features = features
        if inferred_features is None:
            first_item = items_list[0]
            mapped_first_row = {
                col: first_item.get(source_field) for col, source_field in columns.items()
            }
            inferred_features = Features(
                {
                    column: _value_feature_from_python(value)
                    for column, value in mapped_first_row.items()
                }
            )

        ds = HFDataset.from_generator(
            _generator_from_stream,
            gen_kwargs={"items": items_list, "columns": columns},
            features=inferred_features,
            keep_in_memory=False,  # Enable streaming mode
        )
        out[split] = ds
    return DatasetDict(out)


def upload_streams_to_huggingface(
    streams: Mapping[str, Iterable[dict[str, Any]]],
    *,
    repo_id: str,
    columns: dict[str, str],
    commit_message: str,
    token: str | None = None,
    features: Features | None = None,
) -> None:
    """Upload split streams to HF using Dataset.from_generator per split (low memory)."""
    dsd = build_datasetdict_from_streams(streams, columns=columns, features=features)
    upload_dataset_splits_to_huggingface(
        dsd, repo_id=repo_id, commit_message=commit_message, token=token
    )


def upload_streams_to_huggingface_subset(
    streams: Mapping[str, Iterable[dict[str, Any]]],
    *,
    repo_id: str,
    subset_name: str,
    columns: dict[str, str],
    commit_message: str,
    token: str | None = None,
    features: Features | None = None,
) -> None:
    """Upload split streams as a subset under a main dataset repository.

    This uploads to the main repo with subset-specific split names.
    """
    if token is None:
        token_env = os.getenv("HF_TOKEN")
        if token_env is None:
            typer.echo(
                "❌ Hugging Face token not provided. Set HF_TOKEN or pass --token.",
                err=True,
            )
            raise typer.Exit(1)
        token = token_env

    # Check if repository exists before attempting upload
    if not _check_repository_exists(repo_id, token):
        typer.echo(
            f"❌ Repository '{repo_id}' does not exist. "
            f"Please create it first using the 'create_huggingface_dataset_repo' command.",
            err=True,
        )
        raise typer.Exit(1)

    # Build the dataset with original split names (no subset prefix needed)
    dsd = build_datasetdict_from_streams(streams, columns=columns, features=features)

    try:
        dsd.push_to_hub(
            repo_id=repo_id,
            config_name=subset_name,
            token=token,
            commit_message=commit_message,
        )
    except Exception as e:
        typer.echo(f"❌ Upload failed: {e}", err=True)
        raise typer.Exit(1) from e
