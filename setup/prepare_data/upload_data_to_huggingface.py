import os
from typing import Optional, List, Dict, Any, Iterable, Mapping
from datasets import Dataset as HFDataset, DatasetDict, Features, Value
import typer
from pydantic import BaseModel, Field, Extra
from huggingface_hub import HfApi


class UploadConfig(BaseModel):
    """Configuration for data upload."""

    repo_id: str = Field(..., description="Hugging Face dataset repository ID")
    commit_message: str = Field(..., description="Commit message for the upload")
    token: Optional[str] = Field(
        None, description="Hugging Face token (uses HF_TOKEN env var if None)"
    )
    columns: Dict[str, str] = Field(..., description="Column mapping for the dataset")
    # Subset configuration
    subset_name: str = Field(..., description="Subset name for subset upload")

    class Config:
        extra = Extra.allow  # Allow future extensions without breaking


def upload_dataset_splits_to_huggingface(
    dsd: DatasetDict,
    *,
    repo_id: str,
    commit_message: str,
    token: Optional[str] = None,
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
        raise typer.Exit(1)


def _generator_from_stream(items: List[Dict[str, Any]], columns: Dict[str, str]):
    """Yield rows mapped according to columns spec from a list of items."""
    for item in items:
        row: Dict[str, Any] = {}
        for col, source_field in columns.items():
            row[col] = item.get(source_field)
        yield row


def _check_repository_exists(repo_id: str, token: Optional[str] = None) -> bool:
    """Check if a Hugging Face dataset repository exists."""
    try:
        api = HfApi(token=token)
        api.repo_info(repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def build_datasetdict_from_streams(
    streams: Mapping[str, Iterable[Dict[str, Any]]],
    *,
    columns: Dict[str, str],
) -> DatasetDict:
    """Build a DatasetDict from split-name -> item stream using Dataset.from_generator."""
    out: Dict[str, HFDataset] = {}

    # Define features schema with proper data types
    features = Features(
        {
            "ur50_id": Value("string"),
            "ur90_id": Value("string"),
            "sequence": Value("string"),
            "length": Value("int32"),
        }
    )

    for split, items_iter in streams.items():
        # Materialize the generator to a list (picklable)
        items_list = list(items_iter)

        ds = HFDataset.from_generator(
            _generator_from_stream,
            gen_kwargs={"items": items_list, "columns": columns},
            features=features,
            keep_in_memory=False,  # Enable streaming mode
        )
        out[split] = ds
    return DatasetDict(out)


def upload_streams_to_huggingface(
    streams: Mapping[str, Iterable[Dict[str, Any]]],
    *,
    repo_id: str,
    columns: Dict[str, str],
    commit_message: str,
    token: Optional[str] = None,
) -> None:
    """Upload split streams to HF using Dataset.from_generator per split (low memory)."""
    dsd = build_datasetdict_from_streams(streams, columns=columns)
    upload_dataset_splits_to_huggingface(
        dsd, repo_id=repo_id, commit_message=commit_message, token=token
    )


def upload_streams_to_huggingface_subset(
    streams: Mapping[str, Iterable[Dict[str, Any]]],
    *,
    repo_id: str,
    subset_name: str,
    columns: Dict[str, str],
    commit_message: str,
    token: Optional[str] = None,
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
    dsd = build_datasetdict_from_streams(streams, columns=columns)

    try:
        dsd.push_to_hub(
            repo_id=repo_id,
            config_name=subset_name,
            token=token,
            commit_message=commit_message,
        )
    except Exception as e:
        typer.echo(f"❌ Upload failed: {e}", err=True)
        raise typer.Exit(1)
