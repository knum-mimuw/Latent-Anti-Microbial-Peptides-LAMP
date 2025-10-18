from typing import Generator, Dict, Any, Optional, List, Iterable
import os
from pathlib import Path
import typer
from datasets import load_dataset
from tqdm import tqdm
from pydantic import BaseModel, Field, Extra

from .utils import SequenceItem, load_config_file
from .upload_data_to_huggingface import (
    upload_streams_to_huggingface_subset,
    UploadConfig,
)


class PrepareESM2UniRefConfig(BaseModel):
    """Configuration for ESM2 UniRef data preparation."""

    dataset_name: str = Field(
        "nvidia/esm2_uniref_pretraining_data",
        description="HF dataset repo for ESM-2 UniRef pretraining data",
    )
    # Per-call settings for the single-split generator
    split: str = Field("train", description="Dataset split to stream from")
    max_length: int = Field(50, description="Keep sequences with length <= this value")
    max_sequences: Optional[int] = Field(
        None, description="Optional cap on number of yielded sequences"
    )
    splits: List[str] = Field(
        default_factory=lambda: ["train", "validation"],
        description="List of dataset splits to process (e.g., train, validation)",
    )

    class Config:
        extra = Extra.allow  # Allow future extensions without breaking


def create_sequence_item(item: Dict[str, Any]) -> SequenceItem:
    """Create a standardized sequence item from ESM-2 dataset item."""
    return {
        "ur50_id": item["ur50_id"],
        "ur90_id": item["ur90_id"],
        "sequence": item["sequence"],
        "length": len(item["sequence"]),  # Keep as integer for int32 schema
    }


def _stream_split(
    *,
    dataset_name: str,
    split: str,
    max_length: int,
    max_sequences: Optional[int],
) -> Generator[SequenceItem, None, None]:
    """Stream one split and yield standardized items with length filter."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    count = 0
    for item in tqdm(dataset, desc=f"Processing {split} sequences"):
        sequence = item.get("sequence")
        if not sequence:
            continue
        if len(sequence) <= max_length:
            yield create_sequence_item(item)
            count += 1
            if max_sequences and count >= max_sequences:
                break


def prepare_esm2_short_sequences_for_splits(
    cfg: PrepareESM2UniRefConfig,
) -> Dict[str, Generator[SequenceItem, None, None]]:
    """Prepare generators for multiple splits using only the config.

    Returns a dict of split name -> generator of standardized items.
    """
    if not cfg.splits:
        raise ValueError("splits list cannot be empty")
    return {
        split_name: prepare_esm2_short_sequences(
            cfg=cfg.model_copy(update={"split": split_name})
        )
        for split_name in cfg.splits
    }


def prepare_esm2_short_sequences(
    cfg: PrepareESM2UniRefConfig,
) -> Generator[SequenceItem, None, None]:
    """Prepare short sequences from a single split using only the config.

    Returns a generator of standardized items.
    """
    return _stream_split(
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        max_length=cfg.max_length,
        max_sequences=cfg.max_sequences,
    )


def prepare_and_upload_esm2_uniref_command(
    *,
    config_file: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help=(
            "YAML/JSON config with separate 'prepare' and 'upload' sections. "
            "Prepare section: dataset_name, splits, max_length, max_sequences. "
            "Upload section: repo_id, commit_message, columns, token"
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Prepare <=N AA sequences for configured splits and upload to HF as DatasetDict."""
    # Load required config (fail fast if missing or invalid)
    raw = load_config_file(config_file)

    # Extract and validate preparation config
    prepare_raw = raw.get("prepare", {})
    if not prepare_raw:
        typer.echo("❌ Config must include 'prepare' section.", err=True)
        raise typer.Exit(1)
    prepare_cfg = PrepareESM2UniRefConfig(**prepare_raw)

    # Extract and validate upload config
    upload_raw = raw.get("upload", {})
    if not upload_raw:
        typer.echo("❌ Config must include 'upload' section.", err=True)
        raise typer.Exit(1)

    # Set token from config or environment
    if not upload_raw.get("token"):
        upload_raw["token"] = os.getenv("HF_TOKEN")

    upload_cfg = UploadConfig(**upload_raw)

    typer.echo(
        f"🚀 Preparing ESM2 UniRef sequences (<= {prepare_cfg.max_length} AA) for splits {prepare_cfg.splits}"
    )
    # Build streaming generators per split (no full materialization)
    streams = prepare_esm2_short_sequences_for_splits(prepare_cfg)

    # Upload as subset (subset_name is always required)
    typer.echo(
        f"📤 Uploading as subset '{upload_cfg.subset_name}' under {upload_cfg.repo_id}..."
    )
    upload_streams_to_huggingface_subset(
        streams,
        repo_id=upload_cfg.repo_id,
        subset_name=upload_cfg.subset_name,
        columns=upload_cfg.columns,
        commit_message=upload_cfg.commit_message,
        token=upload_cfg.token,
    )
    typer.echo(
        f"📦 Uploaded subset to https://huggingface.co/datasets/{upload_cfg.repo_id}/{upload_cfg.subset_name}"
    )


def _demo():
    cfg = PrepareESM2UniRefConfig()
    streams = prepare_esm2_short_sequences_for_splits(
        cfg=cfg.model_copy(update={"max_length": 50, "max_sequences": 5})
    )
    print(f"✅ Prepared generators for splits: {list(streams.keys())}")
    print("\n📊 Sample from train (first 3 items):")
    count = 0
    for seq in streams["train"]:
        print(f"  {count+1}. {seq['ur50_id']} ({seq['length']} AA)")
        print(f"     {seq['sequence']}")
        count += 1
        if count >= 3:
            break


if __name__ == "__main__":
    _demo()
