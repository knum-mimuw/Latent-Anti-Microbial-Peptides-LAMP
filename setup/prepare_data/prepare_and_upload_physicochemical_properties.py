import os
from pathlib import Path
from typing import Any

import typer
from datasets import Features, Value, load_dataset
from pydantic import BaseModel, Extra, Field
from tqdm import tqdm

from .physicochemical_core import SUPPORTED_PROPERTIES, compute_properties
from .upload_data_to_huggingface import upload_streams_to_huggingface_subset
from .utils import load_config_file

CONFIG_FILE_OPTION = typer.Option(
    ...,
    "--config",
    "-c",
    help="YAML/JSON config with 'prepare' and 'upload' sections.",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)

DEFAULT_PROPERTIES = [
    "charge",
    "hydrophobicity",
    "isoelectric_point",
    "molecular_weight",
    "instability_index",
    "gravy",
    "aliphatic_index",
    "boman_index",
]


class PreparePhysicochemicalConfig(BaseModel):
    """Configuration for physicochemical sidecar feature generation."""

    dataset_name: str = Field(..., description="HF dataset repository ID with peptide sequences.")
    dataset_config: str | None = Field(None, description="Optional HF config name.")
    split: str = Field("train", description="Input split to process.")
    peptide_id_column: str = Field("peptide_id", description="Primary key column.")
    sequence_column: str = Field("sequence", description="Peptide sequence column.")
    batch_size: int = Field(50000, description="Chunk size for processing.")
    use_tqdm: bool = Field(False, description="Show progress bar over sequence chunks.")
    ph: float = Field(7.0, description="pH value used for charge calculation.")
    hydrophobicity_scale: str = Field(
        "Aboderin",
        description="Scale name for peptides.Peptide.hydrophobicity().",
    )
    properties: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PROPERTIES),
        description="Subset of physicochemical properties to compute.",
    )

    class Config:
        extra = Extra.allow


class PhysicochemicalUploadConfig(BaseModel):
    """Upload settings for physicochemical sidecar datasets."""

    repo_id: str = Field(..., description="Target HF dataset repository ID.")
    subset_name: str = Field(..., description="Target subset/config name.")
    split: str = Field("train", description="Output split name.")
    commit_message: str = Field(..., description="HF commit message.")
    token: str | None = Field(None, description="HF token. Uses HF_TOKEN when unset.")

    class Config:
        extra = Extra.allow


def _physicochemical_upload_features(feature_columns: list[str]) -> Features:
    return Features(
        {
            "peptide_id": Value("string"),
            **{column_name: Value("float64") for column_name in feature_columns},
        }
    )


def _validate_output_columns(rows: list[dict[str, Any]], expected_columns: list[str]) -> None:
    missing = [column for column in expected_columns if rows and column not in rows[0]]
    if missing:
        raise ValueError(f"Missing expected upload columns: {missing}")


def _iter_dataset_chunks(dataset: Any, chunk_size: int):
    for start in range(0, len(dataset), chunk_size):
        stop = min(start + chunk_size, len(dataset))
        chunk = dataset[start:stop]
        keys = list(chunk.keys())
        row_count = len(chunk[keys[0]]) if keys else 0
        yield [{column: chunk[column][idx] for column in keys} for idx in range(row_count)]


def _build_physicochemical_rows(
    cfg: PreparePhysicochemicalConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not cfg.properties:
        raise ValueError("properties must not be empty.")
    unknown = [name for name in cfg.properties if name not in SUPPORTED_PROPERTIES]
    if unknown:
        raise ValueError(f"Unsupported properties requested: {unknown}")

    dataset = load_dataset(cfg.dataset_name, name=cfg.dataset_config, split=cfg.split)
    for required in (cfg.peptide_id_column, cfg.sequence_column):
        if required not in dataset.column_names:
            raise ValueError(f"Input dataset missing required column '{required}'.")

    all_rows: list[dict[str, Any]] = []
    chunks = _iter_dataset_chunks(dataset, cfg.batch_size)
    iterator = tqdm(chunks, desc="Computing physicochemical properties") if cfg.use_tqdm else chunks
    for chunk in iterator:
        for item in chunk:
            key = item.get(cfg.peptide_id_column)
            sequence = item.get(cfg.sequence_column)
            if key is None:
                raise ValueError(f"Found null key in column '{cfg.peptide_id_column}'.")
            if not isinstance(sequence, str) or not sequence.strip():
                raise ValueError(f"Found invalid sequence for key '{key}'.")

            row: dict[str, Any] = {"peptide_id": key}
            row.update(
                compute_properties(
                    sequence,
                    ph=cfg.ph,
                    hydrophobicity_scale=cfg.hydrophobicity_scale,
                    properties=cfg.properties,
                )
            )
            all_rows.append(row)

    return all_rows, list(cfg.properties)


def prepare_and_upload_physicochemical_properties_command(
    *,
    config_file: Path = CONFIG_FILE_OPTION,
) -> None:
    """Generate physicochemical sidecar rows and upload as a HF subset."""
    raw = load_config_file(config_file)
    prepare_raw = raw.get("prepare", {})
    upload_raw = raw.get("upload", {})
    if not prepare_raw:
        typer.echo("❌ Config must include 'prepare' section.", err=True)
        raise typer.Exit(1)
    if not upload_raw:
        typer.echo("❌ Config must include 'upload' section.", err=True)
        raise typer.Exit(1)

    prepare_cfg = PreparePhysicochemicalConfig(**prepare_raw)
    if not upload_raw.get("token"):
        upload_raw["token"] = os.getenv("HF_TOKEN")
    upload_cfg = PhysicochemicalUploadConfig(**upload_raw)

    typer.echo(
        f"🚀 Building physicochemical sidecar for {prepare_cfg.dataset_name}"
        f" ({prepare_cfg.dataset_config or 'default-config'}/{prepare_cfg.split})"
    )
    rows, feature_columns = _build_physicochemical_rows(prepare_cfg)
    expected_columns = ["peptide_id", *feature_columns]
    _validate_output_columns(rows, expected_columns)
    features = _physicochemical_upload_features(feature_columns)
    columns = {"peptide_id": "peptide_id", **{col: col for col in feature_columns}}
    streams = {upload_cfg.split: rows}
    upload_streams_to_huggingface_subset(
        streams=streams,
        repo_id=upload_cfg.repo_id,
        subset_name=upload_cfg.subset_name,
        columns=columns,
        commit_message=upload_cfg.commit_message,
        token=upload_cfg.token,
        features=features,
    )
    typer.echo(
        "📦 Uploaded physicochemical sidecar to "
        f"https://huggingface.co/datasets/{upload_cfg.repo_id}/{upload_cfg.subset_name}"
    )
