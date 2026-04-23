import os
from numbers import Real
from pathlib import Path
from typing import Any

import typer
from datasets import Features, Value, load_dataset
from pydantic import BaseModel, Extra, Field

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


class PrepareStrainConditionsConfig(BaseModel):
    """Configuration for strain-condition feature generation."""

    dataset_name: str = Field(
        ...,
        description="HF dataset repository ID containing APEX predictions.",
    )
    dataset_config: str | None = Field(None, description="Optional HF config name.")
    split: str = Field("train", description="Input split to process.")
    peptide_id_column: str = Field("peptide_id", description="Primary key column.")
    strain_columns: list[str] = Field(..., description="Selected numeric strain MIC columns.")
    strain_groups: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Optional group->strain columns mapping for grouped means.",
    )
    batch_size: int = Field(50000, description="Chunk size for feature processing.")

    class Config:
        extra = Extra.allow


class StrainConditionUploadConfig(BaseModel):
    """Upload settings for condition sidecar datasets."""

    repo_id: str = Field(..., description="Target HF dataset repository ID.")
    subset_name: str = Field(..., description="Target subset/config name.")
    split: str = Field("train", description="Output split name.")
    commit_message: str = Field(..., description="HF commit message.")
    token: str | None = Field(None, description="HF token. Uses HF_TOKEN when unset.")

    class Config:
        extra = Extra.allow


def _strain_condition_upload_features(feature_columns: list[str]) -> Features:
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


def _validate_strain_group_columns(
    strain_columns: list[str], strain_groups: dict[str, list[str]]
) -> None:
    strain_set = set(strain_columns)
    for group, columns in strain_groups.items():
        if not columns:
            raise ValueError(f"strain_groups['{group}'] cannot be empty.")
        missing = [column for column in columns if column not in strain_set]
        if missing:
            raise ValueError(f"Group '{group}' references unknown strain columns: {missing}")


def _ensure_numeric(value: Any, *, column: str, key: Any) -> float:
    if not isinstance(value, Real):
        raise ValueError(f"Column '{column}' for key '{key}' must be numeric.")
    return float(value)


def _build_condition_rows(
    cfg: PrepareStrainConditionsConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not cfg.strain_columns:
        raise ValueError("strain_columns must not be empty.")

    dataset = load_dataset(cfg.dataset_name, name=cfg.dataset_config, split=cfg.split)
    required_columns = [cfg.peptide_id_column, *cfg.strain_columns]
    for required in required_columns:
        if required not in dataset.column_names:
            raise ValueError(f"Input dataset missing required column '{required}'.")

    _validate_strain_group_columns(cfg.strain_columns, cfg.strain_groups)

    feature_columns = [
        "cond_mean_all_strains",
        *[f"cond_mean_{name}" for name in cfg.strain_groups],
    ]
    all_rows: list[dict[str, Any]] = []

    for chunk in _iter_dataset_chunks(dataset, cfg.batch_size):
        for item in chunk:
            key = item.get(cfg.peptide_id_column)
            if key is None:
                raise ValueError(f"Found null key in column '{cfg.peptide_id_column}'.")

            strain_values = [
                _ensure_numeric(item.get(column), column=column, key=key)
                for column in cfg.strain_columns
            ]
            row: dict[str, Any] = {
                "peptide_id": key,
                "cond_mean_all_strains": sum(strain_values) / len(strain_values),
            }
            for group, columns in cfg.strain_groups.items():
                values = [
                    _ensure_numeric(item.get(column), column=column, key=key)
                    for column in columns
                ]
                row[f"cond_mean_{group}"] = sum(values) / len(values)
            all_rows.append(row)

    return all_rows, feature_columns


def prepare_and_upload_strain_conditions_command(
    *,
    config_file: Path = CONFIG_FILE_OPTION,
) -> None:
    """Generate mean-based strain condition features and upload as a HF subset."""
    raw = load_config_file(config_file)
    prepare_raw = raw.get("prepare", {})
    upload_raw = raw.get("upload", {})
    if not prepare_raw:
        typer.echo("❌ Config must include 'prepare' section.", err=True)
        raise typer.Exit(1)
    if not upload_raw:
        typer.echo("❌ Config must include 'upload' section.", err=True)
        raise typer.Exit(1)

    prepare_cfg = PrepareStrainConditionsConfig(**prepare_raw)
    if not upload_raw.get("token"):
        upload_raw["token"] = os.getenv("HF_TOKEN")
    upload_cfg = StrainConditionUploadConfig(**upload_raw)

    typer.echo(
        f"🚀 Building strain-condition features from {prepare_cfg.dataset_name}"
        f" ({prepare_cfg.dataset_config or 'default-config'}/{prepare_cfg.split})"
    )
    rows, feature_columns = _build_condition_rows(prepare_cfg)
    expected_columns = ["peptide_id", *feature_columns]
    _validate_output_columns(rows, expected_columns)
    features = _strain_condition_upload_features(feature_columns)
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
        f"📦 Uploaded strain conditions to "
        f"https://huggingface.co/datasets/{upload_cfg.repo_id}/{upload_cfg.subset_name}"
    )
