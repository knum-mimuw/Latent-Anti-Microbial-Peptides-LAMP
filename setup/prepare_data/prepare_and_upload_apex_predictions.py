import os
from numbers import Real
from pathlib import Path
from typing import Any, Literal

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


class PrepareApexPredictionsConfig(BaseModel):
    """Configuration for APEX prediction sidecar generation."""

    dataset_name: str = Field(..., description="HF dataset repository ID with peptide sequences.")
    dataset_config: str | None = Field(None, description="Optional HF config name.")
    split: str | None = Field(
        None,
        description="Dataset split to process. If null, all source splits are processed.",
    )
    peptide_id_column: str = Field("peptide_id", description="Column used as merge key.")
    sequence_column: str = Field("sequence", description="Peptide sequence column.")
    device: str = Field("cpu", description="Torch device string, e.g. cpu/cuda.")
    batch_size: int = Field(2048, description="Batch size used by PredictorAPEX internals.")
    chunk_size: int = Field(20000, description="Number of rows per prediction chunk.")
    path_mode: Literal["apex_pathogen", "apex_full_pathogen"] = Field(
        "apex_pathogen",
        description="Model set selector for APEX weights.",
    )
    use_tqdm: bool = Field(False, description="Enable progress bars in predictor.")

    class Config:
        extra = Extra.allow


class ApexUploadConfig(BaseModel):
    """Upload settings for prediction sidecar datasets."""

    repo_id: str = Field(..., description="Target HF dataset repository ID.")
    subset_name: str = Field(..., description="Target subset/config name.")
    commit_message: str = Field(..., description="HF commit message.")
    token: str | None = Field(None, description="HF token. Uses HF_TOKEN when unset.")

    class Config:
        extra = Extra.allow


def _apex_upload_features(id_column_name: str, pathogen_columns: list[str]) -> Features:
    return Features(
        {
            id_column_name: Value("string"),
            **{column_name: Value("float64") for column_name in pathogen_columns},
        }
    )


def _validate_output_columns(rows: list[dict[str, Any]], expected_columns: list[str]) -> None:
    missing = [column for column in expected_columns if rows and column not in rows[0]]
    if missing:
        raise ValueError(f"Missing expected upload columns: {missing}")


def _apex_internal_path(path_mode: str) -> str:
    if path_mode == "apex_pathogen":
        return "default"
    if path_mode == "apex_full_pathogen":
        return "all"
    raise ValueError(f"Unsupported path_mode '{path_mode}'.")


def _iter_dataset_chunks(dataset: Any, chunk_size: int):
    for start in range(0, len(dataset), chunk_size):
        stop = min(start + chunk_size, len(dataset))
        chunk = dataset[start:stop]
        keys = list(chunk.keys())
        row_count = len(chunk[keys[0]]) if keys else 0
        yield [{column: chunk[column][idx] for column in keys} for idx in range(row_count)]


def _build_prediction_rows_for_split(
    cfg: PrepareApexPredictionsConfig,
    *,
    split_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        from apex import PredictorAPEX
    except ImportError as exc:
        raise RuntimeError(
            "Failed importing apex.PredictorAPEX. "
            "Ensure workspace dependency 'lamp-apex' is available."
        ) from exc

    if cfg.batch_size <= 0 or cfg.chunk_size <= 0:
        raise ValueError("batch_size and chunk_size must be positive integers.")

    dataset = load_dataset(cfg.dataset_name, name=cfg.dataset_config, split=split_name)
    for required in (cfg.peptide_id_column, cfg.sequence_column):
        if required not in dataset.column_names:
            raise ValueError(f"Input dataset missing required column '{required}'.")

    predictor = PredictorAPEX(
        device=cfg.device,
        batch_size=cfg.batch_size,
        path=_apex_internal_path(cfg.path_mode),
    )
    pathogen_columns = [str(name) for name in predictor.pathogen_list]

    all_rows: list[dict[str, Any]] = []
    for chunk in _iter_dataset_chunks(dataset, cfg.chunk_size):
        sequence_batch = []
        key_batch = []
        for item in chunk:
            sequence = item.get(cfg.sequence_column)
            key = item.get(cfg.peptide_id_column)
            if key is None:
                raise ValueError(f"Found null peptide id in column '{cfg.peptide_id_column}'.")
            if not isinstance(sequence, str) or not sequence.strip():
                raise ValueError(f"Found invalid sequence value for key '{key}'.")
            key_batch.append(key)
            sequence_batch.append(sequence)

        predictions = predictor.predict(sequence_batch, use_tqdm=cfg.use_tqdm)
        if len(predictions) != len(sequence_batch):
            raise ValueError("Prediction row count mismatch.")

        for row_idx, key in enumerate(key_batch):
            row: dict[str, Any] = {cfg.peptide_id_column: key}
            for col_idx, column_name in enumerate(pathogen_columns):
                value = predictions[row_idx][col_idx]
                if not isinstance(value, Real):
                    raise ValueError(
                        f"Prediction value for '{column_name}' at key '{key}' is not numeric."
                    )
                row[column_name] = float(value)
            all_rows.append(row)

    return all_rows, pathogen_columns


def prepare_and_upload_apex_predictions_command(
    *,
    config_file: Path = CONFIG_FILE_OPTION,
) -> None:
    """Generate APEX MIC prediction sidecar rows and upload as a HF subset."""
    raw = load_config_file(config_file)
    prepare_raw = raw.get("prepare", {})
    upload_raw = raw.get("upload", {})
    if not prepare_raw:
        typer.echo("❌ Config must include 'prepare' section.", err=True)
        raise typer.Exit(1)
    if not upload_raw:
        typer.echo("❌ Config must include 'upload' section.", err=True)
        raise typer.Exit(1)

    prepare_cfg = PrepareApexPredictionsConfig(**prepare_raw)
    if not upload_raw.get("token"):
        upload_raw["token"] = os.getenv("HF_TOKEN")
    upload_cfg = ApexUploadConfig(**upload_raw)

    dataset_obj = load_dataset(prepare_cfg.dataset_name, name=prepare_cfg.dataset_config)
    if prepare_cfg.split is not None:
        if prepare_cfg.split not in dataset_obj:
            raise ValueError(
                f"Requested split '{prepare_cfg.split}' not found in source dataset."
            )
        split_names = [prepare_cfg.split]
    else:
        split_names = list(dataset_obj.keys())
    if not split_names:
        raise ValueError("No source dataset splits found.")

    typer.echo(
        f"🚀 Running APEX predictions for {prepare_cfg.dataset_name}"
        f" ({prepare_cfg.dataset_config or 'default-config'}) across splits: {split_names}"
    )

    streams: dict[str, list[dict[str, Any]]] = {}
    pathogen_columns: list[str] | None = None
    for split_name in split_names:
        rows, split_pathogen_columns = _build_prediction_rows_for_split(
            prepare_cfg,
            split_name=split_name,
        )
        if pathogen_columns is None:
            pathogen_columns = split_pathogen_columns
        elif split_pathogen_columns != pathogen_columns:
            raise ValueError("Pathogen output columns changed across splits.")
        expected_columns = [prepare_cfg.peptide_id_column, *split_pathogen_columns]
        _validate_output_columns(rows, expected_columns)
        streams[split_name] = rows

    if pathogen_columns is None:
        raise ValueError("No pathogen columns produced.")
    features = _apex_upload_features(prepare_cfg.peptide_id_column, pathogen_columns)
    columns = {
        prepare_cfg.peptide_id_column: prepare_cfg.peptide_id_column,
        **{col: col for col in pathogen_columns},
    }
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
        f"📦 Uploaded APEX predictions to "
        f"https://huggingface.co/datasets/{upload_cfg.repo_id}/{upload_cfg.subset_name}"
    )
