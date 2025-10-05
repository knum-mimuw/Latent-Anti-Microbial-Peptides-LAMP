import os
from typing import Optional, List, Dict, Any
from datasets import Dataset as HFDataset
import typer


def _build_dataset_dict(
    items: List[Dict[str, Any]],
    *,
    columns: Dict[str, Any],
) -> Dict[str, List[Any]]:
    """Convert items to a datasets-compatible dict using explicit columns only.

    columns maps output column -> source (str key) or callable(item)->value.
    No inference or defaults are applied.
    """
    if not isinstance(columns, dict) or not columns:
        raise ValueError("`columns` must be a non-empty dict of column specs.")

    out: Dict[str, List[Any]] = {}
    for col, spec in columns.items():
        if callable(spec):
            out[col] = [spec(item) for item in items]
        elif isinstance(spec, str):
            out[col] = [item.get(spec) for item in items]
        else:
            raise TypeError(f"Unsupported column spec for {col}: {type(spec)}")
    return out


def upload_data_to_huggingface(
    sequences: List[Dict[str, Any]],
    *,
    repo_id: str,
    columns: Dict[str, Any],
    token: Optional[str] = None,
    batch_size: Optional[int] = None,
    commit_message: str = "Add sequences",
    logger: Optional[Any] = None,
) -> None:
    """Upload sequences to a Hugging Face dataset repository.

    - If batch_size is None or >= len(sequences), uploads in a single commit.
    - If batch_size > 0, processes items in batches, aggregates, and uploads once.
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

    if not batch_size or batch_size >= len(sequences):
        dataset_data = _build_dataset_dict(sequences, columns=columns)
    else:
        merged: List[Dict[str, Any]] = []
        total = len(sequences)
        if logger is not None:
            logger.info("Processing %s sequences in batches of %s", total, batch_size)
        for i in range(0, total, batch_size):
            batch = sequences[i : i + batch_size]
            if logger is not None:
                logger.info(
                    "Processing batch %s/%s",
                    i // batch_size + 1,
                    (total + batch_size - 1) // batch_size,
                )
            merged.extend(batch)
        dataset_data = _build_dataset_dict(merged, columns=columns)

    try:
        hf_dataset = HFDataset.from_dict(dataset_data)
        hf_dataset.push_to_hub(
            repo_id=repo_id, token=token, commit_message=commit_message
        )
    except Exception as e:
        typer.echo(f"❌ Upload failed: {e}", err=True)
        raise typer.Exit(1)
    if logger is not None:
        logger.info("Successfully uploaded %s sequences to %s", len(sequences), repo_id)
