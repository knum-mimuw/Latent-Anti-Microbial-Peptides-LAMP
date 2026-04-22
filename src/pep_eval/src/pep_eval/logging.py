"""MLflow logging helpers for evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import mlflow

from .panels import panel_dict


def slugify_dataset_name(dataset_name: str) -> str:
    """Normalize dataset name for run names."""
    return (
        dataset_name.strip()
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def log_evaluation_run(
    *,
    experiment_name: str,
    dataset_source: str,
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str | None,
    sequence_column: str,
    hc50_column: str,
    strain_columns: dict[str, str],
    run_name_prefix: str,
    scorecard: dict[str, float],
    per_peptide: list[dict[str, Any]],
    potency_threshold_um: float,
) -> str:
    """Log complete evaluation metadata/metrics directly to MLflow."""
    tracking_uri = mlflow.get_tracking_uri()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set before evaluation.")

    run_name = (
        f"{run_name_prefix}::{slugify_dataset_name(dataset_name)}::{dataset_split}::"
        f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "hf_dataset_name": dataset_name,
                "hf_dataset_split": dataset_split,
                "dataset_source": dataset_source,
                "pipeline": "amp_evaluation",
            }
        )
        mlflow.log_params(
            {
                "dataset_name": dataset_name,
                "dataset_split": dataset_split,
                "dataset_source": dataset_source,
                "dataset_revision": dataset_revision or "",
                "sequence_column": sequence_column,
                "hc50_column": hc50_column,
                "potency_threshold_um": potency_threshold_um,
                "rows_evaluated": len(per_peptide),
            }
        )
        mlflow.log_dict(strain_columns, "config/strain_columns.json")
        mlflow.log_dict(panel_dict(), "config/panels.json")
        mlflow.log_metrics(scorecard)
        mlflow.log_dict(
            {"scorecard": scorecard, "rows_evaluated": len(per_peptide)},
            "results/scorecard.json",
        )
        mlflow.log_table(per_peptide, "results/per_peptide_metrics.json")
        return run.info.run_id
