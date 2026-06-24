"""Top-level API for AMP evaluation."""

from __future__ import annotations

from typing import Any

from .io import dataset_from_source
from .logging import log_evaluation_run
from .metrics import compute_per_peptide_metrics, compute_scorecard
from .panels import POTENCY_THRESHOLD_UM


def evaluate_dataset(
    *,
    experiment_name: str,
    dataset_source: str,
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str | None,
    sequence_column: str,
    hc50_column: str,
    strain_columns: dict[str, str],
    run_name_prefix: str = "amp-eval",
) -> dict[str, Any]:
    """Evaluate dataset and log full results to MLflow."""
    dataset = dataset_from_source(
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
    )
    records = [dict(row) for row in dataset]
    per_peptide = compute_per_peptide_metrics(
        records=records,
        sequence_column=sequence_column,
        hc50_column=hc50_column,
        strain_columns=strain_columns,
    )
    scorecard = compute_scorecard(per_peptide)
    run_id = log_evaluation_run(
        experiment_name=experiment_name,
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        sequence_column=sequence_column,
        hc50_column=hc50_column,
        strain_columns=strain_columns,
        run_name_prefix=run_name_prefix,
        scorecard=scorecard,
        per_peptide=per_peptide,
        potency_threshold_um=POTENCY_THRESHOLD_UM,
    )
    return {"mlflow_run_id": run_id, "scorecard": scorecard}
