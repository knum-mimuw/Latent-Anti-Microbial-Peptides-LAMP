"""ZenML step wrapper around the standalone pep_eval package."""

from __future__ import annotations

from typing import Any

from pep_eval import evaluate_dataset
from zenml import step


@step(enable_cache=False)
def evaluate_hf_dataset(
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
    """Evaluate and log all category metrics to MLflow."""
    return evaluate_dataset(
        experiment_name=experiment_name,
        dataset_source=dataset_source,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        sequence_column=sequence_column,
        hc50_column=hc50_column,
        strain_columns=strain_columns,
        run_name_prefix=run_name_prefix,
    )
