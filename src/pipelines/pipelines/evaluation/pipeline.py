"""ZenML pipeline entrypoint for AMP Challenge evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from .._zenml_sqlalchemy_uuid_compat import apply as _apply_zenml_uuid_compat

_apply_zenml_uuid_compat()

from zenml import pipeline

from .steps.evaluate_hf_dataset_step import evaluate_hf_dataset


@dataclass(frozen=True)
class EvalConfig:
    """Run-scoped evaluation configuration."""

    experiment_name: str
    dataset_source: str
    dataset_name: str
    dataset_split: str
    dataset_revision: str | None
    sequence_column: str
    hc50_column: str
    strain_columns: dict[str, str]
    run_name_prefix: str


def load_eval_config(config_path: str) -> EvalConfig:
    """Load and validate the YAML configuration for evaluation."""
    path = Path(config_path)
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}

    mlflow_payload = payload.get("mlflow", {})
    eval_payload = payload.get("evaluation", {})
    dataset_payload = eval_payload.get("dataset", {})
    columns_payload = eval_payload.get("columns", {})

    experiment_name = str(mlflow_payload.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ValueError(f"{path} must define mlflow.experiment_name.")

    dataset_source = str(dataset_payload.get("source", "")).strip()
    if dataset_source not in {"huggingface", "disk"}:
        raise ValueError(
            f"{path} must define evaluation.dataset.source as 'huggingface' or 'disk'."
        )

    dataset_name = str(dataset_payload.get("name", "")).strip()
    if not dataset_name:
        raise ValueError(f"{path} must define evaluation.dataset.name.")

    dataset_split = str(dataset_payload.get("split", "")).strip()
    if not dataset_split:
        raise ValueError(f"{path} must define evaluation.dataset.split.")

    dataset_revision_raw = dataset_payload.get("revision")
    dataset_revision = None if dataset_revision_raw is None else str(dataset_revision_raw).strip()
    if dataset_revision == "":
        dataset_revision = None

    sequence_column = str(columns_payload.get("sequence", "")).strip()
    hc50_column = str(columns_payload.get("hc50", "")).strip()
    strain_columns_raw = columns_payload.get("strains", {})

    if not sequence_column or not hc50_column:
        raise ValueError(
            f"{path} must define evaluation.columns.sequence and evaluation.columns.hc50."
        )
    if not isinstance(strain_columns_raw, dict) or not strain_columns_raw:
        raise ValueError(f"{path} must define a non-empty evaluation.columns.strains mapping.")

    strain_columns = {
        str(canonical).strip(): str(column).strip()
        for canonical, column in strain_columns_raw.items()
    }
    if any(not key or not value for key, value in strain_columns.items()):
        raise ValueError(f"{path} contains empty keys/values in evaluation.columns.strains.")

    run_name_prefix = str(eval_payload.get("run_name_prefix", "amp-eval")).strip()
    if not run_name_prefix:
        raise ValueError(f"{path} defines an empty evaluation.run_name_prefix.")

    return EvalConfig(
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


@pipeline
def evaluation_pipeline(
    experiment_name: str,
    dataset_source: str,
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str | None,
    sequence_column: str,
    hc50_column: str,
    strain_columns: dict[str, str],
    run_name_prefix: str = "amp-eval",
):
    """Run AMP challenge evaluation and log all category metrics to MLflow."""
    evaluate_hf_dataset(
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


def main() -> None:
    """CLI entrypoint for the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate an AMP dataset and log metrics to MLflow")
    parser.add_argument("run_config_path", help="YAML config file for evaluation.")
    args = parser.parse_args()

    cfg = load_eval_config(args.run_config_path)
    evaluation_pipeline(
        experiment_name=cfg.experiment_name,
        dataset_source=cfg.dataset_source,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        dataset_revision=cfg.dataset_revision,
        sequence_column=cfg.sequence_column,
        hc50_column=cfg.hc50_column,
        strain_columns=cfg.strain_columns,
        run_name_prefix=cfg.run_name_prefix,
    )


if __name__ == "__main__":
    main()
