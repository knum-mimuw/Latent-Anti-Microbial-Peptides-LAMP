"""Training pipeline -- orchestrates training + artifact logging via ZenML.

Usage::

    uv run python -m pipelines.training \\
        path/to/run-config.yaml \\
        configs/trainer/grugru_vae.yaml \\
        configs/logger/mlflow_local.yaml \\
        configs/callbacks/checkpoint.yaml
"""

from __future__ import annotations

import sys

from zenml import pipeline

from .steps.train_step import train


@pipeline
def training_pipeline(config_paths: list[str], run_config_path: str):
    """Run Lightning training with MLflow-managed canonical artifacts."""
    train(config_paths, run_config_path)


if __name__ == "__main__":
    config_paths = sys.argv[1:]
    if len(config_paths) < 2:
        print(
            "Usage: python -m pipelines.training <run-config.yaml> <config1.yaml> [config2.yaml ...]"
        )
        sys.exit(1)
    training_pipeline(run_config_path=config_paths[0], config_paths=config_paths[1:])
