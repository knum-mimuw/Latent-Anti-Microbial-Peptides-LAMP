"""Training pipeline -- orchestrates training + artifact logging via ZenML."""

from __future__ import annotations

import sys

from .._zenml_sqlalchemy_uuid_compat import apply as _apply_zenml_uuid_compat

_apply_zenml_uuid_compat()

from zenml import pipeline

from .steps.train_step import train


@pipeline
def training_pipeline(config_paths: list[str], run_config_path: str):
    """Run Lightning training with MLflow-managed canonical artifacts."""
    train(config_paths, run_config_path)


def main() -> None:
    """CLI entrypoint."""
    config_paths = sys.argv[1:]
    if len(config_paths) < 2:
        print(
            "Usage: python -m pipelines.training <run-config.yaml> <config1.yaml> [config2.yaml ...]"
        )
        sys.exit(1)
    training_pipeline(run_config_path=config_paths[0], config_paths=config_paths[1:])


if __name__ == "__main__":
    main()
