"""Unit tests for evaluation pipeline config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.evaluation.pipeline import load_eval_config


def test_load_eval_config_requires_dataset_name(tmp_path: Path) -> None:
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(
        """
mlflow:
  experiment_name: amp_challenge_eval
evaluation:
  dataset:
    source: huggingface
    split: train
  columns:
    sequence: sequence
    hc50: hc50
    strains:
      A. baumannii ATCC 19606: col
        """.strip()
    )

    with pytest.raises(ValueError, match="evaluation.dataset.name"):
        load_eval_config(str(cfg))
