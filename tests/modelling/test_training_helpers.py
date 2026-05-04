"""Tests for training utilities (collate labels, Hydra flatten)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

from modelling.src.datamodules.collate import TokenizerCollate, TokenizerCollateConfig
from modelling.src.training.hydra_overrides import flatten_yaml_file


def test_tokenizer_collate_adds_shifted_labels() -> None:
    cfg = TokenizerCollateConfig(
        tokenizer_path="dummy",
        sequence_column="sequence",
        tokenizer_kwargs={},
        add_shifted_labels=True,
    )
    collate = object.__new__(TokenizerCollate)
    collate.config = cfg
    mock_tok = MagicMock()
    mock_tok.return_value = {"input_ids": torch.tensor([[5, 6, 7, 0], [1, 2, 0, 0]])}
    collate.tokenizer = mock_tok

    batch = [{"sequence": "AB"}, {"sequence": "C"}]
    out = collate(batch)

    assert "labels" in out
    assert torch.equal(out["labels"], out["input_ids"][:, 1:])


def test_flatten_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "training:\n  num_train_epochs: 3\n  learning_rate: 1.0e-5\n",
        encoding="utf-8",
    )
    overrides = flatten_yaml_file(path)
    assert "training.num_train_epochs=3" in overrides
    assert any(o.startswith("training.learning_rate=") for o in overrides)
