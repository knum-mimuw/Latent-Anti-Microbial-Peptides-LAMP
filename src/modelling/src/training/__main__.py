"""Hydra CLI entry for ``uv run modelling``."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from modelling.src.training.build import build_trainer

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIG_ROOT), config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = build_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
