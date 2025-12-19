"""Callbacks for scheduling loss weights during training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_warn


@dataclass
class LinearWarmupConfig:
    """Linear warmup from start_weight to end_weight over warmup_steps."""

    loss_name: str
    start_weight: float = 0.0
    end_weight: float = 1.0
    warmup_steps: int = 10_000


class LinearLossWeightWarmup(Callback):
    """Linearly warm up a LossManager loss weight by name.

    This is useful for KL warmup in VAEs when using MetaModule + LossManager.
    It assumes the LightningModule has ``loss_manager.losses[loss_name]['weight']``.
    """

    def __init__(
        self,
        loss_name: str = "kl_divergence",
        start_weight: float = 0.0,
        end_weight: float = 0.001,
        warmup_steps: int = 10_000,
    ) -> None:
        super().__init__()
        self.cfg = LinearWarmupConfig(
            loss_name=loss_name,
            start_weight=float(start_weight),
            end_weight=float(end_weight),
            warmup_steps=int(warmup_steps),
        )
        self._warned_missing = False

    def _set_weight(self, pl_module, weight: float) -> None:
        loss_manager = getattr(pl_module, "loss_manager", None)
        losses = getattr(loss_manager, "losses", None) if loss_manager is not None else None
        if not isinstance(losses, dict) or self.cfg.loss_name not in losses:
            if not self._warned_missing:
                self._warned_missing = True
                rank_zero_warn(
                    f"LinearLossWeightWarmup could not find loss '{self.cfg.loss_name}' on "
                    "pl_module.loss_manager.losses; skipping weight scheduling."
                )
            return
        losses[self.cfg.loss_name]["weight"] = float(weight)

    def on_fit_start(self, trainer, pl_module) -> None:
        self._set_weight(pl_module, self.cfg.start_weight)

    def on_train_batch_start(
        self,
        trainer,
        pl_module,
        batch,
        batch_idx: int,
    ) -> None:
        if self.cfg.warmup_steps <= 0:
            self._set_weight(pl_module, self.cfg.end_weight)
            return

        step = int(trainer.global_step)
        progress = min(1.0, max(0.0, step / float(self.cfg.warmup_steps)))
        weight = self.cfg.start_weight + progress * (self.cfg.end_weight - self.cfg.start_weight)
        self._set_weight(pl_module, weight)

