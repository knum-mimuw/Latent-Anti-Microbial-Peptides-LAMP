"""Trainer subclass with automatic sub-loss logging."""

from __future__ import annotations

from transformers import Trainer


class LoggingTrainer(Trainer):
    """Trainer that logs ``train/`` and ``val/`` prefixed metrics.

    Any model whose ``forward`` returns a ``ModelOutput`` with a
    ``sub_losses: dict[str, Tensor] | None`` attribute will get each
    entry logged alongside the main loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        prefix = "train" if model.training else "val"
        metrics = {f"{prefix}/loss": loss.item()}
        sub = getattr(outputs, "sub_losses", None)
        if sub:
            metrics.update({f"{prefix}/{k}": v.item() for k, v in sub.items()})
        self.log(metrics)

        return (loss, outputs) if return_outputs else loss
