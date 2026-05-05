"""Callback that normalises HF Trainer log keys and injects sub-losses."""

from __future__ import annotations

from transformers import TrainerCallback


def _is_noisy_eval_timing_metric(key: str) -> bool:
    if not key.startswith("eval"):
        return False
    return (
        key.endswith("_runtime")
        or key.endswith("_samples_per_second")
        or key.endswith("_steps_per_second")
    )


def _scalar(x):
    if hasattr(x, "dim") and x.dim() > 0:
        x = x.mean()
    return x.detach().float().item()


class LoggingCallback(TrainerCallback):
    """Rewrites HF log dicts to use ``train/`` / ``eval/`` prefixes and appends sub-losses."""

    def __init__(self):
        self._last_sub_losses = None
        self._last_mode = "train"

    def on_step_end(self, args, state, control, **kwargs):
        outputs = kwargs.get("outputs")
        model = kwargs.get("model")

        if outputs is None:
            return

        sub = getattr(outputs, "sub_losses", None)
        if sub:
            self._last_sub_losses = sub
            self._last_mode = "train" if model.training else "eval"

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        new_logs = {}

        for k, v in logs.items():
            if _is_noisy_eval_timing_metric(k):
                continue

            if k.startswith("eval_"):
                new_logs[f"eval/{k[5:]}"] = v
            elif not k.startswith("train/"):
                new_logs[f"train/{k}"] = v
            else:
                new_logs[k] = v

        if self._last_sub_losses:
            prefix = self._last_mode
            for k, v in self._last_sub_losses.items():
                new_logs[f"{prefix}/{k}"] = _scalar(v)

        logs.clear()
        logs.update(new_logs)
