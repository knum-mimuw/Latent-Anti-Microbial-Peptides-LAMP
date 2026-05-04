"""Call ``set_epoch`` on Hugging Face iterable datasets each epoch."""

from __future__ import annotations

from transformers import TrainerCallback


class IterableEpochCallback(TrainerCallback):
    """For streaming / iterable training data, forward Trainer epoch to ``set_epoch``."""

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        del args, kwargs
        if train_dataloader is None:
            return control
        ds = getattr(train_dataloader, "dataset", None)
        while ds is not None and hasattr(ds, "dataset"):
            inner = getattr(ds, "dataset", None)
            if inner is ds:
                break
            ds = inner
        if ds is not None and callable(getattr(ds, "set_epoch", None)):
            ds.set_epoch(int(state.epoch))
        return control
