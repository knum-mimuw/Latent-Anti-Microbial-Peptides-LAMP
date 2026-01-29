"""Epoch-related callbacks for PyTorch Lightning."""

import pytorch_lightning as pl
from datasets import IterableDataset


class HFIterableSetEpoch(pl.Callback):
    """Callback to set epoch on HuggingFace IterableDataset for proper shuffling."""

    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        ds = getattr(dm, "train_dataset", None)
        if isinstance(ds, IterableDataset):
            ds.set_epoch(trainer.current_epoch)
