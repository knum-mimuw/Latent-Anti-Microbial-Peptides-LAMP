"""Instantiate model, datasets, callbacks, and :class:`transformers.Trainer`."""

from __future__ import annotations

from typing import Any

import hydra.utils
from omegaconf import DictConfig
from transformers import Trainer, TrainingArguments

from modelling.src.training.data import build_datasets, dataset_provides_iterable_train

from modelling.src.callbacks.iterable_epoch import IterableEpochCallback
from modelling.src.callbacks.logging import LoggingCallback


def build_callbacks(cfg: DictConfig, training_args: TrainingArguments) -> list[Any]:
    """Trainer callbacks (logging, optional iterable epoch, user-defined)."""
    del training_args
    callbacks: list[Any] = [
        LoggingCallback(),
    ]
    if dataset_provides_iterable_train(cfg.data):
        callbacks.append(IterableEpochCallback())
    extra = cfg.get("callbacks") or []
    for item in extra:
        callbacks.append(hydra.utils.instantiate(item))
    return callbacks


def build_trainer(cfg: DictConfig) -> Trainer:
    """Compose ``Trainer`` from the composed Hydra configuration."""
    model = hydra.utils.instantiate(cfg.model)
    train_ds, eval_ds, collator = build_datasets(cfg.data)
    training_args: TrainingArguments = hydra.utils.instantiate(cfg.training)

    callbacks = build_callbacks(cfg, training_args)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
