"""Instantiate model, datasets, callbacks, and :class:`transformers.Trainer`."""

from __future__ import annotations

from typing import Any

import hydra.utils
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from modelling.src.training.data import build_datasets, dataset_provides_iterable_train
from modelling.src.training.trainer import LoggingTrainer

from modelling.src.callbacks.iterable_epoch import IterableEpochCallback
from modelling.src.callbacks.manifest import ManifestCallback


def build_callbacks(cfg: DictConfig, training_args: TrainingArguments) -> list[Any]:
    """Trainer callbacks (manifest, optional iterable epoch, user-defined)."""
    del training_args
    mcfg = cfg.get("manifest") or {}
    callbacks: list[Any] = [
        ManifestCallback(
            manifest_path=mcfg.get("manifest_path"),
            checkpoint_artifact_path=mcfg.get("checkpoint_artifact_path"),
            manifest_artifact_path=mcfg.get("manifest_artifact_path"),
        ),
    ]
    if dataset_provides_iterable_train(cfg.data):
        callbacks.append(IterableEpochCallback())
    extra = cfg.get("callbacks") or []
    for item in extra:
        callbacks.append(hydra.utils.instantiate(item))
    return callbacks


def build_trainer(cfg: DictConfig) -> LoggingTrainer:
    """Compose ``Trainer`` from the composed Hydra configuration."""
    model = hydra.utils.instantiate(cfg.model)
    train_ds, eval_ds, collator = build_datasets(cfg.data)
    training_args: TrainingArguments = hydra.utils.instantiate(cfg.training)

    compute_metrics = None
    if OmegaConf.select(cfg, "metrics.compute_metrics_target") is not None:
        compute_metrics = hydra.utils.instantiate(cfg.metrics.compute_metrics_target)

    callbacks = build_callbacks(cfg, training_args)

    return LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
