"""Build Hugging Face ``datasets`` and collators for :class:`transformers.Trainer`."""

from __future__ import annotations

from typing import Any

from datasets import concatenate_datasets, load_dataset
from omegaconf import DictConfig, OmegaConf

from modelling.src.utils.importing import get_obj_from_import_path

# Resolved ``data`` YAML: mapping of name -> HF load spec to concatenate for training.
DEFAULT_TRAIN_SPECS_KEY = "train_datasets"


def _to_container(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Resolve to a plain ``dict`` (Hydra ``DictConfig`` or already-materialized dict)."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, dict):
        return dict(cfg)
    msg = f"Expected DictConfig or dict, got {type(cfg).__name__}"
    raise TypeError(msg)


def _load_one_dataset(dataset_cfg: DictConfig | dict[str, Any]) -> Any:
    """Load a single split from ``datasets.load_dataset``."""
    dc = _to_container(dataset_cfg)
    hf_kwargs = dict(dc["hf_kwargs"])
    shuffle_kwargs = dc.get("shuffle_kwargs")
    ds = load_dataset(**hf_kwargs)
    if isinstance(ds, dict):
        split = hf_kwargs.get("split")
        if split is None:
            raise ValueError("Dataset returns multiple splits but hf_kwargs.split is missing.")
        ds = ds[split]
    if shuffle_kwargs:
        ds = ds.shuffle(**shuffle_kwargs)
    return ds


def _merge_dataset_specs(specs: DictConfig | dict[str, Any]) -> Any:
    """Load each named spec and concatenate (single spec returns that dataset as-is)."""
    mapping = _to_container(specs)
    pieces = [_load_one_dataset(spec) for spec in mapping.values()]
    return concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]


def build_collator(data_cfg: DictConfig | dict[str, Any]):
    """Instantiate ``TokenizerCollate`` from ``data.collate``."""
    dc = _to_container(data_cfg)
    coll = dc.get("collate")
    if coll is None:
        raise ValueError("data.collate is required for sequence training.")
    collate_class = get_obj_from_import_path(coll["collate_class"])
    collate_config_class = get_obj_from_import_path(coll["collate_config_class"])
    collate_kwargs = coll.get("collate_kwargs") or {}
    cfg = collate_config_class(**collate_kwargs)
    return collate_class(cfg)


def build_datasets(data_cfg: DictConfig | dict[str, Any]) -> tuple[Any, dict[str, Any] | None, Any]:
    """Return ``train_dataset``, optional ``eval_dataset`` dict, and ``data_collator``."""
    dc = _to_container(data_cfg)
    specs_key = dc.get("train_specs_key") or DEFAULT_TRAIN_SPECS_KEY
    if specs_key not in dc:
        msg = f"data.{specs_key} is missing (train_specs_key={specs_key!r})."
        raise ValueError(msg)
    train = _merge_dataset_specs(dc[specs_key])

    eval_dataset = None
    if dc.get("val_datasets"):
        eval_dataset = {
            name: _load_one_dataset(ds_cfg) for name, ds_cfg in dc["val_datasets"].items()
        }

    collator = build_collator(data_cfg)
    return train, eval_dataset, collator


def dataset_provides_iterable_train(data_cfg: DictConfig | dict[str, Any]) -> bool:
    """True if the training path uses an iterable (e.g. streaming) dataset."""
    dc = _to_container(data_cfg)
    specs_key = dc.get("train_specs_key") or DEFAULT_TRAIN_SPECS_KEY
    first = next(iter(dc[specs_key].values()))
    return bool(first["hf_kwargs"].get("streaming", False))
