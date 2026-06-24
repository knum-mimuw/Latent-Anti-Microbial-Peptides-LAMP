"""Dataset loading utilities for evaluation."""

from __future__ import annotations

from datasets import Dataset, load_dataset, load_from_disk


def dataset_from_source(
    dataset_source: str,
    dataset_name: str,
    dataset_split: str,
    dataset_revision: str | None,
) -> Dataset:
    """Load dataset from HF hub or local disk."""
    if dataset_source == "huggingface":
        return load_dataset(dataset_name, split=dataset_split, revision=dataset_revision)
    if dataset_source == "disk":
        loaded = load_from_disk(dataset_name)
        if isinstance(loaded, Dataset):
            if dataset_split != "train":
                raise ValueError(
                    "When dataset_source='disk' points to a Dataset object, dataset_split must be 'train'."
                )
            return loaded
        if dataset_split not in loaded:
            raise ValueError(
                f"Split '{dataset_split}' not found in local dataset. Available: {list(loaded.keys())}"
            )
        return loaded[dataset_split]
    raise ValueError(f"Unsupported dataset_source='{dataset_source}'. Use 'huggingface' or 'disk'.")
