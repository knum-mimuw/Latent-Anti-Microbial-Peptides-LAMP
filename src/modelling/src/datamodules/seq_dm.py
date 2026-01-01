from typing import Any, Dict, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset
from pydantic import BaseModel, Field, ConfigDict


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    hf_kwargs: Dict[str, Any] = Field(
        ..., description="Arguments passed to load_dataset()"
    )
    shuffle_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments for .shuffle() (shuffle: bool, buffer_size: int)",
    )


class SequenceDataModuleConfig(BaseModel):
    """Configuration for the SequenceDataModule."""

    # Dict of {name: DatasetConfig}
    train_datasets: Dict[str, DatasetConfig] = Field(
        ..., description="Training datasets: {name: DatasetConfig}"
    )
    val_datasets: Optional[Dict[str, DatasetConfig]] = None
    test_datasets: Optional[Dict[str, DatasetConfig]] = None

    # Separate dataloader configs for train/val/test
    train_dataloader_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="DataLoader kwargs for training (batch_size, num_workers, etc.)",
    )
    val_dataloader_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="DataLoader kwargs for validation (batch_size, num_workers, etc.)",
    )
    test_dataloader_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="DataLoader kwargs for testing (batch_size, num_workers, etc.)",
    )

    model_config = ConfigDict(extra="allow")


class SequenceDataModule(LightningDataModule):
    """Lightning DataModule for multiple Hugging Face datasets."""

    def __init__(self, config: SequenceDataModuleConfig):
        super().__init__()
        self.config = config
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Dict[str, Dataset]] = None
        self.test_datasets: Optional[Dict[str, Dataset]] = None

    def _load_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        """Load one dataset from Hugging Face."""
        ds = load_dataset(**dataset_config.hf_kwargs)
        if isinstance(ds, dict):
            split = dataset_config.hf_kwargs.get("split")
            if split is None:
                raise ValueError(
                    "Dataset has multiple splits but no split specified in config"
                )
            ds = ds[split]

        # Apply shuffling if shuffle_kwargs provided
        if dataset_config.shuffle_kwargs:
            ds = ds.shuffle(**dataset_config.shuffle_kwargs)

        return ds

    def _merge_datasets(self, datasets_cfg: Dict[str, DatasetConfig]) -> Dataset:
        """Concatenate multiple HF datasets into one."""
        datasets = [self._load_dataset(cfg) for cfg in datasets_cfg.values()]
        return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    def setup(self, stage: Optional[str] = None):
        """Load datasets and prepare them for PyTorch."""
        if stage in (None, "fit"):
            self.train_dataset = self._merge_datasets(self.config.train_datasets)

            if self.config.val_datasets:
                self.val_datasets = {
                    name: self._load_dataset(cfg)
                    for name, cfg in self.config.val_datasets.items()
                }

        if stage in (None, "test") and self.config.test_datasets:
            self.test_datasets = {
                name: self._load_dataset(cfg)
                for name, cfg in self.config.test_datasets.items()
            }

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        if self.train_dataset is None:
            raise RuntimeError(
                "train_dataset is None. Make sure setup() has been called with stage='fit' or stage=None."
            )
        return DataLoader(self.train_dataset, **self.config.train_dataloader_kwargs)

    def val_dataloader(self) -> Optional[Dict[str, DataLoader]]:
        """Create validation DataLoaders."""
        if self.val_datasets is None:
            return None

        if not self.val_datasets:
            raise RuntimeError(
                "val_datasets is empty. Make sure setup() has been called with stage='fit' or stage=None."
            )

        return {
            name: DataLoader(dataset, **self.config.val_dataloader_kwargs)
            for name, dataset in self.val_datasets.items()
        }

    def test_dataloader(self) -> Optional[Dict[str, DataLoader]]:
        """Create test DataLoaders."""
        if self.test_datasets is None:
            return None

        if not self.test_datasets:
            raise RuntimeError(
                "test_datasets is empty. Make sure setup() has been called with stage='test' or stage=None."
            )

        return {
            name: DataLoader(dataset, **self.config.test_dataloader_kwargs)
            for name, dataset in self.test_datasets.items()
        }
