from typing import Dict, List, Optional, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, Dataset
from pydantic import BaseModel, Field, ConfigDict


class HFDatasetConfig(BaseModel):
    """Configuration for loading a Hugging Face dataset."""

    path: str = Field(..., description="Hugging Face dataset name or local path")
    name: Optional[str] = Field(None, description="Dataset configuration name")
    split: Optional[str] = Field(
        None, description="Dataset split to load (train/validation/test)"
    )
    data_files: Optional[Union[str, List[str], Dict[str, str]]] = Field(
        None, description="Paths to source data files"
    )
    cache_dir: Optional[str] = Field(None, description="Cache directory")
    streaming: Optional[bool] = Field(None, description="Enable streaming mode")
    num_proc: Optional[int] = Field(None, description="Number of processes for loading")

    model_config = ConfigDict(extra="allow")


class HFDatasetItemConfig(BaseModel):
    """Configuration for a named dataset item."""

    name: str = Field(..., description="Name for this dataset")
    cfg: HFDatasetConfig = Field(..., description="Dataset configuration")


class DataLoaderConfig(BaseModel):
    """Configuration for PyTorch DataLoader."""

    batch_size: Optional[int] = Field(None, description="Batch size")
    num_workers: Optional[int] = Field(
        None, description="Number of data loading workers"
    )
    shuffle: Optional[bool] = Field(None, description="Shuffle dataset")
    pin_memory: Optional[bool] = Field(None, description="Pin memory for GPU transfer")
    drop_last: Optional[bool] = Field(None, description="Drop last incomplete batch")

    model_config = ConfigDict(extra="allow")


class SequenceDataModuleConfig(BaseModel):
    """Configuration for the SequenceDataModule."""

    train_datasets: List[HFDatasetItemConfig] = Field(
        ..., description="Training datasets with custom names"
    )
    val_datasets: Optional[List[HFDatasetItemConfig]] = Field(
        None, description="Validation datasets with custom names"
    )
    test_datasets: Optional[List[HFDatasetItemConfig]] = Field(
        None, description="Test datasets with custom names"
    )
    train_dataloader: Optional[DataLoaderConfig] = Field(
        None, description="Training DataLoader configuration"
    )
    val_dataloader: Optional[DataLoaderConfig] = Field(
        None, description="Validation DataLoader configuration"
    )
    test_dataloader: Optional[DataLoaderConfig] = Field(
        None, description="Test DataLoader configuration"
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

    def _load_dataset(self, cfg: HFDatasetConfig) -> Dataset:
        """Load one dataset or split from Hugging Face."""
        ds = load_dataset(**cfg.model_dump(exclude_none=True))
        if isinstance(ds, dict):
            if cfg.split is None:
                raise ValueError(
                    "Dataset has multiple splits but no split specified in config"
                )
            ds = ds[cfg.split]
        return ds

    def _merge_datasets(self, cfgs: List[HFDatasetItemConfig]) -> Dataset:
        """Concatenate multiple HF datasets into one."""
        datasets = [self._load_dataset(item.cfg) for item in cfgs]
        return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]

    def setup(self, stage: Optional[str] = None):
        """Load datasets and prepare them for PyTorch."""
        if stage in (None, "fit"):
            self.train_dataset = self._merge_datasets(self.config.train_datasets)

            if self.config.val_datasets:
                self.val_datasets = {
                    item.name: self._load_dataset(item.cfg)
                    for item in self.config.val_datasets
                }

        if stage in (None, "test") and self.config.test_datasets:
            self.test_datasets = {
                item.name: self._load_dataset(item.cfg)
                for item in self.config.test_datasets
            }

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        dl_kwargs = (
            self.config.train_dataloader.model_dump(exclude_none=True)
            if self.config.train_dataloader
            else {}
        )
        return DataLoader(self.train_dataset, **dl_kwargs)

    def val_dataloader(self) -> Optional[Dict[str, DataLoader]]:
        """Create validation DataLoaders."""
        if self.val_datasets is None:
            return None

        dl_kwargs = (
            self.config.val_dataloader.model_dump(exclude_none=True)
            if self.config.val_dataloader
            else {}
        )
        # Override shuffle for validation
        dl_kwargs.update(shuffle=False)

        return {
            name: DataLoader(dataset, **dl_kwargs)
            for name, dataset in self.val_datasets.items()
        }

    def test_dataloader(self) -> Optional[Dict[str, DataLoader]]:
        """Create test DataLoaders."""
        if self.test_datasets is None:
            return None

        dl_kwargs = (
            self.config.test_dataloader.model_dump(exclude_none=True)
            if self.config.test_dataloader
            else {}
        )
        # Override shuffle and drop_last for test
        dl_kwargs.update(shuffle=False, drop_last=False)

        return {
            name: DataLoader(dataset, **dl_kwargs)
            for name, dataset in self.test_datasets.items()
        }
