from typing import Dict, Iterable, List, Optional, Union
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


class SequencePreprocessingConfig(BaseModel):
    """Configuration for sequence tokenization and tensor preparation."""

    sequence_field: str = Field(
        "sequence", description="Field name in the dataset that contains raw sequences"
    )
    target_field: Optional[str] = Field(
        None,
        description=(
            "Optional field containing target sequences. If omitted, targets will reuse"
            " tokenized input IDs when a target key is provided."
        ),
    )
    input_ids_key: str = Field(
        "input_ids", description="Key to store tokenized input IDs"
    )
    target_key: Optional[str] = Field(
        "target", description="Key to store tokenized targets"
    )
    max_length: int = Field(256, description="Fixed sequence length after padding")
    pad_token_id: int = Field(0, description="Token ID used for padding")
    unk_token_id: int = Field(0, description="Token ID used for unknown tokens")
    vocab: Dict[str, int] = Field(
        ..., description="Mapping from tokens (e.g., amino acids) to integer IDs"
    )
    padding_side: str = Field(
        "right", description="Padding direction: 'right' or 'left'", pattern="^(right|left)$"
    )
    remove_columns: bool = Field(
        True,
        description=(
            "Whether to drop original columns after tokenization. Only the new tensor"
            " fields are retained when True."
        ),
    )
    num_proc: Optional[int] = Field(
        None,
        description="Number of processes for dataset.map during preprocessing",
    )


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
    preprocessing: Optional[SequencePreprocessingConfig] = Field(
        None,
        description="Optional preprocessing to tokenize and pad sequences",
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
        self.preprocessing_config = config.preprocessing

    @staticmethod
    def _pad_sequence(
        tokens: List[int],
        max_length: int,
        pad_token_id: int,
        padding_side: str = "right",
    ) -> List[int]:
        """Pad or truncate a token list to a fixed length."""

        tokens = tokens[:max_length]
        pad_len = max_length - len(tokens)
        if pad_len <= 0:
            return tokens

        padding = [pad_token_id] * pad_len
        return tokens + padding if padding_side == "right" else padding + tokens

    def _tokenize_sequence(self, sequence: str) -> List[int]:
        """Convert a raw sequence to token IDs using the configured vocabulary."""

        if self.preprocessing_config is None:
            raise ValueError("Preprocessing config must be set to tokenize sequences")

        vocab = self.preprocessing_config.vocab
        unk_token_id = self.preprocessing_config.unk_token_id
        max_length = self.preprocessing_config.max_length
        pad_token_id = self.preprocessing_config.pad_token_id
        padding_side = self.preprocessing_config.padding_side

        tokens = [vocab.get(token, unk_token_id) for token in sequence]
        return self._pad_sequence(tokens, max_length, pad_token_id, padding_side)

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Apply tokenization, padding, and tensor conversion to a dataset."""

        if self.preprocessing_config is None:
            return dataset

        config = self.preprocessing_config
        output_keys: List[str] = [config.input_ids_key]
        if config.target_key is not None:
            output_keys.append(config.target_key)

        def _tokenize_example(example: Dict[str, str]) -> Dict[str, List[int]]:
            input_tokens = self._tokenize_sequence(example[config.sequence_field])
            example_dict = {config.input_ids_key: input_tokens}

            if config.target_key is not None:
                if config.target_field is None:
                    target_tokens = list(input_tokens)
                else:
                    target_raw = example[config.target_field]
                    if isinstance(target_raw, str):
                        target_tokens = self._tokenize_sequence(target_raw)
                    elif isinstance(target_raw, Iterable):
                        target_tokens = self._pad_sequence(
                            list(target_raw),
                            config.max_length,
                            config.pad_token_id,
                            config.padding_side,
                        )
                    else:
                        raise TypeError(
                            "Target field must be a string or an iterable of token IDs"
                        )

                example_dict[config.target_key] = target_tokens

            return example_dict

        remove_columns = dataset.column_names if config.remove_columns else None

        processed = dataset.map(
            _tokenize_example,
            remove_columns=remove_columns,
            num_proc=config.num_proc,
        )

        return processed.with_format("torch", columns=output_keys)

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
            self.train_dataset = self.prepare_dataset(
                self._merge_datasets(self.config.train_datasets)
            )

            if self.config.val_datasets:
                self.val_datasets = {
                    item.name: self.prepare_dataset(self._load_dataset(item.cfg))
                    for item in self.config.val_datasets
                }

        if stage in (None, "test") and self.config.test_datasets:
            self.test_datasets = {
                item.name: self.prepare_dataset(self._load_dataset(item.cfg))
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
