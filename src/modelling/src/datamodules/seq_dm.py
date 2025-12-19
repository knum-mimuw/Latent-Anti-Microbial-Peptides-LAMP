from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Union
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as TorchIterableDataset
from datasets import load_dataset, concatenate_datasets, Dataset
from pydantic import BaseModel, Field, ConfigDict

try:
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - optional import for older/newer datasets versions
    HFIterableDataset = None


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
    num_examples: Optional[int] = Field(
        None,
        description=(
            "Optional number of examples in this split. Useful for Hugging Face streaming datasets "
            "(IterableDataset) where tqdm/Lightning can't infer an epoch length automatically."
        ),
        ge=1,
    )


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
    decoder_input_key: Optional[str] = Field(
        None,
        description=(
            "Optional key to store shifted decoder inputs for teacher forcing (e.g. 'input'). "
            "When set, a decoder input sequence will be created as [BOS] + target[:-1]."
        ),
    )
    max_length: int = Field(256, description="Fixed sequence length after padding")
    pad_token_id: int = Field(0, description="Token ID used for padding")
    unk_token_id: int = Field(0, description="Token ID used for unknown tokens")
    bos_token_id: Optional[int] = Field(
        None,
        description=(
            "Optional BOS token ID used when creating shifted decoder inputs. "
            "If omitted, pad_token_id is used as BOS."
        ),
    )
    eos_token_id: Optional[int] = Field(
        None,
        description=(
            "Optional EOS token ID appended to sequences before padding/truncation. "
            "Useful when training next-token prediction with shifted teacher forcing."
        ),
    )
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
        self.train_num_examples: Optional[int] = None
        self.val_num_examples: Optional[Dict[str, int]] = None
        self.test_num_examples: Optional[Dict[str, int]] = None
        self.preprocessing_config = config.preprocessing

    class _IterableDatasetWithLen(TorchIterableDataset):
        def __init__(self, dataset: TorchIterableDataset, num_examples: int):
            super().__init__()
            self._dataset = dataset
            self._num_examples = int(num_examples)

        def __iter__(self) -> Iterator[object]:
            return iter(self._dataset)

        def __len__(self) -> int:
            return self._num_examples

        def __getattr__(self, item: str):
            return getattr(self._dataset, item)

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
        eos_token_id = self.preprocessing_config.eos_token_id

        tokens = [vocab.get(token, unk_token_id) for token in sequence]
        if eos_token_id is not None:
            if max_length < 1:
                raise ValueError("max_length must be >= 1 when eos_token_id is set")
            tokens = tokens[: max_length - 1] + [int(eos_token_id)]
        return self._pad_sequence(tokens, max_length, pad_token_id, padding_side)

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Apply tokenization, padding, and tensor conversion to a dataset."""

        if self.preprocessing_config is None:
            return dataset

        config = self.preprocessing_config
        output_keys: List[str] = [config.input_ids_key]
        if config.target_key is not None:
            output_keys.append(config.target_key)
        if config.decoder_input_key is not None:
            output_keys.append(config.decoder_input_key)

        def _tokenize_example(example: Dict[str, str]) -> Dict[str, List[int]]:
            input_tokens = self._tokenize_sequence(example[config.sequence_field])
            example_dict: Dict[str, List[int]] = {config.input_ids_key: input_tokens}

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

                if config.decoder_input_key is not None:
                    bos_token_id = (
                        config.pad_token_id
                        if config.bos_token_id is None
                        else int(config.bos_token_id)
                    )
                    if not target_tokens:
                        decoder_input_tokens = []
                    else:
                        decoder_input_tokens = [bos_token_id] + list(target_tokens[:-1])
                    decoder_input_tokens = self._pad_sequence(
                        decoder_input_tokens,
                        config.max_length,
                        config.pad_token_id,
                        config.padding_side,
                    )
                    example_dict[config.decoder_input_key] = decoder_input_tokens

            return example_dict

        remove_columns = dataset.column_names if config.remove_columns else None

        map_kwargs = {"remove_columns": remove_columns}
        if HFIterableDataset is None or not isinstance(dataset, HFIterableDataset):
            map_kwargs["num_proc"] = config.num_proc

        processed = dataset.map(_tokenize_example, **map_kwargs)

        if HFIterableDataset is not None and isinstance(processed, HFIterableDataset):
            return processed.with_format("torch")

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

    @staticmethod
    def _infer_num_examples(dataset: Dataset, split: Optional[str]) -> Optional[int]:
        info = getattr(dataset, "info", None) or getattr(dataset, "_info", None)
        if info is None:
            return None

        splits = getattr(info, "splits", None)
        if not splits:
            return None

        split_name = split
        if split_name is None:
            ds_split = getattr(dataset, "split", None) or getattr(dataset, "_split", None)
            if ds_split is not None:
                split_name = str(ds_split)

        if not split_name:
            return None

        split_info = splits.get(split_name)
        num_examples = getattr(split_info, "num_examples", None) if split_info is not None else None
        if num_examples is None:
            return None

        try:
            num_examples_int = int(num_examples)
        except (TypeError, ValueError):
            return None
        return num_examples_int if num_examples_int > 0 else None

    @classmethod
    def _maybe_wrap_iterable_with_len(cls, dataset: Dataset, num_examples: Optional[int]) -> Dataset:
        if num_examples is None:
            return dataset
        if not isinstance(dataset, TorchIterableDataset):
            return dataset
        try:
            len(dataset)  # type: ignore[arg-type]
        except TypeError:
            return cls._IterableDatasetWithLen(dataset, num_examples)
        return dataset

    def _merge_datasets(self, cfgs: List[HFDatasetItemConfig]) -> tuple[Dataset, Optional[int]]:
        """Concatenate multiple HF datasets into one (and infer its example count when possible)."""
        datasets: List[Dataset] = []
        lengths: List[Optional[int]] = []
        for item in cfgs:
            dataset = self._load_dataset(item.cfg)
            datasets.append(dataset)
            lengths.append(item.num_examples or self._infer_num_examples(dataset, item.cfg.split))

        merged = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
        if any(length is None for length in lengths):
            return merged, None
        return merged, sum(length for length in lengths if length is not None)

    def setup(self, stage: Optional[str] = None):
        """Load datasets and prepare them for PyTorch."""
        if stage in (None, "fit"):
            train_dataset, train_num_examples = self._merge_datasets(self.config.train_datasets)
            self.train_num_examples = train_num_examples
            self.train_dataset = self.prepare_dataset(train_dataset)
            self.train_dataset = self._maybe_wrap_iterable_with_len(
                self.train_dataset, self.train_num_examples
            )

            if self.config.val_datasets:
                self.val_datasets = {}
                self.val_num_examples = {}
                for item in self.config.val_datasets:
                    dataset = self._load_dataset(item.cfg)
                    num_examples = item.num_examples or self._infer_num_examples(
                        dataset, item.cfg.split
                    )
                    dataset = self.prepare_dataset(dataset)
                    dataset = self._maybe_wrap_iterable_with_len(dataset, num_examples)
                    self.val_datasets[item.name] = dataset
                    if num_examples is not None:
                        self.val_num_examples[item.name] = num_examples

        if stage in (None, "test") and self.config.test_datasets:
            self.test_datasets = {}
            self.test_num_examples = {}
            for item in self.config.test_datasets:
                dataset = self._load_dataset(item.cfg)
                num_examples = item.num_examples or self._infer_num_examples(
                    dataset, item.cfg.split
                )
                dataset = self.prepare_dataset(dataset)
                dataset = self._maybe_wrap_iterable_with_len(dataset, num_examples)
                self.test_datasets[item.name] = dataset
                if num_examples is not None:
                    self.test_num_examples[item.name] = num_examples

    @staticmethod
    def _strip_shuffle_for_iterable(dataset: Dataset, dl_kwargs: Dict[str, object]) -> Dict[str, object]:
        """Torch DataLoader forbids `shuffle` when the dataset is iterable/streaming."""

        if isinstance(dataset, TorchIterableDataset) and "shuffle" in dl_kwargs:
            rank_zero_warn(
                "DataLoader received `shuffle=...` but the dataset is an IterableDataset (e.g. "
                "Hugging Face streaming). Dropping `shuffle` to satisfy torch DataLoader constraints."
            )
            dl_kwargs = dict(dl_kwargs)
            dl_kwargs.pop("shuffle", None)
        return dl_kwargs

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        dl_kwargs = (
            self.config.train_dataloader.model_dump(exclude_none=True)
            if self.config.train_dataloader
            else {}
        )
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set; did you forget to call setup('fit')?")
        dl_kwargs = self._strip_shuffle_for_iterable(self.train_dataset, dl_kwargs)
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
        # Override shuffle for validation (and drop it entirely for iterable datasets).
        dl_kwargs.update(shuffle=False)

        return {
            name: DataLoader(dataset, **self._strip_shuffle_for_iterable(dataset, dl_kwargs))
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
            name: DataLoader(dataset, **self._strip_shuffle_for_iterable(dataset, dl_kwargs))
            for name, dataset in self.test_datasets.items()
        }
