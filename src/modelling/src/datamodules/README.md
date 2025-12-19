# DataModules

PyTorch Lightning data modules for loading and preparing datasets.

## SequenceDataModule

A data module for handling sequence datasets, particularly from Hugging Face datasets.

### Features

- **Hugging Face Integration**: Load datasets from Hugging Face Hub or local paths
- **Multiple Datasets**: Support for combining multiple datasets
- **Flexible Configuration**: Configure train/val/test splits independently
- **DataLoader Configuration**: Customizable batch size, workers, etc.
- **Built-in Preprocessing**: Optional tokenization, padding, and tensor conversion

### Configuration

#### SequenceDataModuleConfig

- `train_datasets`: List of `HFDatasetItemConfig` for training data
- `val_datasets`: Optional list of `HFDatasetItemConfig` for validation data
- `test_datasets`: Optional list of `HFDatasetItemConfig` for test data
- `dataloader`: `DataLoaderConfig` for DataLoader settings
- `preprocessing`: Optional `SequencePreprocessingConfig` to tokenize and pad sequences

#### HFDatasetConfig

Configuration for loading a Hugging Face dataset:

- `path`: Hugging Face dataset name or local path (required)
- `name`: Dataset configuration name (optional)
- `split`: Dataset split to load (train/validation/test) (optional)
- `data_files`: Paths to source data files (optional)
- `cache_dir`: Cache directory (optional)
- `streaming`: Enable streaming mode (optional)
- `num_proc`: Number of processes for loading (optional)

#### HFDatasetItemConfig

Named dataset item:

- `name`: Name for this dataset (required)
- `cfg`: `HFDatasetConfig` for dataset configuration (required)
- `num_examples`: Optional number of examples in the split; enables correct tqdm totals for streaming datasets

#### DataLoaderConfig

DataLoader configuration:

- `batch_size`: Batch size (required)
- `num_workers`: Number of worker processes (default: 0)
- `pin_memory`: Pin memory for faster GPU transfer (default: False)
- `shuffle`: Shuffle training data (default: True)
- `drop_last`: Drop last incomplete batch (default: False)
- `**kwargs`: Additional DataLoader arguments

#### SequencePreprocessingConfig

Sequence preprocessing configuration:

- `sequence_field`: Column containing raw sequences (default: `sequence`)
- `target_field`: Optional column for target sequences; if omitted, targets reuse inputs
- `input_ids_key` / `target_key`: Keys for tokenized tensors (defaults: `input_ids`, `target`)
- `vocab`: Mapping from tokens to integer IDs
- `max_length`: Fixed length to pad or truncate sequences
- `pad_token_id` / `unk_token_id`: IDs for padding and unknown tokens
- `padding_side`: Pad on `right` (default) or `left`
- `remove_columns`: Drop original columns after tokenization (default: True)
- `num_proc`: Optional parallel workers for `dataset.map`

Embedding-based models (e.g. GRU/Transformer token embeddings) typically expect an `input_ids` tensor. To produce it from raw `sequence` strings, enable `preprocessing` and provide a `vocab`.

### Usage Example

```python
from datamodules import SequenceDataModule, SequenceDataModuleConfig

config = SequenceDataModuleConfig(
    train_datasets=[
        HFDatasetItemConfig(
            name="train",
            cfg=HFDatasetConfig(
                path="my_dataset",
                split="train",
            ),
        ),
    ],
    val_datasets=[
        HFDatasetItemConfig(
            name="val",
            cfg=HFDatasetConfig(
                path="my_dataset",
                split="validation",
            ),
        ),
    ],
    dataloader=DataLoaderConfig(
        batch_size=32,
        num_workers=4,
        shuffle=True,
    ),
)

dm = SequenceDataModule(config)
trainer.fit(model, dm)
```

### Preprocessing Example

```python
from datamodules import (
    SequenceDataModule,
    SequenceDataModuleConfig,
    SequencePreprocessingConfig,
    HFDatasetItemConfig,
    HFDatasetConfig,
    DataLoaderConfig,
)

aa_vocab = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

config = SequenceDataModuleConfig(
    train_datasets=[HFDatasetItemConfig(name="train", cfg=HFDatasetConfig(path="...", split="train"))],
    train_dataloader=DataLoaderConfig(batch_size=32),
    # Recommended special tokens:
    # PAD=0 (ignored in CE), UNK=1, BOS=2, EOS=3, amino acids from 4..
    preprocessing=SequencePreprocessingConfig(
        vocab=aa_vocab,
        max_length=256,
        pad_token_id=0,
        unk_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        decoder_input_key="input",
    ),
)

dm = SequenceDataModule(config)
```

### Multiple Datasets

You can combine multiple datasets:

```python
config = SequenceDataModuleConfig(
    train_datasets=[
        HFDatasetItemConfig(
            name="dataset1",
            cfg=HFDatasetConfig(path="dataset1", split="train"),
        ),
        HFDatasetItemConfig(
            name="dataset2",
            cfg=HFDatasetConfig(path="dataset2", split="train"),
        ),
    ],
    # ...
)
```

The data module will concatenate datasets with the same split.
