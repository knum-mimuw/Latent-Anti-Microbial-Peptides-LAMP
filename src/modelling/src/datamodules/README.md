# DataModules

PyTorch Lightning data modules for loading and preparing datasets.

## SequenceDataModule

A data module for handling sequence datasets, particularly from Hugging Face datasets.

### Features

- **Hugging Face Integration**: Load datasets from Hugging Face Hub or local paths
- **Multiple Datasets**: Support for combining multiple datasets
- **Flexible Configuration**: Configure train/val/test splits independently
- **DataLoader Configuration**: Customizable batch size, workers, etc.

### Configuration

#### SequenceDataModuleConfig

- `train_datasets`: List of `HFDatasetItemConfig` for training data
- `val_datasets`: Optional list of `HFDatasetItemConfig` for validation data
- `test_datasets`: Optional list of `HFDatasetItemConfig` for test data
- `dataloader`: `DataLoaderConfig` for DataLoader settings

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

#### DataLoaderConfig

DataLoader configuration:

- `batch_size`: Batch size (required)
- `num_workers`: Number of worker processes (default: 0)
- `pin_memory`: Pin memory for faster GPU transfer (default: False)
- `shuffle`: Shuffle training data (default: True)
- `drop_last`: Drop last incomplete batch (default: False)
- `**kwargs`: Additional DataLoader arguments

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

