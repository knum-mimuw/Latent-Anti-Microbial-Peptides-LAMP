# DataModules

PyTorch Lightning data modules for loading and preparing datasets.

## SequenceDataModule

A data module for handling sequence datasets from Hugging Face.

### Features

- **Hugging Face Integration**: Load datasets from Hugging Face Hub or local paths
- **Multiple Datasets**: Support for combining multiple datasets
- **Flexible Configuration**: Configure train/val/test splits independently
- **Tokenizer Collate**: On-the-fly tokenization via configurable collate function
- **Streaming Support**: Works with HF streaming datasets

### Configuration

#### SequenceDataModuleConfig

- `train_datasets`: Dict of `{name: DatasetConfig}` for training data
- `val_datasets`: Optional dict of `{name: DatasetConfig}` for validation data
- `test_datasets`: Optional dict of `{name: DatasetConfig}` for test data
- `collate`: Optional `CollateConfig` for tokenization/batching
- `train_dataloader_kwargs`: DataLoader kwargs for training
- `val_dataloader_kwargs`: DataLoader kwargs for validation
- `test_dataloader_kwargs`: DataLoader kwargs for testing

#### DatasetConfig

Configuration for a single dataset:

- `hf_kwargs`: Arguments passed directly to `load_dataset()` (path, split, streaming, etc.)
- `shuffle_kwargs`: Optional arguments for `.shuffle()` (buffer_size for streaming)

#### CollateConfig

Configuration for a collate function:

- `class_path`: Import path to collate class
- `config_class_path`: Import path to collate config class
- `config`: Dict of config arguments

#### TokenizerCollateConfig

Configuration for the tokenizer collate function:

- `tokenizer_path`: HuggingFace tokenizer path or local path (required)
- `sequence_column`: Column containing sequences (default: `"sequence"`)
- `tokenizer_kwargs`: Dict of kwargs passed to `tokenizer()` call (default: `{padding: "longest", return_tensors: "pt"}`)
- `preserve_columns`: List of columns to preserve from original batch (default: `[]`)

### YAML Configuration Example

```yaml
data:
  class_path: modelling.src.datamodules.seq_dm.SequenceDataModule
  init_args:
    config:
      collate:
        class_path: modelling.src.datamodules.collate.TokenizerCollate
        config_class_path: modelling.src.datamodules.collate.TokenizerCollateConfig
        config:
          tokenizer_path: pszmk/protein-aa-fast-tokenizer
          sequence_column: sequence
          tokenizer_kwargs:
            padding: longest
            max_length: 52
            truncation: true
            return_tensors: pt

      train_datasets:
        train:
          hf_kwargs:
            path: "pszmk/LAMP-datasets"
            split: "train"

      val_datasets:
        val:
          hf_kwargs:
            path: "pszmk/LAMP-datasets"
            split: "validation"

      train_dataloader_kwargs:
        batch_size: 32
        num_workers: 4
        pin_memory: true
        shuffle: true
```

### Batch Output

With `TokenizerCollate`, batches will contain:

- `input_ids`: Tokenized sequences as PyTorch tensor `[batch_size, seq_len]`
- `attention_mask`: Attention mask tensor `[batch_size, seq_len]`
- Any columns specified in `preserve_columns`

### Multiple Datasets

Combine multiple datasets by adding more entries:

```yaml
train_datasets:
  dataset1:
    hf_kwargs:
      path: "dataset1"
      split: "train"
  dataset2:
    hf_kwargs:
      path: "dataset2"
      split: "train"
```

Datasets are concatenated during setup.

### Streaming Mode

For large datasets, enable streaming:

```yaml
train_datasets:
  train:
    hf_kwargs:
      path: "large_dataset"
      split: "train"
      streaming: true
    shuffle_kwargs:
      buffer_size: 10000
```

Note: Use `num_workers: 0` with streaming datasets.
