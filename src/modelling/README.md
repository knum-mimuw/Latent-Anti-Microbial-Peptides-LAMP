# Modelling Package

A flexible PyTorch Lightning-based framework for training models with configurable losses and metrics.

## Overview

This package provides a unified interface for model training with:
- **MetaModule**: General PyTorch Lightning module that wraps any model
- **LossManager**: Flexible loss computation with argument mapping
- **MetricsCallback**: Configurable metrics computation via callbacks
- **DataModules**: Hugging Face dataset integration
- **Shared Utilities**: Argument mapping, dynamic imports, etc.

## Architecture

### Core Components

1. **MetaModule** (`src/metamodule/`): Main training module that wraps models and handles losses
2. **Callbacks** (`src/callbacks/`): PyTorch Lightning callbacks for metrics
3. **DataModules** (`src/datamodules/`): Data loading and preparation
4. **Utils** (`src/utils/`): Shared utilities for imports and model loading

### Design Principles

- **Separation of Concerns**: Losses handled in module, metrics handled in callbacks
- **Flexible Argument Mapping**: Map batch/output keys to function arguments
- **Dynamic Loading**: Load models, losses, and metrics from import paths
- **Standardized Output**: Consistent `StepOutput` format for callbacks

## Quick Start

### Basic Usage

```python
from metamodule import MetaModule, MetaModuleConfig, LossConfig, LossManagerConfig
from metamodule.utils.lightning import OptimizerConfig
from callbacks import MetricsCallback, MetricConfig
from datamodules import SequenceDataModule, SequenceDataModuleConfig

# Configure model
model_config = MetaModuleConfig(
    model_class_path="models.aes.simpletons.base.VAE",
    model_kwargs={"config": vae_config},
    loss_manager=LossManagerConfig(
        losses=[
            LossConfig(
                loss_class_path="torch.nn.MSELoss",
                name="reconstruction",
                batch_key_mapping={"target": "target"},
                output_key_mapping={"reconstruction": "input"},
            ),
        ]
    ),
    optimizer=OptimizerConfig(
        optimizer_class_path="torch.optim.Adam",
        optimizer_kwargs={"lr": 1e-4}
    ),
)

# Create module
module = MetaModule(model_config)

# Configure metrics callback
metrics_callback = MetricsCallback(
    metric_configs=[
        MetricConfig(
            metric_class_path="torchmetrics.Accuracy",
            name="accuracy",
            batch_key_mapping={"target": "target"},
            output_key_mapping={"logits": "preds"},
            stages=["val"],
        ),
    ]
)

# Configure data
data_config = SequenceDataModuleConfig(
    train_datasets=[...],
    dataloader=DataLoaderConfig(batch_size=32),
)

# Train
trainer = Trainer(callbacks=[metrics_callback])
trainer.fit(module, SequenceDataModule(data_config))
```

## Module Structure

```
src/modelling/src/
├── metamodule/          # Core training module
│   ├── metamodule.py    # MetaModule and StepOutput
│   ├── loss_manager.py  # Loss computation and management
│   └── utils/          # Utilities (argument mapping, lightning config)
├── callbacks/           # PyTorch Lightning callbacks
│   └── metrics.py      # MetricsCallback
├── datamodules/         # Data loading modules
│   └── seq_dm.py       # SequenceDataModule
├── models/              # Model implementations
│   └── aes/            # Autoencoder models
├── utils/               # Shared utilities
│   └── importing.py    # Dynamic imports and Hugging Face loading
└── compute_numbers/     # Legacy computation functions
```

## Key Concepts

### StepOutput

All step methods return a standardized format:

```python
class StepOutput(TypedDict):
    outputs: Dict[str, Any]  # Model forward outputs
    loss: Optional[torch.Tensor]  # Total loss
```

This allows callbacks to consume outputs consistently.

### Argument Mapping

Both losses and metrics use flexible argument mapping:

```python
# Map batch['target'] -> labels, outputs['logits'] -> predictions
batch_key_mapping={"target": "labels"}
output_key_mapping={"logits": "predictions"}
```

This allows using any loss/metric function without modifying your model outputs.

### Dynamic Loading

Models, losses, and metrics are loaded from import paths:

```python
model_class_path="torch.nn.Linear"
loss_class_path="torch.nn.MSELoss"
metric_class_path="torchmetrics.Accuracy"
```

This enables configuration-driven training without code changes.

## Documentation

- [MetaModule](src/metamodule/README.md): Core training module
- [Callbacks](src/callbacks/README.md): Metrics callback
- [DataModules](src/datamodules/README.md): Data loading
- [Utils](src/utils/README.md): Shared utilities
- [MetaModule Utils](src/metamodule/utils/README.md): Argument mapping and lightning config

## Examples

See the configuration files in `configs/` for complete examples.

