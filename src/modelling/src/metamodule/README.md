# MetaModule

A general PyTorch Lightning module that wraps any model and provides a unified interface for training with configurable losses.

## Overview

The `MetaModule` acts as the main training endpoint, wrapping any model and handling:
- Forward pass through the wrapped model
- Loss computation via `LossManager`
- Optimizer and scheduler configuration
- Standardized output format for callbacks

## Key Features

- **Model Wrapping**: Wraps any PyTorch model via import path
- **Configurable Losses**: Losses are configured via `LossManager` with flexible argument mapping
- **Separation of Concerns**: Losses handled in module, metrics handled via callbacks
- **Standardized Output**: Returns `StepOutput` TypedDict for consistent callback interface

## Architecture

### StepOutput

All step methods return a standardized `StepOutput` format:

```python
class StepOutput(TypedDict):
    outputs: Dict[str, Any]  # Model forward outputs
    loss: Optional[torch.Tensor]  # Total loss (required for training)
```

This allows callbacks (like `MetricsCallback`) to consume outputs consistently.

### Loss Computation

Losses are computed via `LossManager` during each step:
- Uses flexible argument mapping (`batch_key_mapping`, `output_key_mapping`)
- Supports weighted combination of multiple losses
- Returns individual loss values and aggregated total loss

### Model Interface

The wrapped model's forward method should accept keyword arguments from the batch and return a dictionary:

```python
def forward(self, **kwargs) -> Dict[str, Any]:
    # Process inputs
    # Return dict with outputs and any intermediate values
    return {
        "logits": logits,
        "embeddings": embeddings,
        # ... other outputs needed for losses/metrics
    }
```

## Configuration

### MetaModuleConfig

- `model_class_path`: Import path to the model class (required)
- `model_kwargs`: Optional arguments for model initialization
- `loss_manager`: `LossManagerConfig` containing list of loss configurations
- `optimizer`: `OptimizerConfig` for optimizer configuration
- `scheduler`: Optional `SchedulerConfig` for learning rate scheduler

### LossManagerConfig

- `losses`: List of `LossConfig` objects

### LossConfig

- `loss_class_path`: Import path to loss class/function (required)
- `loss_kwargs`: Optional arguments for loss initialization
- `weight`: Weight for this loss in total loss (default: 1.0)
- `name`: Name for this loss (required)
- `batch_key_mapping`: Maps batch keys to loss function arguments (required, can be empty dict)
- `output_key_mapping`: Maps output keys to loss function arguments (required, can be empty dict)

## Usage Example

```python
from metamodule import MetaModule, MetaModuleConfig, LossConfig, LossManagerConfig
from metamodule.utils.lightning import OptimizerConfig, SchedulerConfig

config = MetaModuleConfig(
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
            LossConfig(
                loss_class_path="torch.nn.KLDivLoss",
                name="kl_divergence",
                output_key_mapping={"mean": "input", "log_std": "log_target"},
                weight=0.001,
            ),
        ]
    ),
    optimizer=OptimizerConfig(
        optimizer_class_path="torch.optim.Adam",
        optimizer_kwargs={"lr": 1e-4}
    ),
    scheduler=SchedulerConfig(
        scheduler_class_path="torch.optim.lr_scheduler.CosineAnnealingLR",
        scheduler_kwargs={"T_max": 100}
    ),
)

module = MetaModule(config)
```

## Integration with Callbacks

The `StepOutput` format allows callbacks to consume outputs:

```python
from callbacks import MetricsCallback, MetricConfig

# Metrics are handled separately via callbacks
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

trainer = Trainer(callbacks=[metrics_callback])
```

## Related Modules

- **LossManager**: Handles loss initialization, argument mapping, and aggregation
- **MetricsCallback**: Handles metric computation via callbacks
- **argument_mapping**: Shared utilities for mapping batch/output keys to function arguments
