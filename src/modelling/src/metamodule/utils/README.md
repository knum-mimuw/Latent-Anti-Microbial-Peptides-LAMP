# MetaModule Utils

Utility modules for the MetaModule package.

## argument_mapping.py

Shared utilities for argument mapping from batch/output keys to function arguments.

### `prepare_function_args`

Maps keys from batch and outputs dictionaries to function argument names.

```python
from metamodule.utils.argument_mapping import prepare_function_args

# Map batch['target'] -> labels, outputs['logits'] -> predictions
args = prepare_function_args(
    outputs={"logits": tensor(...)},
    batch={"target": tensor(...)},
    batch_key_mapping={"target": "labels"},
    output_key_mapping={"logits": "predictions"},
)
# Returns: {"labels": tensor(...), "predictions": tensor(...)}
```

**Parameters:**
- `outputs`: Model outputs dictionary
- `batch`: Batch data dictionary
- `batch_key_mapping`: Maps batch keys to function argument names
- `output_key_mapping`: Maps output keys to function argument names

**Returns:** Dictionary of arguments to pass to function

**Raises:** `KeyError` if a mapped key is not found in source dictionary

### Usage

This function is used by:
- `LossManager` to map arguments for loss functions
- `MetricsCallback` to map arguments for metric functions

## lightning.py

Utilities for configuring optimizers and schedulers.

### `OptimizerConfig`

Configuration for an optimizer:

- `optimizer_class_path`: Import path to optimizer class (required)
- `optimizer_kwargs`: Optional arguments for optimizer initialization

### `SchedulerConfig`

Configuration for a learning rate scheduler:

- `scheduler_class_path`: Import path to scheduler class (required)
- `scheduler_kwargs`: Optional arguments for scheduler initialization

### `configure_optimizers`

Configures optimizers and schedulers for PyTorch Lightning.

```python
from metamodule.utils.lightning import configure_optimizers, OptimizerConfig, SchedulerConfig

optimizer_config = OptimizerConfig(
    optimizer_class_path="torch.optim.Adam",
    optimizer_kwargs={"lr": 1e-4}
)

scheduler_config = SchedulerConfig(
    scheduler_class_path="torch.optim.lr_scheduler.CosineAnnealingLR",
    scheduler_kwargs={"T_max": 100}
)

optimizer_dict = configure_optimizers(
    optimizer_config,
    model.parameters(),
    scheduler_config,
)
```

**Parameters:**
- `optimizer_config`: `OptimizerConfig` instance
- `parameters`: Model parameters (iterable)
- `scheduler_config`: Optional `SchedulerConfig` instance

**Returns:** Dictionary with `"optimizer"` and optionally `"lr_scheduler"` keys

## metrics.py

Legacy metrics utilities (deprecated). Metrics are now handled via `MetricsCallback` in the callbacks module.

