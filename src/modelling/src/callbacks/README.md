# Callbacks Module

This module contains PyTorch Lightning callbacks for handling metrics computation.

## MetricsCallback

The `MetricsCallback` computes metrics with flexible argument mapping and configurable computation frequencies.

### Key Features

- **Flexible Argument Mapping**: Maps batch and output keys to metric function arguments
- **Configurable Frequencies**: Compute metrics every N steps or N epochs
- **Stage Control**: Configure which stages (train/val/test) to compute metrics for
- **Shared Logic**: Uses the same argument mapping logic as `LossManager` for consistency

### Architecture

The callback consumes `StepOutput` from `training_step`/`validation_step`/`test_step`:
```python
class StepOutput(TypedDict):
    outputs: Dict[str, Any]  # Model forward outputs
    loss: Optional[torch.Tensor]  # Total loss
```

### MetricConfig

Configuration for a metric function:

- `metric_class_path`: Import path to metric class/function (required)
- `metric_kwargs`: Optional arguments for metric initialization
- `name`: Name for this metric (required, allows using same metric function multiple times)
- `batch_key_mapping`: Maps batch keys to metric function argument names (required, can be empty dict)
- `output_key_mapping`: Maps output keys to metric function argument names (required, can be empty dict)
- `every_n_steps`: Compute every N training steps (optional)
- `every_n_epochs`: Compute every N epochs (optional)
- `on_train_epoch_end`: Compute at end of each training epoch (default: False)
- `on_val_epoch_end`: Compute at end of each validation epoch (default: True)
- `on_test_epoch_end`: Compute at end of each test epoch (default: True)
- `stages`: List of stages to compute for (default: ["val"])

### Argument Mapping

Metrics use the same argument mapping system as losses. The mapping allows you to:
- Map batch keys (e.g., `batch['target']`) to metric function arguments (e.g., `labels`)
- Map output keys (e.g., `outputs['logits']`) to metric function arguments (e.g., `predictions`)

Example:
```python
MetricConfig(
    metric_class_path="torchmetrics.Accuracy",
    name="accuracy",
    batch_key_mapping={"target": "target"},  # batch['target'] -> metric_fn(target=...)
    output_key_mapping={"logits": "preds"},  # outputs['logits'] -> metric_fn(preds=...)
    stages=["val", "test"],
    on_val_epoch_end=True,
)
```

### Usage Example

```python
from callbacks import MetricsCallback, MetricConfig

metrics_callback = MetricsCallback(
    metric_configs=[
        MetricConfig(
            metric_class_path="torchmetrics.Accuracy",
            metric_kwargs={"task": "multiclass", "num_classes": 10},
            name="accuracy",
            batch_key_mapping={"target": "target"},
            output_key_mapping={"logits": "preds"},
            stages=["val", "test"],
            on_val_epoch_end=True,
        ),
        MetricConfig(
            metric_class_path="torchmetrics.F1Score",
            metric_kwargs={"task": "multiclass", "num_classes": 10},
            name="f1",
            batch_key_mapping={"target": "target"},
            output_key_mapping={"logits": "preds"},
            stages=["val"],
            every_n_epochs=2,  # Compute every 2 epochs
        ),
    ]
)

trainer = Trainer(callbacks=[metrics_callback])
```

### How It Works

1. **Initialization**: Loads metric functions from import paths and stores configurations
2. **Batch End**: On each batch end, checks if metrics should be computed based on frequency settings
3. **Argument Mapping**: Uses `prepare_function_args()` to map batch/output keys to metric function arguments
4. **Computation**: Calls metric function with mapped arguments
5. **Logging**: Logs metric values with stage prefix (e.g., `val/metric/accuracy`)

### Integration with MetaModule

The callback automatically works with `MetaModule` since it returns `StepOutput`:
```python
from metamodule import MetaModule, MetaModuleConfig
from callbacks import MetricsCallback, MetricConfig

module = MetaModule(config)
metrics_callback = MetricsCallback([...])
trainer = Trainer(callbacks=[metrics_callback])
```

