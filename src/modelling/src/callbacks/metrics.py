"""Metrics callback for computing metrics with flexible argument mapping."""

from typing import Any, Dict, List, Optional
from pytorch_lightning.callbacks import Callback
from pydantic import BaseModel, Field
from ..utils.importing import get_obj_from_import_path
from ..metamodule.utils.argument_mapping import prepare_function_args
from ..metamodule.metamodule import StepOutput


class MetricConfig(BaseModel):
    """Configuration for a metric function with computation frequency."""

    metric_class_path: str = Field(
        ..., description="Import path to metric class/function"
    )
    metric_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for metric initialization"
    )
    name: str = Field(
        ...,
        description="Name for this metric (required, allows using same metric function multiple times)",
    )
    # Mapping from batch keys to metric function argument names
    batch_key_mapping: Dict[str, str] = Field(
        ...,
        description="Map batch keys to metric function argument names. "
        "e.g., {'target': 'labels'} maps batch['target'] to metric_fn(labels=...). "
        "Can be empty dict if not using batch data.",
    )
    # Mapping from output keys to metric function argument names
    output_key_mapping: Dict[str, str] = Field(
        ...,
        description="Map output keys to metric function argument names. "
        "e.g., {'logits': 'predictions'} maps outputs['logits'] to metric_fn(predictions=...). "
        "Can be empty dict if not using output data.",
    )
    # Frequency options
    every_n_steps: Optional[int] = Field(
        None, description="Compute every N training steps"
    )
    every_n_epochs: Optional[int] = Field(None, description="Compute every N epochs")
    on_train_epoch_end: bool = Field(
        False, description="Compute at end of each training epoch"
    )
    on_val_epoch_end: bool = Field(
        True, description="Compute at end of each validation epoch"
    )
    on_test_epoch_end: bool = Field(
        True, description="Compute at end of each test epoch"
    )
    stages: List[str] = Field(
        default=["val"], description="Stages to compute metric for (train/val/test)"
    )


class MetricsCallback(Callback):
    """
    Callback for computing metrics with flexible argument mapping.

    Metrics are computed from the standardized output format returned by
    training_step/validation_step/test_step: {"outputs": ...}
    """

    def __init__(self, metric_configs: List[MetricConfig]):
        """
        Initialize metrics callback with metric configurations.

        Args:
            metric_configs: List of MetricConfig objects
        """
        super().__init__()
        self.metrics: Dict[str, Dict[str, Any]] = {}

        for metric_cfg in metric_configs:
            # Get metric function/class from import path
            metric_fn = get_obj_from_import_path(metric_cfg.metric_class_path)

            # Initialize if it's a class, otherwise use directly
            if isinstance(metric_fn, type):
                metric_kwargs = metric_cfg.metric_kwargs or {}
                metric_instance = metric_fn(**metric_kwargs)
            else:
                metric_instance = metric_fn

            # Use the explicitly provided name
            metric_name = metric_cfg.name

            # Store metric configuration
            self.metrics[metric_name] = {
                "fn": metric_instance,
                "config": metric_cfg,
                "batch_key_mapping": metric_cfg.batch_key_mapping,
                "output_key_mapping": metric_cfg.output_key_mapping,
            }

    def _should_compute_metric(
        self, metric_name: str, stage: str, is_epoch_end: bool, trainer
    ) -> bool:
        """Determine if a metric should be computed at this point."""
        metric_config = self.metrics[metric_name]["config"]

        if stage not in metric_config.stages:
            return False

        if is_epoch_end:
            epoch_end_flags = {
                "train": metric_config.on_train_epoch_end,
                "val": metric_config.on_val_epoch_end,
                "test": metric_config.on_test_epoch_end,
            }

            if epoch_end_flags.get(stage, False):
                if metric_config.every_n_epochs:
                    return trainer.current_epoch % metric_config.every_n_epochs == 0
                return True
            return False

        if stage == "train" and metric_config.every_n_steps:
            return trainer.global_step % metric_config.every_n_steps == 0

        return False

    def _compute_metric(
        self, metric_name: str, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute a single metric."""
        metric_info = self.metrics[metric_name]
        metric_args = prepare_function_args(
            outputs=outputs,
            batch=batch,
            batch_key_mapping=metric_info["batch_key_mapping"],
            output_key_mapping=metric_info["output_key_mapping"],
        )
        metric_value = metric_info["fn"](**metric_args)
        return (
            metric_value
            if isinstance(metric_value, dict)
            else {metric_name: metric_value}
        )

    def _process_batch_end(
        self, trainer, outputs: StepOutput, batch: Dict[str, Any], stage: str
    ) -> None:
        """Common logic for processing batch-end metrics."""
        for metric_name in self.metrics.keys():
            if self._should_compute_metric(metric_name, stage, False, trainer):
                metric_values = self._compute_metric(
                    metric_name, outputs["outputs"], batch
                )
                for key, value in metric_values.items():
                    trainer.logger.log_metrics(
                        {f"{stage}/metric/{key}": value}, step=trainer.global_step
                    )

    def on_train_batch_end(
        self, trainer, pl_module, outputs: StepOutput, batch, batch_idx
    ) -> None:
        self._process_batch_end(trainer, outputs, batch, "train")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: StepOutput,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self._process_batch_end(trainer, outputs, batch, "val")

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: StepOutput,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        self._process_batch_end(trainer, outputs, batch, "test")
