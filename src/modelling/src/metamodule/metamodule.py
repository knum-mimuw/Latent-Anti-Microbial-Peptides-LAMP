"""
MetaModule: A general PyTorch Lightning module acting as the main training endpoint.

This module wraps any model and provides a unified interface for training with
configurable losses. Losses are computed during training/validation steps.

Metrics should be handled by callbacks that consume the standardized output format
returned from training_step/validation_step/test_step.
"""

from typing import Any, Dict, Optional, TypedDict
import inspect
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pydantic import BaseModel, Field, ConfigDict
from torch.nn import Module
import torch

from .utils.lightning import OptimizerConfig, SchedulerConfig, configure_optimizers
from .loss_manager import LossManager, LossManagerConfig
from ..utils.importing import get_obj_from_import_path


class StepOutput(TypedDict):
    """Standardized output format from training/validation/test steps."""

    outputs: Dict[str, Any]
    loss: Optional[torch.Tensor]


class MetaModuleConfig(BaseModel):
    """Configuration for MetaModule."""

    model_class_path: str = Field(..., description="Import path to the model class")
    model_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for model initialization"
    )

    loss_manager: LossManagerConfig = Field(
        default_factory=lambda: LossManagerConfig(losses=[]),
        description="Loss manager configuration",
    )

    optimizer: OptimizerConfig = Field(..., description="Optimizer configuration")
    scheduler: Optional[SchedulerConfig] = Field(
        None, description="Scheduler configuration"
    )

    model_config = ConfigDict(extra="allow")


class MetaModule(LightningModule):
    """
    A general PyTorch Lightning module that wraps any model and provides
    configurable losses.

    The forward pass of the wrapped model should return a dictionary containing
    the model outputs and any intermediate values needed for loss computation.
    Losses are computed during training/validation steps.

    Metrics should be handled by callbacks that consume StepOutput from
    training_step/validation_step/test_step.
    """

    def __init__(self, config: MetaModuleConfig):
        super().__init__()
        self.config = config
        self._warned_model_call_fallback = False

        self.model: Module = get_obj_from_import_path(config.model_class_path)(
            **(config.model_kwargs or {})
        )

        self.loss_manager = LossManager(config.loss_manager.losses)

    def _filter_batch_for_model(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Filter batch keys to match the wrapped model's forward signature.

        Hugging Face datasets often yield extra fields (e.g. ids, raw sequences).
        Those should not be passed to the model unless the model accepts **kwargs.
        """
        forward_sig = inspect.signature(self.model.forward)
        params = list(forward_sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return batch
        allowed = {p.name for p in params if p.name != "self"}
        return {k: v for k, v in batch.items() if k in allowed}

    def _run_model(self, batch: Any) -> Dict[str, Any]:
        """Run the wrapped model with a best-effort batch-to-forward mapping.

        - If the batch is a dict and the model's forward signature can be satisfied
          with (a subset of) batch keys, call the model with keyword arguments.
        - Otherwise, pass the batch through as a single positional argument.
        """
        if not isinstance(batch, dict):
            return self.model(batch)

        filtered = self._filter_batch_for_model(batch)

        forward_sig = inspect.signature(self.model.forward)
        if "self" in forward_sig.parameters:
            forward_sig = forward_sig.replace(
                parameters=[
                    p for name, p in forward_sig.parameters.items() if name != "self"
                ]
            )

        try:
            forward_sig.bind(**filtered)
        except TypeError as exc:
            if not self._warned_model_call_fallback:
                self._warned_model_call_fallback = True
                expected = [
                    name
                    for name, p in forward_sig.parameters.items()
                    if p.default is inspect._empty
                ]
                rank_zero_warn(
                    "MetaModule could not call the wrapped model with keyword arguments and is "
                    "falling back to passing the whole batch as a single positional argument. "
                    f"Model forward required params: {expected}. Batch keys: {sorted(batch.keys())}. "
                    f"Filtered keys: {sorted(filtered.keys())}. Original error: {exc}"
                )
            return self.model(batch)

        return self.model(**filtered)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> StepOutput:
        outputs = self._run_model(batch)
        loss_dict = self.loss_manager.compute_losses(outputs, batch)

        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return StepOutput(outputs=outputs, loss=loss_dict["loss"])

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> StepOutput:
        outputs = self._run_model(batch)
        loss_dict = self.loss_manager.compute_losses(outputs, batch)

        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return StepOutput(outputs=outputs)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> StepOutput:
        outputs = self._run_model(batch)
        loss_dict = self.loss_manager.compute_losses(outputs, batch)

        self.log_dict(
            {f"test/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return StepOutput(outputs=outputs)

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        return configure_optimizers(
            self.config.optimizer, self.parameters(), self.config.scheduler
        )
