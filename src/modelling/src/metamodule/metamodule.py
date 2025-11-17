"""
MetaModule: A general PyTorch Lightning module acting as the main training endpoint.

This module wraps any model and provides a unified interface for training with
configurable losses. Losses are computed during training/validation steps.

Metrics should be handled by callbacks that consume the standardized output format
returned from training_step/validation_step/test_step.
"""

from typing import Any, Dict, Optional, TypedDict
from pytorch_lightning import LightningModule
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

        self.model: Module = get_obj_from_import_path(config.model_class_path)(
            **config.model_kwargs
        )

        self.loss_manager = LossManager(config.loss_manager.losses)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> StepOutput:
        outputs = self.model(**batch)
        loss_dict = self.loss_manager.compute_losses(outputs, batch, self.device)

        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return StepOutput(outputs=outputs, loss=loss_dict["loss"])

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> StepOutput:
        outputs = self.model(**batch)
        loss_dict = self.loss_manager.compute_losses(outputs, batch, self.device)

        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return StepOutput(outputs=outputs)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> StepOutput:
        outputs = self.model(**batch)
        loss_dict = self.loss_manager.compute_losses(outputs, batch, self.device)

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
