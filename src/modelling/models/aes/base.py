from pytorch_lightning import LightningModule
from .utils import (
    OptimizerConfig,
    SchedulerConfig,
    add_sfx_to_dkeys,
    configure_optimizers,
)
from typing import Any, Dict, Optional, Tuple
import torch


class AESModel(BaseModel): ...


class AESModuleConfig(BaseModel):
    model_config: AESModel
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig


class AESModule(LightningModule):
    def __init__(
        self,
        config: AESModuleConfig,
    ):
        super().__init__()
        self.cfg = config
        # self.model = config.model_config

    def training_step(self, batch: Any, dataloader_idx: int, batch_idx: int) -> Any:
        out = self.model(batch)
        self.log_dict(
            add_sfx_to_dkeys(out["metrics"], suffix="/train"),
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        return out

    def validation_step(self, batch: Any, dataloader_idx: int, batch_idx: int) -> Any:
        pass

    def test_step(self, batch: Any, dataloader_idx: int, batch_idx: int) -> Any:
        pass

    def predict_step(self, batch: Any, dataloader_idx: int, batch_idx: int) -> Any:
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        return configure_optimizers(
            optimizer_config=self.cfg.optimizer_config,
            parameters=self.parameters(),
            scheduler_config=self.cfg.scheduler_config,
        )
