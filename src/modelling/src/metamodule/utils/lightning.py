from typing import Any, Optional, Iterable, Dict
from pydantic import BaseModel
import torch
from ...utils.importing import get_obj_from_import_path


def add_suffix_to_dict_keys(dict: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    return {f"{key}{suffix}": value for key, value in dict.items()}


class OptimizerConfig(BaseModel):
    optimizer_class_path: str
    optimizer_kwargs: Optional[Dict[str, Any]] = None


class SchedulerConfig(BaseModel):
    scheduler_class_path: str
    scheduler_kwargs: Optional[Dict[str, Any]] = None


def configure_optimizers(
    optimizer_config: OptimizerConfig,
    parameters: Iterable[torch.nn.Parameter],
    scheduler_config: Optional[SchedulerConfig] = None,
):
    optimizer_class = get_obj_from_import_path(optimizer_config.optimizer_class_path)
    optimizer = optimizer_class(parameters, **optimizer_config.optimizer_kwargs)
    if scheduler_config:
        scheduler_class = get_obj_from_import_path(
            scheduler_config.scheduler_class_path
        )
        scheduler = scheduler_class(optimizer, **scheduler_config.scheduler_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    else:
        return {"optimizer": optimizer}
