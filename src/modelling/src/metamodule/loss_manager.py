"""Loss management module for initializing and computing losses with flexible argument mapping."""

from typing import Any, Dict, List, Optional
import torch
from pydantic import BaseModel, Field
from ..utils.importing import get_obj_from_import_path
from .utils.argument_mapping import prepare_function_args


class LossConfig(BaseModel):
    """Configuration for a loss function."""

    loss_class_path: str = Field(..., description="Import path to loss class/function")
    loss_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Arguments for loss initialization"
    )
    weight: float = Field(1.0, description="Weight for this loss in total loss")
    name: str = Field(
        ...,
        description="Name for this loss (required, allows using same loss function multiple times)",
    )
    # Mapping from batch keys to loss function argument names
    batch_key_mapping: Dict[str, str] = Field(
        ...,
        description="Map batch keys to loss function argument names. "
        "e.g., {'target': 'labels'} maps batch['target'] to loss_fn(labels=...). "
        "Can be empty dict if not using batch data.",
    )
    # Mapping from output keys to loss function argument names
    output_key_mapping: Dict[str, str] = Field(
        ...,
        description="Map output keys to loss function argument names. "
        "e.g., {'logits': 'predictions'} maps outputs['logits'] to loss_fn(predictions=...). "
        "Can be empty dict if not using output data.",
    )


class LossManagerConfig(BaseModel):
    """Configuration for the LossManager containing a list of loss configurations."""

    losses: List[LossConfig] = Field(
        default_factory=list, description="List of loss configurations"
    )


class LossManager:
    """
    Manages loss function initialization, argument mapping, and aggregation.

    Handles:
    - Initializing losses from import paths
    - Mapping batch/output keys to loss function arguments
    - Computing losses with proper argument passing
    - Aggregating losses with weights
    - Returning values for logging
    """

    def __init__(self, loss_configs: List[LossConfig]):
        """
        Initialize loss manager with loss configurations.

        Args:
            loss_configs: List of LossConfig objects
        """
        self.losses: Dict[str, Dict[str, Any]] = {}

        for loss_cfg in loss_configs:
            # Get loss function/class from import path
            loss_fn = get_obj_from_import_path(loss_cfg.loss_class_path)

            # Initialize if it's a class, otherwise use directly
            if isinstance(loss_fn, type):
                loss_kwargs = loss_cfg.loss_kwargs or {}
                loss_instance = loss_fn(**loss_kwargs)
            else:
                loss_instance = loss_fn

            # Use the explicitly provided name
            loss_name = loss_cfg.name

            # Store loss configuration
            self.losses[loss_name] = {
                "fn": loss_instance,
                "weight": loss_cfg.weight,
                "batch_key_mapping": loss_cfg.batch_key_mapping,
                "output_key_mapping": loss_cfg.output_key_mapping,
            }

    def compute_losses(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all configured losses with argument mapping and aggregation.

        Args:
            outputs: Model outputs dictionary
            batch: Batch data dictionary

        Returns:
            Dictionary containing:
            - Individual loss values (keyed by loss name)
            - "loss": aggregated total loss (for backprop)
        """
        loss_dict = {}

        for loss_name, loss_info in self.losses.items():
            loss_args = prepare_function_args(
                outputs=outputs,
                batch=batch,
                batch_key_mapping=loss_info["batch_key_mapping"],
                output_key_mapping=loss_info["output_key_mapping"],
            )
            loss_value: torch.Tensor = loss_info["fn"](**loss_args)
            loss_dict[loss_name] = loss_value

        loss_dict["loss"] = torch.stack(
            [
                loss_dict[loss_name] * loss_info["weight"]
                for loss_name, loss_info in self.losses.items()
            ]
        ).sum()

        return loss_dict
