"""Functions for computing losses."""
from typing import Any, Dict
import torch


def compute_losses(
    losses: Dict[str, Dict[str, Any]],
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute all configured losses."""
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)

    for loss_name, loss_info in losses.items():
        loss_value = loss_info["fn"](outputs, batch)
        if isinstance(loss_value, dict):
            # If loss returns a dict, extract the main loss value
            main_loss = loss_value.get(
                "loss", loss_value.get("total", next(iter(loss_value.values())))
            )
            loss_dict.update({f"{k}_{loss_name}": v for k, v in loss_value.items()})
            loss_value = main_loss
        else:
            loss_dict[f"{loss_name}"] = loss_value

        weighted_loss = loss_value * loss_info["weight"]
        total_loss = total_loss + weighted_loss

    loss_dict["loss"] = total_loss
    return loss_dict







