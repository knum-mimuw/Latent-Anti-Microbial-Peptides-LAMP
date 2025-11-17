"""Functions for computing metrics with frequency control."""

from typing import Any, Dict, List, Optional
import torch


class MetricState:
    """State tracker for metric computation frequencies."""

    def __init__(self):
        self.step_counts: Dict[str, int] = {}
        self.epoch_counts: Dict[str, int] = {}
        self.accumulated_outputs: Dict[str, List[Dict[str, Any]]] = {}
        self.accumulated_batches: Dict[str, List[Dict[str, Any]]] = {}
        self.last_outputs: Dict[str, Dict[str, Any]] = {}
        self.last_batches: Dict[str, Dict[str, Any]] = {}


def should_compute_metric(
    metric_name: str,
    metric_config: Any,
    stage: str,
    is_epoch_end: bool,
    state: MetricState,
) -> bool:
    """Determine if a metric should be computed at this point."""
    # Check if metric is enabled for this stage
    if stage not in metric_config.stages:
        return False

    # Check epoch-based frequencies
    if is_epoch_end:
        key = f"{metric_name}_{stage}"
        epoch = state.epoch_counts.get(key, 0)

        if stage == "train" and metric_config.on_train_epoch_end:
            if metric_config.every_n_epochs:
                if epoch % metric_config.every_n_epochs == 0:
                    state.epoch_counts[key] = epoch + 1
                    return True
            else:
                return True

        elif stage == "val" and metric_config.on_val_epoch_end:
            if metric_config.every_n_epochs:
                if epoch % metric_config.every_n_epochs == 0:
                    state.epoch_counts[key] = epoch + 1
                    return True
            else:
                return True

        elif stage == "test" and metric_config.on_test_epoch_end:
            if metric_config.every_n_epochs:
                if epoch % metric_config.every_n_epochs == 0:
                    state.epoch_counts[key] = epoch + 1
                    return True
            else:
                return True

    # Check step-based frequencies (only for training)
    if stage == "train" and not is_epoch_end and metric_config.every_n_steps:
        key = f"{metric_name}_{stage}"
        step = state.step_counts.get(key, 0)
        if step % metric_config.every_n_steps == 0:
            state.step_counts[key] = step + 1
            return True

    return False


def accumulate_outputs(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    stage: str,
    state: MetricState,
):
    """Accumulate outputs for epoch-end metric computation."""
    if stage not in state.accumulated_outputs:
        state.accumulated_outputs[stage] = []
        state.accumulated_batches[stage] = []

    outputs_copy = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in outputs.items()
    }
    batch_copy = {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    state.accumulated_outputs[stage].append(outputs_copy)
    state.accumulated_batches[stage].append(batch_copy)
    state.last_outputs[stage] = outputs
    state.last_batches[stage] = batch


def get_accumulated_data(
    stage: str, state: MetricState
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Get accumulated outputs and batches for a stage."""
    outputs_list = state.accumulated_outputs.get(stage, [])
    batches_list = state.accumulated_batches.get(stage, [])

    if not outputs_list:
        return {}, {}

    # Concatenate accumulated tensors
    accumulated_outputs = {}
    accumulated_batches = {}

    for key in outputs_list[0].keys():
        values = [out[key] for out in outputs_list if key in out]
        if values and isinstance(values[0], torch.Tensor):
            accumulated_outputs[key] = torch.cat(
                [v.cpu() if v.device != torch.device("cpu") else v for v in values],
                dim=0,
            )
        else:
            accumulated_outputs[key] = values

    for key in batches_list[0].keys():
        values = [batch[key] for batch in batches_list if key in batch]
        if values and isinstance(values[0], torch.Tensor):
            accumulated_batches[key] = torch.cat(
                [v.cpu() if v.device != torch.device("cpu") else v for v in values],
                dim=0,
            )
        else:
            accumulated_batches[key] = values

    return accumulated_outputs, accumulated_batches


def compute_metrics(
    metrics: Dict[str, Dict[str, Any]],
    stage: str,
    is_epoch_end: bool,
    state: MetricState,
    log_fn,
):
    """Compute metrics for the given stage."""
    if is_epoch_end:
        outputs, batch = get_accumulated_data(stage, state)
        if not outputs:
            return
    else:
        outputs = state.last_outputs.get(stage, {})
        batch = state.last_batches.get(stage, {})
        if not outputs:
            return

    # Compute metrics that should be computed at this point
    for metric_name, metric_info in metrics.items():
        metric_config = metric_info["config"]
        if should_compute_metric(
            metric_name, metric_config, stage, is_epoch_end, state
        ):
            try:
                metric_fn = metric_info["fn"]
                metric_values = metric_fn(outputs, batch)

                if not isinstance(metric_values, dict):
                    metric_values = {metric_name: metric_values}

                # Log metrics
                log_dict = {f"{stage}/metric/{k}": v for k, v in metric_values.items()}

                log_fn(
                    log_dict,
                    on_step=not is_epoch_end,
                    on_epoch=is_epoch_end,
                    prog_bar=False,
                    logger=True,
                )
            except Exception as e:
                # Log error but don't crash training
                print(f"Error computing metric {metric_name} at {stage}: {e}")


def clear_accumulation(stage: str, state: MetricState):
    """Clear accumulated data for a stage."""
    state.accumulated_outputs[stage] = []
    state.accumulated_batches[stage] = []
