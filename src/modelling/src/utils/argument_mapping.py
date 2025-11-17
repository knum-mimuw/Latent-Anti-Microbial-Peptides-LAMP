"""Shared utilities for argument mapping from batch/output keys to function arguments."""

from typing import Any, Dict


def _map_keys(
    source: Dict[str, Any],
    mapping: Dict[str, str],
    source_name: str,
) -> Dict[str, Any]:
    """Map keys from source dictionary to function arguments."""
    result = {}
    for source_key, arg_name in mapping.items():
        if source_key in source:
            result[arg_name] = source[source_key]
        else:
            raise KeyError(
                f"{source_name} key '{source_key}' not found. "
                f"Available keys: {list(source.keys())}"
            )
    return result


def prepare_function_args(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    batch_key_mapping: Dict[str, str],
    output_key_mapping: Dict[str, str],
) -> Dict[str, Any]:
    """
    Prepare arguments for function call by mapping keys.

    Args:
        outputs: Model outputs dictionary
        batch: Batch data dictionary
        batch_key_mapping: Maps batch keys to function argument names
        output_key_mapping: Maps output keys to function argument names

    Returns:
        Dictionary of arguments to pass to function
    """
    function_args = {}
    function_args.update(_map_keys(batch, batch_key_mapping, "Batch"))
    function_args.update(_map_keys(outputs, output_key_mapping, "Output"))
    return function_args
