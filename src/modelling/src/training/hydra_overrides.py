"""Flatten YAML override files into Hydra CLI dotlist overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if any(ch in value for ch in " :=\t#'\""):
            return repr(value)
        return value
    return repr(value)


def _flatten(prefix: str, obj: Any) -> list[str]:
    if isinstance(obj, dict):
        out: list[str] = []
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_flatten(path, val))
        return out
    if isinstance(obj, list):
        # Hydra list literal
        inner = ",".join(_format_scalar(x) for x in obj)
        return [f"{prefix}=[{inner}]"]
    return [f"{prefix}={_format_scalar(obj)}"]


def flatten_yaml_file(path: Path) -> list[str]:
    """Load *path* and return Hydra CLI overrides (skip ``defaults`` roots)."""
    cfg = OmegaConf.load(path)
    root = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(root, dict):
        raise TypeError(f"Expected mapping at root of {path}, got {type(root)}")
    skip = {"defaults"}
    pieces: list[str] = []
    for key, val in root.items():
        if key in skip:
            continue
        pieces.extend(_flatten(str(key), val))
    return pieces
