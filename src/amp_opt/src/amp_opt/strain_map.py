"""Strain-name mapping helpers for APEX optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StrainMap:
    """Mapping from canonical challenge strain names to APEX column names."""

    strain_to_apex: dict[str, str | None]

    def mapped_name(self, canonical_name: str) -> str | None:
        """Return mapped APEX strain name, or None when unresolved."""
        return self.strain_to_apex.get(canonical_name)


def load_strain_map(config_path: Path) -> StrainMap:
    """Load strain mapping YAML from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Strain map config must be a dictionary.")
    mapping = raw.get("strain_to_apex")
    if not isinstance(mapping, dict):
        raise ValueError("Strain map config must define 'strain_to_apex' mapping.")

    parsed: dict[str, str | None] = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Strain map keys must be non-empty strings.")
        if value is None:
            parsed[key] = None
        elif isinstance(value, str) and value.strip():
            parsed[key] = value
        else:
            raise ValueError(f"Invalid mapping value for '{key}'.")

    return StrainMap(strain_to_apex=parsed)
