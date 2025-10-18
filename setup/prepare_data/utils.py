import json
from pathlib import Path
from typing import Dict, Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    yaml = None

SequenceItem = Dict[str, Any]


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON config file into a dictionary.

    Supports .yaml/.yml (requires PyYAML) and .json.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to read YAML configs")
        return yaml.safe_load(config_path.read_text()) or {}
    if suffix == ".json":
        return json.loads(config_path.read_text())
    raise ValueError("Unsupported config format. Use .yaml/.yml or .json")
