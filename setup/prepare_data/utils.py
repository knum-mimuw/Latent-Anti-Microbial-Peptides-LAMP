import os
import json
from pathlib import Path
from typing import Generator, Dict, Any, Optional
from datasets import Dataset as HFDataset

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    yaml = None

SequenceItem = Dict[str, Any]


def stream_sequences_to_huggingface(
    sequences_generator: Generator[Dict[str, Any], None, None],
    repo_id: str,
    token: Optional[str] = None,
    commit_message: str = "Add sequences",
    batch_size: int = 1000,
) -> str:
    """
    Stream sequences from a generator directly to Hugging Face without saving locally.

    Args:
        sequences_generator: Generator yielding sequence dictionaries
        repo_id: Hugging Face repository ID (org/repo_name)
        token: Hugging Face token (uses HF_TOKEN env var if None)
        commit_message: Git commit message
        batch_size: Batch size for processing

    Returns:
        Repository URL
    """
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError(
                "Hugging Face token required. Set HF_TOKEN environment variable."
            )

    # Collect sequences from generator
    sequences = list(sequences_generator)

    # Convert to Hugging Face format
    hf_data = {
        "sequences": [seq["sequence"] for seq in sequences],
        "lengths": [seq["length"] for seq in sequences],
    }

    # Add optional fields if present
    if sequences and "accession" in sequences[0]:
        hf_data["accessions"] = [seq["accession"] for seq in sequences]
        hf_data["entry_names"] = [seq["entry_name"] for seq in sequences]
        hf_data["descriptions"] = [seq["description"] for seq in sequences]

    if sequences and "ur50_id" in sequences[0]:
        hf_data["ur50_ids"] = [seq["ur50_id"] for seq in sequences]
        hf_data["ur90_ids"] = [seq["ur90_id"] for seq in sequences]

    # Create and upload dataset
    hf_dataset = HFDataset.from_dict(hf_data)
    hf_dataset.push_to_hub(repo_id=repo_id, token=token, commit_message=commit_message)

    return f"https://huggingface.co/datasets/{repo_id}"


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
