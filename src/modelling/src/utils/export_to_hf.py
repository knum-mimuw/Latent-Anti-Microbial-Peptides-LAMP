"""Export Lightning checkpoints to HuggingFace Hub.

Usage:
    python -m modelling.src.utils.export_to_hf \
        --checkpoint path/to/checkpoint.ckpt \
        --repo-id username/model-name
"""

import argparse
from pathlib import Path
import torch

from .importing import get_obj_from_import_path


def export_to_huggingface(
    checkpoint_path: str | Path,
    repo_id: str,
) -> None:
    """Export a Lightning checkpoint to HuggingFace Hub.

    Args:
        checkpoint_path: Path to the Lightning checkpoint file.
        repo_id: HuggingFace repo ID (e.g., "username/model-name").
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract model info from checkpoint
    hp = ckpt["hyper_parameters"]["config"]
    model_class_path = hp["model"]["model_class_path"]
    config_class_path = hp["model"]["config_class_path"]
    config_dict = hp["model"]["config"]

    print(f"Model: {model_class_path}")
    print(f"Config: {config_class_path}")

    # Load model and config classes dynamically
    model_class = get_obj_from_import_path(model_class_path)
    config_class = get_obj_from_import_path(config_class_path)

    # Create config
    config = config_class(**config_dict)

    # Extract weights (strip "model." prefix from MetaModule)
    print("Extracting model weights...")
    weights = {
        k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
    }

    # Create model and load weights
    print("Creating model and loading weights...")
    model = model_class(config)
    model.load_state_dict(weights)

    # Push to HuggingFace Hub
    print(f"Pushing to HuggingFace Hub: {repo_id}")
    model.push_to_hub(repo_id)
    print(f"Done! Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Lightning checkpoint to HuggingFace Hub"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the Lightning checkpoint file",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/model-name)",
    )

    args = parser.parse_args()

    export_to_huggingface(
        checkpoint_path=args.checkpoint,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
