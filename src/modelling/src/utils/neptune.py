import neptune
from pathlib import Path
from typing import Dict, Any
import yaml
import json

from src.utils.paths import PROJECT_ROOT


def download_neptune_config(
    run_id: str, project_name: str, api_token: str, output_path: Path | None = None
) -> Dict[str, Any]:
    """
    Download and load the config file from a Neptune experiment.

    Args:
        run_id: Neptune run ID
        project_name: Neptune project name
        api_token: Neptune API token
        output_path: Optional path to save the config. If None, saves in .neptune cache

    Returns:
        Dict containing the loaded config
    """
    # Set up cache directory if no output path specified
    if output_path is None:
        cache_dir = Path(".neptune") / project_name / run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / "model_config.yaml"

    print(f"📄 Downloading config from run `{run_id}` to {output_path}")

    # Initialize Neptune run and download config
    run = neptune.init_run(with_id=run_id, project=project_name, api_token=api_token)
    run["config/model_config.yaml"].download(destination=str(output_path))
    run.stop()
    print("✅ Config download complete")

    # Load and return the config
    with output_path.open() as f:
        if output_path.suffix.lower() in {".yml", ".yaml"}:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    return config


def download_checkpoint(
    run_id: str,
    checkpoint_name: str,
    neptune_project: str,
    api_token: str,
) -> Path | None:
    """
    Download a checkpoint from Neptune if it doesn't exist locally.

    Args:
        run_id: Neptune run ID
        checkpoint_name: Name of the checkpoint to download
        neptune_project: Neptune project name
        api_token: Neptune API token

    Returns:
        Path to the downloaded checkpoint or None if download failed
    """
    cache_dir = PROJECT_ROOT / ".neptune" / neptune_project / run_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    auto_ckpt = cache_dir / f"{checkpoint_name}.ckpt"
    print(f"📥 Checkpoint path: {auto_ckpt}")

    if not auto_ckpt.exists():
        print(f"    → Downloading `{checkpoint_name}` from run `{run_id}`…")
        run = neptune.init_run(
            with_id=run_id, project=neptune_project, api_token=api_token
        )
        run[f"training/model/checkpoints/{checkpoint_name}"].download(
            destination=str(auto_ckpt)
        )
        run.stop()
        print("    → Download complete.")
    else:
        print("    → Already cached, skipping download.")

    return auto_ckpt if auto_ckpt.exists() else None
