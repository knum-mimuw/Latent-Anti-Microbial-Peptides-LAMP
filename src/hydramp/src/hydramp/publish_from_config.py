"""Upload HydrAMP model or tokenizer from a YAML config (default Hub branch: ``main``)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from .export_to_hf import export_to_huggingface
from .export_tokenizer_to_hf import export_tokenizer_to_huggingface

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return data


def _resolve_path(config_path: Path, value: str | None, default: str | None) -> Path:
    raw = value if value is not None else default
    if raw is None:
        raise ValueError("missing path")
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (config_path.parent / p).resolve()


def _revision_for_api(revision: str) -> str | None:
    """Pass ``None`` when targeting ``main`` so exporters use the same default path."""
    return None if revision == "main" else revision


def publish_from_config_file(config_path: Path, *, token: str | None = None) -> str:
    """Load YAML at ``config_path`` and upload. Returns the Hub model URL string."""
    cfg = _load_config(config_path)
    artifact = cfg.get("artifact")
    if artifact not in ("model", "tokenizer"):
        raise ValueError("config must set artifact to 'model' or 'tokenizer'")

    repo_id = cfg.get("repo_id")
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("config must set repo_id (str)")

    revision_raw = cfg.get("revision")
    revision = revision_raw if isinstance(revision_raw, str) and revision_raw else "main"

    tag = cfg.get("tag")
    if tag == "":
        tag = None
    if tag is not None and not isinstance(tag, str):
        raise ValueError("tag must be a string when set")

    private = bool(cfg.get("private", False))
    commit_message = cfg.get("commit_message")
    if commit_message is not None and not isinstance(commit_message, str):
        raise ValueError("commit_message must be a string when set")

    rev_arg = _revision_for_api(revision)
    if artifact == "model":
        weights_dir = _resolve_path(config_path, cfg.get("weights_dir"), "../weights")
        return export_to_huggingface(
            weights_dir=weights_dir,
            repo_id=repo_id,
            revision=rev_arg,
            tag=tag,
            private=private,
            commit_message=commit_message,
            token=token,
        )

    return export_tokenizer_to_huggingface(
        repo_id=repo_id,
        revision=rev_arg,
        tag=tag,
        private=private,
        commit_message=commit_message,
        token=token,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Upload HydrAMP model or tokenizer from YAML (default Hub branch: main).",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to hydramp_model_hub.yaml or hydramp_tokenizer_hub.yaml",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token override (else HF_TOKEN env or cached hub login)",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"config file not found: {config_path}")

    url = publish_from_config_file(config_path, token=args.token)
    logger.info("%s", url)


if __name__ == "__main__":
    main()
