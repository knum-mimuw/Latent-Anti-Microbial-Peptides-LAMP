"""Export Hugging Face Trainer checkpoints (model folders) to the Hub."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

from transformers import AutoModel


class HfExportResult(TypedDict):
    """Structured result describing a completed Hugging Face export."""

    repo_id: str
    hub_url: str
    revision: str
    tag: str | None
    checkpoint_path: str
    model_class_path: str
    config_class_path: str


def export_to_huggingface(
    checkpoint_path: str | Path,
    repo_id: str,
    *,
    revision: str | None = None,
    tag: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
    model_card_title: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HfExportResult:
    """Export a Trainer checkpoint directory (``save_pretrained`` layout) to the Hub.

    Args:
        checkpoint_path: Directory containing ``config.json`` and model weights
            (e.g. ``model.safetensors``), typically a Hugging Face Trainer checkpoint folder.
        repo_id: Hugging Face repo ID (e.g. ``username/model-name``).
        revision: Optional branch or revision target for the upload.
        tag: Optional immutable tag to create after the upload.
        private: Whether the repo should be created as private if missing.
        commit_message: Optional Hub commit message.
        token: Optional Hub token. When omitted, Hub environment variables apply.
        model_card_title: Optional title override for the generated README.
        metadata: Optional metadata to save locally and include in the model card.
    """
    from huggingface_hub import HfApi, create_branch, create_tag
    from huggingface_hub.errors import HfHubHTTPError

    weights_dir = Path(checkpoint_path).resolve()
    loaded = _load_model_from_trainer_dir(weights_dir)
    metadata_dict = dict(metadata or {})

    print(f"Loading weights from: {weights_dir}")
    print(f"Model: {loaded['model_class_path']}")
    print(f"Config: {loaded['config_class_path']}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )

    commit_message = commit_message or f"Upload model from {weights_dir.name}"
    upload_revision = revision or "main"
    if upload_revision != "main":
        create_branch(
            repo_id=repo_id,
            repo_type="model",
            branch=upload_revision,
            revision="main",
            token=token,
            exist_ok=True,
        )

    resolved_title = model_card_title or repo_id.split("/")[-1]

    with tempfile.TemporaryDirectory(prefix="lamp-hf-export-") as temp_dir:
        export_dir = Path(temp_dir)
        loaded["model"].save_pretrained(export_dir)
        _inject_auto_map(
            export_dir=export_dir,
            config_class=loaded["config_class"],
            model_class=loaded["model_class"],
        )
        _write_remote_code_files(
            export_dir=export_dir,
            model_class=loaded["model_class"],
            config_class=loaded["config_class"],
        )
        _write_metadata_artifacts(
            export_dir=export_dir,
            repo_id=repo_id,
            revision=upload_revision,
            source_path=weights_dir,
            model_class_path=loaded["model_class_path"],
            config_class_path=loaded["config_class_path"],
            model_card_title=resolved_title,
            metadata=metadata_dict,
        )
        _write_runtime_requirements(export_dir)

        print(f"Uploading export folder to Hugging Face Hub: {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            revision=upload_revision,
            commit_message=commit_message,
        )

    if upload_revision != "main":
        _upload_main_readme(
            api=api,
            repo_id=repo_id,
            model_card_title=resolved_title,
            revision=upload_revision,
            tag=tag,
        )

    if tag:
        try:
            create_tag(
                repo_id=repo_id,
                repo_type="model",
                tag=tag,
                revision=upload_revision,
                token=token,
            )
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code == 409:
                print(f"Tag '{tag}' already exists on {repo_id}; skipping.")
            else:
                raise

    result: HfExportResult = {
        "repo_id": repo_id,
        "hub_url": f"https://huggingface.co/{repo_id}",
        "revision": upload_revision,
        "tag": tag,
        "checkpoint_path": str(weights_dir),
        "model_class_path": loaded["model_class_path"],
        "config_class_path": loaded["config_class_path"],
    }
    print(f"Done! Model available at: {result['hub_url']}")
    return result


def _qual(obj: type[Any]) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _load_model_from_trainer_dir(weights_dir: Path) -> dict[str, Any]:
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Expected a directory with model files, got {weights_dir}")
    if not (weights_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json under {weights_dir}")

    model = AutoModel.from_pretrained(
        str(weights_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    cfg_cls = model.config.__class__
    model_cls = model.__class__
    return {
        "model": model,
        "model_class": model_cls,
        "config_class": cfg_cls,
        "model_class_path": _qual(model_cls),
        "config_class_path": _qual(cfg_cls),
    }


def _inject_auto_map(
    export_dir: Path,
    config_class: type[Any],
    model_class: type[Any],
) -> None:
    """Ensure config.json contains the auto_map needed for trust_remote_code loading."""
    config_json_path = export_dir / "config.json"
    config_data = json.loads(config_json_path.read_text())
    if "auto_map" not in config_data:
        config_data["auto_map"] = {
            "AutoConfig": f"config.{config_class.__name__}",
            "AutoModel": f"model.{model_class.__name__}",
        }
        config_json_path.write_text(json.dumps(config_data, indent=2, sort_keys=True) + "\n")


def _upload_main_readme(
    api: Any,
    repo_id: str,
    model_card_title: str,
    revision: str,
    tag: str | None,
) -> None:
    """Upload a README to main so the repo page isn't empty."""
    load_ref = tag or revision
    readme = "\n".join(
        [
            "---",
            "library_name: transformers",
            "tags:",
            "- lamp",
            "- pytorch",
            "- custom-code",
            "---",
            f"# {model_card_title}",
            "",
            "## Loading",
            "",
            "```python",
            "from transformers import AutoModel",
            "",
            "model = AutoModel.from_pretrained(",
            f'    "{repo_id}",',
            f'    revision="{load_ref}",',
            "    trust_remote_code=True,",
            ")",
            "```",
            "",
            f"See the [`{revision}`](https://huggingface.co/{repo_id}/tree/{revision})"
            " branch for full model files and metadata.",
            "",
        ]
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        revision="main",
        commit_message=f"Update main README (latest release: {load_ref})",
    )


def _write_remote_code_files(
    export_dir: Path,
    model_class: type[Any],
    config_class: type[Any],
) -> None:
    """Copy custom config/model source files so Hub loading stays reproducible."""
    config_source = _get_source_path(config_class)
    model_source = _get_source_path(model_class)

    (export_dir / "__init__.py").write_text('"""Remote code package for Hub loading."""\n')
    shutil.copyfile(config_source, export_dir / "config.py")
    shutil.copyfile(model_source, export_dir / "model.py")


def _write_runtime_requirements(export_dir: Path) -> None:
    """Write a minimal runtime dependency manifest for Hub consumers."""
    requirement_lines = []
    for package_name in ["torch", "transformers", "einops"]:
        try:
            version = importlib.metadata.version(package_name)
            requirement_lines.append(f"{package_name}=={version}")
        except importlib.metadata.PackageNotFoundError:
            requirement_lines.append(package_name)

    (export_dir / "requirements.txt").write_text("\n".join(requirement_lines) + "\n")


def _write_metadata_artifacts(
    export_dir: Path,
    repo_id: str,
    revision: str,
    source_path: Path,
    model_class_path: str,
    config_class_path: str,
    model_card_title: str,
    metadata: Mapping[str, Any],
) -> None:
    """Write model card, loading example, and structured metadata."""
    payload = {
        "weights_dir": str(source_path),
        "model_class_path": model_class_path,
        "config_class_path": config_class_path,
        "revision": revision,
        "metadata": dict(metadata),
    }
    (export_dir / "lamp_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    loading_revision = metadata.get("hf_load_revision") or metadata.get("model_version") or revision
    (export_dir / "loading_example.py").write_text(
        "\n".join(
            [
                "from transformers import AutoModel",
                "",
                f'MODEL_ID = "{repo_id}"',
                f'REVISION = "{loading_revision}"',
                "",
                "model = AutoModel.from_pretrained(",
                "    MODEL_ID,",
                "    revision=REVISION,",
                "    trust_remote_code=True,",
                ")",
                "",
                "print(model.__class__.__name__)",
            ]
        )
        + "\n"
    )

    metadata_block = json.dumps(metadata, indent=2, sort_keys=True) if metadata else "{}"
    readme = "\n".join(
        [
            "---",
            "library_name: transformers",
            "tags:",
            "- lamp",
            "- pytorch",
            "- custom-code",
            "---",
            f"# {model_card_title}",
            "",
            "## Overview",
            "",
            f"This model was exported from the Trainer checkpoint directory `{source_path.name}`.",
            "",
            "## Loading",
            "",
            "```python",
            "from transformers import AutoModel",
            "",
            f'model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)',
            "```",
            "",
            "Load a specific release with a revision or tag:",
            "",
            "```python",
            "from transformers import AutoModel",
            "",
            "model = AutoModel.from_pretrained(",
            f'    "{repo_id}",',
            f'    revision="{loading_revision}",',
            "    trust_remote_code=True,",
            ")",
            "```",
            "",
            "## Provenance",
            "",
            f"- Source weights dir: `{source_path}`",
            f"- Model class: `{model_class_path}`",
            f"- Config class: `{config_class_path}`",
            f"- Suggested revision: `{loading_revision}`",
            "",
            "## Metadata",
            "",
            "```json",
            metadata_block,
            "```",
        ]
    )
    (export_dir / "README.md").write_text(readme + "\n")


def _get_source_path(obj: type[Any]) -> Path:
    """Resolve the Python file that defines an object."""
    source_path = inspect.getsourcefile(obj)
    if source_path is None:
        raise RuntimeError(f"Could not determine source file for {obj}.")
    return Path(source_path)


def main() -> None:
    """CLI entry point for exporting Trainer weights to Hugging Face Hub."""
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face Trainer checkpoint folder to the Hub",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--weights-dir",
        type=str,
        help="Directory with config.json + model weights from Trainer.save_pretrained",
    )
    src.add_argument(
        "--checkpoint",
        type=str,
        help="Deprecated alias for --weights-dir (directory path, not a .ckpt file).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g. username/model-name)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hub revision or branch to upload to",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional immutable Hub tag to create after the upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not already exist",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Optional Hub commit message",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hub token. If omitted, standard Hub env vars are used.",
    )
    parser.add_argument(
        "--model-card-title",
        type=str,
        default=None,
        help="Optional title for the generated README",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default=None,
        help="Optional JSON file with metadata to embed in the export",
    )

    args = parser.parse_args()
    weights_dir = args.weights_dir or args.checkpoint
    metadata_dict: dict[str, Any] | None = None
    if args.metadata_json:
        metadata_dict = json.loads(Path(args.metadata_json).read_text())

    export_to_huggingface(
        checkpoint_path=weights_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        tag=args.tag,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
        model_card_title=args.model_card_title,
        metadata=metadata_dict,
    )


if __name__ == "__main__":
    main()
