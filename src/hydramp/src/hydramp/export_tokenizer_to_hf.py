"""Upload HydrAMP AA tokenizer to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_branch, create_tag
from huggingface_hub.errors import HfHubHTTPError

from .citation import HYDRAMP_README_CITATION_SECTION
from .config import HydrAMPConfig
from .tokenizer import HydrAMPAATokenizer

logger = logging.getLogger(__name__)


def write_tokenizer_export_bundle(
    *,
    export_dir: str | Path,
    repo_id: str = "local-export",
    revision: str = "local-export",
) -> None:
    """Write tokenizer artifacts (same as Hub upload) for local checks or manual upload."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = HydrAMPAATokenizer()
    tokenizer.save_pretrained(export_dir)
    _inject_auto_map(export_dir)
    _copy_remote_code(export_dir)
    _write_config_stub(export_dir)
    _write_readme(export_dir, repo_id=repo_id, revision=revision)


def export_tokenizer_to_huggingface(
    *,
    repo_id: str,
    revision: str | None = None,
    tag: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Create/update a tokenizer repo for HydrAMP amino-acid tokenization."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

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

    with tempfile.TemporaryDirectory(prefix="hydramp-tokenizer-") as tmp:
        out = Path(tmp)
        write_tokenizer_export_bundle(export_dir=out, repo_id=repo_id, revision=upload_revision)

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(out),
            revision=upload_revision,
            commit_message=commit_message or "Upload HydrAMP AA tokenizer",
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
            if exc.response is None or exc.response.status_code != 409:
                raise

    return f"https://huggingface.co/{repo_id}"


def _inject_auto_map(export_dir: Path) -> None:
    cfg_path = export_dir / "tokenizer_config.json"
    payload = json.loads(cfg_path.read_text())
    payload["auto_map"] = {"AutoTokenizer": ["tokenizer.HydrAMPAATokenizer", None]}
    payload.pop("tokenizer_class", None)
    cfg_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_remote_code(export_dir: Path) -> None:
    package_dir = Path(__file__).resolve().parent
    (export_dir / "tokenizer.py").write_text((package_dir / "tokenizer.py").read_text())
    (export_dir / "config.py").write_text((package_dir / "config.py").read_text())


def _write_config_stub(export_dir: Path) -> None:
    cfg = HydrAMPConfig().to_dict()
    cfg["auto_map"] = {"AutoConfig": "config.HydrAMPConfig"}
    (export_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")


def _suggested_model_repo(tokenizer_repo_id: str) -> str:
    if "/" in tokenizer_repo_id:
        org, _rest = tokenizer_repo_id.split("/", 1)
        return f"{org}/hydramp"
    return "your-org/hydramp"


def _write_readme(export_dir: Path, *, repo_id: str, revision: str) -> None:
    model_id_line = _suggested_model_repo(repo_id)
    snippet_revision = (
        "main" if revision == "local-export" or repo_id == "local-export" else revision
    )
    readme = f"""---
library_name: transformers
tags:
- lamp
- hydramp
- tokenizer
- amino-acids
---

# LAMP HydrAMP AA tokenizer

Peptide tokenizer used by **LAMP HydrAMP** Hub models: maps amino-acid strings to
fixed-length token IDs (padding/truncation to the HydrAMP sequence length). Load with
``trust_remote_code=True`` because the class ships in this repo.

When you publish results or reuse HydrAMP tokenization, cite the original *Nature
Communications* paper (Szymczak *et al.*, 2023); **Citation** at the bottom of this README has
BibTeX and links.

**Tokenizer repo:** `{repo_id}`

## Load

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "{repo_id}",
    revision="{snippet_revision}",
    trust_remote_code=True,
)
```

## Use with a HydrAMP model

Point ``AutoModel.from_pretrained`` at your HydrAMP **model** repo (same ``revision`` if
you version them together), then tokenize before ``encode`` / ``forward``:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "{model_id_line}"
tokenizer_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_id,
    revision="{snippet_revision}",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    model_id,
    revision="{snippet_revision}",
    trust_remote_code=True,
)
model.eval()

batch = tokenizer(
    ["ACDEFGHIKLMNPQRSTVWY"],
    padding="max_length",
    truncation=True,
    max_length=model.config.sequence_length,
    return_tensors="pt",
)

with torch.no_grad():
    mean, log_std = model.encoder.encode(batch["input_ids"])
```

{HYDRAMP_README_CITATION_SECTION}
"""
    (export_dir / "README.md").write_text(readme)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Upload HydrAMP AA tokenizer to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "Target tokenizer repo (e.g. user/hydramp-aa-tokenizer). "
            "Required unless --local-export-dir."
        ),
    )
    parser.add_argument(
        "--local-export-dir",
        default=None,
        help="Write tokenizer bundle here and skip Hub upload (no --repo-id needed)",
    )
    parser.add_argument("--revision", default=None, help="Target branch/revision")
    parser.add_argument("--tag", default=None, help="Optional immutable tag")
    parser.add_argument("--private", action="store_true", help="Create repo as private if missing")
    parser.add_argument("--commit-message", default=None, help="Optional commit message")
    parser.add_argument("--token", default=None, help="Optional HF token override")
    args = parser.parse_args()

    if args.local_export_dir:
        out = Path(args.local_export_dir)
        write_tokenizer_export_bundle(
            export_dir=out,
            repo_id=args.repo_id or "local-export",
            revision=args.revision or "local-export",
        )
        logger.info("%s", out.resolve())
        return

    if not args.repo_id:
        raise SystemExit("--repo-id is required unless --local-export-dir is set")

    url = export_tokenizer_to_huggingface(
        repo_id=args.repo_id,
        revision=args.revision,
        tag=args.tag,
        private=args.private,
        commit_message=args.commit_message,
        token=args.token,
    )
    logger.info("%s", url)


if __name__ == "__main__":
    main()

