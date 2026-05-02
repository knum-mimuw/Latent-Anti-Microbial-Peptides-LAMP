"""Export HydrAMP weights to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_branch, create_tag
from huggingface_hub.errors import HfHubHTTPError

from .citation import HYDRAMP_README_CITATION_SECTION
from .config import HydrAMPConfig
from .model import HydrAMPModel

logger = logging.getLogger(__name__)


def write_model_export_bundle(
    *,
    weights_dir: str | Path,
    export_dir: str | Path,
    hub_repo_id: str | None = None,
    readme_revision: str = "local-export",
) -> None:
    """Write the same artifact tree as Hub upload (for local parity checks or manual upload).

    ``hub_repo_id`` and ``readme_revision`` control the generated ``README.md`` Hub card
    (use the real Hub model id when uploading).
    """
    weights_dir = Path(weights_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    model = HydrAMPModel(HydrAMPConfig())
    _load_legacy_weights(model, weights_dir=weights_dir, map_location="cpu")
    model.save_pretrained(export_dir)
    _ensure_auto_map(export_dir)
    _copy_remote_code_files(export_dir)
    _write_model_hub_readme(
        export_dir,
        weights_dir=weights_dir,
        revision=readme_revision,
        hub_repo_id=hub_repo_id,
    )


def export_to_huggingface(
    *,
    weights_dir: str | Path,
    repo_id: str,
    revision: str | None = None,
    tag: str | None = None,
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload HydrAMP model artifacts to HF with trust_remote_code support."""
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

    with tempfile.TemporaryDirectory(prefix="hydramp-hf-export-") as temp_dir:
        export_dir = Path(temp_dir)
        write_model_export_bundle(
            weights_dir=weights_dir,
            export_dir=export_dir,
            hub_repo_id=repo_id,
            readme_revision=upload_revision,
        )

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            revision=upload_revision,
            commit_message=commit_message or "Upload HydrAMP model",
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


def adapt_legacy_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map legacy checkpoints: ``std_linear`` weights -> ``log_std_linear``."""
    adapted = dict(state_dict)
    if "std_linear.weight" in adapted:
        adapted["log_std_linear.weight"] = adapted.pop("std_linear.weight")
    if "std_linear.bias" in adapted:
        adapted["log_std_linear.bias"] = adapted.pop("std_linear.bias")
    return adapted


def _load_legacy_weights(
    model: HydrAMPModel,
    *,
    weights_dir: str | Path,
    map_location: str | torch.device | None = None,
) -> None:
    """Load legacy split encoder/decoder checkpoints into an HF-native model."""
    weights_dir = Path(weights_dir)
    encoder_path = weights_dir / "encoder_weights.pickle"
    decoder_path = weights_dir / "decoder_weights.pickle"
    if not encoder_path.exists() or not decoder_path.exists():
        raise FileNotFoundError(
            f"Expected encoder_weights.pickle and decoder_weights.pickle in {weights_dir}."
        )
    enc_sd = torch.load(encoder_path, map_location=map_location, weights_only=True)
    model.encoder.load_state_dict(adapt_legacy_encoder_state_dict(enc_sd))
    model.decoder.load_state_dict(
        torch.load(decoder_path, map_location=map_location, weights_only=True)
    )


def _ensure_auto_map(export_dir: Path) -> None:
    config_path = export_dir / "config.json"
    payload = json.loads(config_path.read_text())
    payload["auto_map"] = {
        "AutoConfig": "config.HydrAMPConfig",
        "AutoModel": "model.HydrAMPModel",
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_remote_code_files(export_dir: Path) -> None:
    package_root = Path(__file__).resolve().parent
    (export_dir / "__init__.py").write_text('"""HydrAMP remote code package."""\n')
    for filename in ("config.py", "hydramp.py", "model.py", "tokenizer.py"):
        shutil.copyfile(package_root / filename, export_dir / filename)


def _suggested_tokenizer_repo(model_repo_id: str | None) -> str:
    if model_repo_id and "/" in model_repo_id:
        org, _rest = model_repo_id.split("/", 1)
        return f"{org}/hydramp-aa-tokenizer"
    return "your-org/hydramp-aa-tokenizer"


def _read_latent_dim(export_dir: Path) -> int:
    payload = json.loads((export_dir / "config.json").read_text())
    return int(payload.get("latent_dim", 64))


def _write_model_hub_readme(
    export_dir: Path,
    *,
    weights_dir: Path,
    revision: str,
    hub_repo_id: str | None,
) -> None:
    """Write ``README.md`` for the Hub model card (YAML front matter + usage)."""
    model_id_line = hub_repo_id if hub_repo_id is not None else "your-org/hydramp"
    tokenizer_id_line = _suggested_tokenizer_repo(hub_repo_id)
    latent_dim = _read_latent_dim(export_dir)
    # Local-only bundles use readme_revision "local-export"; snippets should use Hub defaults.
    code_revision = revision if hub_repo_id is not None else "main"

    local_note = ""
    if hub_repo_id is None:
        local_note = (
            "> **Local bundle:** replace `your-org/hydramp` and "
            "`your-org/hydramp-aa-tokenizer` with your real Hub `repo_id` values "
            "after upload.\n\n"
        )

    companion = (
        f"Use the **HydrAMP AA tokenizer** Hub repo for peptide strings (often "
        f"`{tokenizer_id_line}` next to this model).\n\n"
    )

    readme = f"""---
library_name: transformers
tags:
- lamp
- hydramp
- antimicrobial-peptides
- custom-code
---

# LAMP HydrAMP

**LAMP** (latent antimicrobial peptide modelling) — **HydrAMP** is an encoder/decoder
for short amino-acid sequences: the encoder maps token IDs to a `{latent_dim}`-D latent
Gaussian (`mean`, `log_std`); the decoder maps latent vectors plus a 2-D **condition**
vector to per-position amino-acid logits. This Hub repo ships **remote Python code**;
load with ``trust_remote_code=True``.

When you publish results or reuse the HydrAMP architecture, cite the original *Nature
Communications* paper (Szymczak *et al.*, 2023); **Citation** at the bottom of this README has
BibTeX and links.

**Model repo:** `{model_id_line}`

{local_note}{companion}## Requirements

- ``transformers`` with ``AutoModel`` and remote-code loading
- ``torch``
- A separate **HydrAMP AA tokenizer** model repo (``AutoTokenizer``, ``trust_remote_code=True``)

## Load the model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "{model_id_line}",
    revision="{code_revision}",
    trust_remote_code=True,
)
model.eval()
```

## Tokenize, encode, reconstruct

``forward`` encodes ``input_ids`` and decodes from the latent **mean** (deterministic
reconstruction).

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "{model_id_line}"
tokenizer_id = "{tokenizer_id_line}"

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_id,
    revision="{code_revision}",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    model_id,
    revision="{code_revision}",
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
    out = model(batch["input_ids"])

# out.mean, out.log_std — out.logits shape [batch, seq_len, vocab_size]
```

## Decode from a custom latent ``z``

``z`` has shape ``[batch, {latent_dim}]``. ``condition`` defaults to the model's
``default_condition`` buffer when omitted.

```python
z = out.mean
with torch.no_grad():
    logits = model.forward_latent_positions(z, return_logits=True).logits
greedy_ids = model.decode_to_token_ids(z)  # or logits.argmax(dim=-1)
```

{HYDRAMP_README_CITATION_SECTION}
## Export provenance

Bundled from local weights directory ``{weights_dir}`` into this Hugging Face layout.
"""
    (export_dir / "README.md").write_text(readme)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Export HydrAMP weights to Hugging Face Hub")
    parser.add_argument(
        "--weights-dir",
        required=True,
        help="Directory with encoder_weights.pickle and decoder_weights.pickle",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Target model repo, e.g. user/hydramp (required unless --local-export-dir)",
    )
    parser.add_argument(
        "--local-export-dir",
        default=None,
        help="Write export bundle to this directory and skip Hub upload (no --repo-id needed)",
    )
    parser.add_argument("--revision", default=None, help="Target branch or revision")
    parser.add_argument("--tag", default=None, help="Optional immutable tag")
    parser.add_argument("--private", action="store_true", help="Create repo as private if missing")
    parser.add_argument("--commit-message", default=None, help="Optional commit message")
    parser.add_argument("--token", default=None, help="Optional HF token override")
    args = parser.parse_args()

    if args.local_export_dir:
        out = Path(args.local_export_dir)
        write_model_export_bundle(weights_dir=args.weights_dir, export_dir=out)
        logger.info("%s", out.resolve())
        return

    if not args.repo_id:
        raise SystemExit("--repo-id is required unless --local-export-dir is set")

    url = export_to_huggingface(
        weights_dir=args.weights_dir,
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

