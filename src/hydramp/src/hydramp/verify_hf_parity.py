"""Deterministic CPU parity: raw `hydramp` modules vs `HydrAMPModel` vs local `AutoModel` export.

This checks the same numerical path as a Hugging Face Hub upload: legacy pickle weights are
loaded into `HydrAMPModel`, an export bundle is written, and `AutoModel.from_pretrained` loads
that bundle with ``trust_remote_code=True``.

For reproducibility, use CPU, ``model.eval()``, and a fixed manual seed (see ``run_parity``).

The optional third path from the project plan — ``pep_compass_jr`` ``HydrAMPEncoderDecoder`` —
uses a different encoder input representation (one-hot via ``pep_compass`` utilities) than the
token-id HydrAMP stack here; it is not compared in this script to avoid a misleading mismatch.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import torch
from transformers import AutoModel

from .config import HydrAMPConfig
from .export_to_hf import (
    _load_legacy_weights,
    adapt_legacy_encoder_state_dict,
    write_model_export_bundle,
)
from .hydramp import HydrAMPDecoder, HydrAMPEncoder
from .model import HydrAMPModel

logger = logging.getLogger(__name__)


def _stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a - b).abs()
    return float(diff.max().item()), float(diff.mean().item())


def run_parity(*, weights_dir: Path, atol: float = 1e-5) -> int:
    """Return 0 if all checks pass, else 1."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    weights_dir = weights_dir.resolve()

    cfg = HydrAMPConfig()
    seq_len = cfg.sequence_length
    batch = 4

    enc_path = weights_dir / "encoder_weights.pickle"
    dec_path = weights_dir / "decoder_weights.pickle"
    if not enc_path.exists() or not dec_path.exists():
        raise FileNotFoundError(
            f"Expected encoder_weights.pickle and decoder_weights.pickle in {weights_dir}."
        )

    raw_enc = HydrAMPEncoder(
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        latent_dim=cfg.latent_dim,
        sequence_length=cfg.sequence_length,
        gru_hidden_size=cfg.encoder_hidden_size,
    ).to(device)
    raw_dec = HydrAMPDecoder(
        sequence_length=cfg.sequence_length,
        latent_dim=cfg.latent_dim,
        condition_dim=cfg.condition_dim,
        hidden_size=cfg.decoder_hidden_size,
        vocab_size=cfg.vocab_size,
    ).to(device)
    enc_sd = torch.load(enc_path, map_location=device, weights_only=True)
    raw_enc.load_state_dict(adapt_legacy_encoder_state_dict(enc_sd))
    raw_dec.load_state_dict(torch.load(dec_path, map_location=device, weights_only=True))
    raw_enc.eval()
    raw_dec.eval()

    combined = HydrAMPModel(cfg).to(device)
    _load_legacy_weights(combined, weights_dir=weights_dir, map_location=device)
    combined.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(1)
    input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device, generator=gen)
    z = torch.randn(batch, cfg.latent_dim, device=device, generator=gen)
    condition = torch.ones(batch, cfg.condition_dim, device=device)

    r_mean, r_log_std = raw_enc.encode(input_ids)
    c_mean, c_log_std = combined.encoder.encode(input_ids)

    dec_in = torch.cat([z, condition], dim=-1)
    r_logits = raw_dec(dec_in, return_logits=True)
    c_logits = combined.forward_latent_positions(z, condition=condition, return_logits=True).logits

    failures: list[str] = []

    def check(label: str, left: torch.Tensor, right: torch.Tensor) -> None:
        mx, mn = _stats(left, right)
        ok = mx <= atol
        logger.info("%s: max_abs=%.3e mean_abs=%.3e pass=%s (atol=%s)", label, mx, mn, ok, atol)
        if not ok:
            failures.append(label)

    check("encoder mean (raw vs HydrAMPModel)", r_mean, c_mean)
    check("encoder log_std (raw vs HydrAMPModel)", r_log_std, c_log_std)
    check("decoder logits (raw vs HydrAMPModel)", r_logits, c_logits)

    r_ids = r_logits.argmax(dim=-1)
    c_ids = c_logits.argmax(dim=-1)
    id_match = bool(torch.equal(r_ids, c_ids))
    logger.info("greedy token ids match (raw vs HydrAMPModel): %s", id_match)
    if not id_match:
        failures.append("greedy token ids (raw vs HydrAMPModel)")

    with tempfile.TemporaryDirectory(prefix="hydramp-parity-export-") as tmp:
        export_dir = Path(tmp)
        write_model_export_bundle(weights_dir=weights_dir, export_dir=export_dir)
        auto = AutoModel.from_pretrained(
            str(export_dir),
            trust_remote_code=True,
            local_files_only=True,
        ).to(device)
        auto.eval()

        a_mean, a_log_std = auto.encoder.encode(input_ids)
        a_logits = auto.forward_latent_positions(z, condition=condition, return_logits=True).logits

    check("encoder mean (raw vs AutoModel export)", r_mean, a_mean)
    check("encoder log_std (raw vs AutoModel export)", r_log_std, a_log_std)
    check("decoder logits (raw vs AutoModel export)", r_logits, a_logits)

    a_ids = a_logits.argmax(dim=-1)
    auto_id_match = bool(torch.equal(r_ids, a_ids))
    logger.info("greedy token ids match (raw vs AutoModel export): %s", auto_id_match)
    if not auto_id_match:
        failures.append("greedy token ids (raw vs AutoModel export)")

    if failures:
        logger.error("FAIL: %s", ", ".join(failures))
        return 1
    logger.info("OK: raw split modules match HydrAMPModel and AutoModel(local export).")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "CPU parity: raw hydramp encoder/decoder vs HydrAMPModel vs local HF export bundle"
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("src/hydramp/weights"),
        help="Directory containing encoder_weights.pickle and decoder_weights.pickle",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="Max absolute error threshold")
    args = parser.parse_args()
    code = run_parity(weights_dir=args.weights_dir, atol=args.atol)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
