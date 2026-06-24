"""Build :class:`~amp_opt.latent_sampling_solver.ProteinLatentSampling` from Hub HydrAMP weights.

Loads ``transformers.AutoModel`` / ``AutoTokenizer`` with ``trust_remote_code=True`` (no local
``hydramp`` package required at runtime). Optional Hub auth: pass ``hf_token`` in solver kwargs or
set ``HF_TOKEN`` in the environment.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox

from amp_opt.latent_sampling_solver import ProteinLatentSampling


def build_latent_sampling_hydramp_solver(
    black_box: AbstractBlackBox,
    x0: np.ndarray,
    y0: np.ndarray,
    kwargs: dict[str, Any],
) -> ProteinLatentSampling:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Latent sampling + Hugging Face HydrAMP requires `transformers` and `huggingface_hub`. "
            "Install them with e.g. `uv sync --package lamp-amp-opt --extra hub`, "
            "or add the same packages to your environment."
        ) from exc

    kw = dict(kwargs)
    hf_model_repo_id = kw.pop("hf_model_repo_id")
    hf_tokenizer_repo_id = kw.pop("hf_tokenizer_repo_id")
    hf_model_revision = kw.pop("hf_model_revision", None)
    hf_tokenizer_revision = kw.pop("hf_tokenizer_revision", None)
    device_str = str(kw.pop("device", "cpu"))
    hf_token = kw.pop("hf_token", None)
    trust_remote_code = bool(kw.pop("trust_remote_code", True))

    token: str | bool | None = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
    if token == "":
        token = None

    tok_common: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    mod_common: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if hf_model_revision is not None:
        mod_common["revision"] = hf_model_revision
    if hf_tokenizer_revision is not None:
        tok_common["revision"] = hf_tokenizer_revision
    if token:
        tok_common["token"] = token
        mod_common["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_repo_id, **tok_common)
    model = AutoModel.from_pretrained(hf_model_repo_id, **mod_common)
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    enc = getattr(model, "encoder", None)
    if enc is None or not callable(getattr(enc, "encode", None)):
        raise TypeError("HydrAMP model must expose callable encoder.encode.")
    if not callable(getattr(model, "decode_to_token_ids", None)):
        raise TypeError("HydrAMP model must expose decode_to_token_ids.")

    cfg = model.config
    seq_len = int(cfg.sequence_length)
    vocab_size = int(cfg.vocab_size)

    x_arr = np.asarray(x0)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D x0, got shape {x_arr.shape}.")
    bb_len = int(x_arr.shape[1])
    if bb_len > seq_len:
        raise ValueError(
            f"Seed row length {bb_len} exceeds model.config.sequence_length ({seq_len}); "
            "truncation is not supported."
        )
    mutable_prefix_len: int | None = None if bb_len == seq_len else bb_len

    alphabet = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]

    def tokenize_row(row: np.ndarray) -> torch.Tensor:
        flat = row.reshape(-1)
        if flat.size != bb_len:
            raise ValueError(
                f"Row length {flat.size} != expected black-box length {bb_len}."
            )
        seq = "".join(str(flat[i]) for i in range(flat.size))
        batch = tokenizer(
            seq,
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            max_length=seq_len,
            truncation=False,
        )
        ids = batch["input_ids"].to(device)
        return ids.long()

    def decode_row(ids_1d: np.ndarray) -> np.ndarray:
        vec = np.asarray(ids_1d, dtype=np.int64).reshape(-1)
        if vec.size != seq_len:
            raise ValueError(f"decode_row expected length {seq_len}, got {vec.size}.")
        chars: list[str] = []
        for vid in vec[:bb_len].tolist():
            t = tokenizer.convert_ids_to_tokens(int(vid))
            if t == tokenizer.pad_token:
                raise ValueError(
                    "Decoded a pad token in the peptide prefix; "
                    "pad is invalid for the APEX peptide box."
                )
            chars.append(str(t))
        return np.array(chars, dtype=object)

    def encode_dist(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = input_ids.to(device)
        with torch.no_grad():
            mean, log_std = model.encoder.encode(input_ids)
        return mean, log_std

    def decode_ids(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.decode_to_token_ids(z)

    return ProteinLatentSampling(
        black_box,
        x0,
        y0,
        encode_dist,
        decode_ids,
        seq_len,
        vocab_size,
        alphabet=alphabet,
        tokenize_row=tokenize_row,
        decode_row=decode_row,
        **kw,
    )
