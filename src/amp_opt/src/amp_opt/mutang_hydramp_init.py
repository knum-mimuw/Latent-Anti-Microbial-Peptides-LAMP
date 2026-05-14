"""Build :class:`~amp_opt.mutang_solver.ProteinMutangUniformMutation` from Hub HydrAMP weights.

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

from amp_opt.mutang_solver import ProteinMutangUniformMutation
from pep_compass_jr.utils import softmax_probs_jacobian_fn


def build_mutang_hydramp_solver(
    black_box: AbstractBlackBox,
    x0: np.ndarray,
    y0: np.ndarray,
    kwargs: dict[str, Any],
) -> ProteinMutangUniformMutation:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised when hub extra missing
        raise ImportError(
            "Mutang + Hugging Face HydrAMP requires `transformers` and `huggingface_hub`. "
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
    jacobian_mode = kw.pop("jacobian_mode", "approx")
    if jacobian_mode not in ("strict", "approx"):
        raise ValueError("jacobian_mode must be 'strict' or 'approx'.")
    jacobian_eps = float(kw.pop("jacobian_eps", 1e-4))
    condition = kw.pop("condition", None)
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
    if not callable(getattr(model, "forward_latent_positions", None)):
        raise TypeError("HydrAMP model must expose forward_latent_positions.")

    cfg = model.config
    seq_len = int(cfg.sequence_length)
    vocab_size = int(cfg.vocab_size)

    x_arr = np.asarray(x0)
    if x_arr.ndim != 2:
        raise ValueError(f"Expected 2D x0, got shape {x_arr.shape}.")
    bb_len = int(x_arr.shape[1])
    if bb_len != seq_len:
        raise ValueError(
            f"Seed row length {bb_len} must equal model.config.sequence_length ({seq_len}) "
            "for HydrAMP mutang runs."
        )

    alphabet = [tokenizer.convert_id_to_token(i) for i in range(vocab_size)]

    if condition is not None:
        cond_list = list(condition)
        cond_dim = int(getattr(cfg, "condition_dim", len(cond_list)))
        if len(cond_list) != cond_dim:
            raise ValueError(
                f"condition has length {len(cond_list)}, expected condition_dim={cond_dim}."
            )

    def tokenize_row(row: np.ndarray) -> torch.Tensor:
        flat = row.reshape(-1)
        if flat.size != seq_len:
            raise ValueError(
                f"Row length {flat.size} != sequence_length {seq_len} (HydrAMP fixed width)."
            )
        seq = "".join(str(flat[i]) for i in range(flat.size))
        batch = tokenizer(
            seq,
            add_special_tokens=False,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        ids = batch["input_ids"].to(device)
        if tuple(ids.shape) != (1, seq_len):
            raise ValueError(
                f"Tokenizer returned input_ids shape {tuple(ids.shape)}, expected (1, {seq_len})."
            )
        return ids.long()

    def decode_row(ids_1d: np.ndarray) -> np.ndarray:
        vec = np.asarray(ids_1d, dtype=np.int64).reshape(-1)
        if vec.size != seq_len:
            raise ValueError(f"decode_row expected length {seq_len}, got {vec.size}.")
        chars: list[str] = []
        for vid in vec.tolist():
            t = tokenizer.convert_id_to_token(int(vid))
            if t == tokenizer.pad_token:
                raise ValueError(
                    "Decoded a pad token in the sequence; pad is invalid for the APEX peptide box."
                )
            chars.append(str(t))
        return np.array(chars, dtype=object)

    def encode(input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(device)
        with torch.no_grad():
            mean, _log_std = model.encoder.encode(input_ids)
        return mean

    def decode_logits(z: torch.Tensor) -> torch.Tensor:
        if condition is None:
            cond_t: torch.Tensor | None = None
        else:
            cond_t = torch.as_tensor(condition, dtype=z.dtype, device=z.device)
            cond_t = cond_t.view(1, -1).expand(z.shape[0], -1)
        out = model.forward_latent_positions(z, condition=cond_t, return_logits=True)
        logits = out.logits
        if logits is None:
            raise RuntimeError("forward_latent_positions returned no logits.")
        return logits

    jacobian_batch_fn = softmax_probs_jacobian_fn(
        decode_logits,
        sequence_length=seq_len,
        vocab_size=vocab_size,
        jacobian_mode=jacobian_mode,  # type: ignore[arg-type]
        jacobian_eps=jacobian_eps,
    )

    return ProteinMutangUniformMutation(
        black_box,
        x0,
        y0,
        encode,
        jacobian_batch_fn,
        seq_len,
        vocab_size,
        alphabet=alphabet,
        tokenize_row=tokenize_row,
        decode_row=decode_row,
        **kw,
    )
