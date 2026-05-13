"""Tangent-space substitution proposals from decoder Jacobians (SVD left singular vectors).

Jacobians use ambient rows in row-major order over ``(sequence position, vocab id)``, i.e. index
``pos * vocab_size + vocab_id``. That matches flattening ``[batch, sequence_length, vocab_size]``
logits or probabilities to ``[batch, ambient]`` before differentiating w.r.t. latent ``z``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def _substitutions_from_u(
    u: np.ndarray,
    *,
    sequence_length: int,
    vocab_size: int,
    token_threshold: float,
) -> dict[int, tuple[int, ...]]:
    """Map ``U`` columns (ambient directions) to per-position candidate vocab ids."""
    if u.ndim != 2:
        raise ValueError(f"U must be 2D, got {u.ndim}D")
    ambient = sequence_length * vocab_size
    if u.shape[0] != ambient:
        raise ValueError(
            f"U ambient dim {u.shape[0]} != sequence_length * vocab_size ({ambient})"
        )
    n_dirs = u.shape[1]
    if n_dirs == 0:
        return {}

    t_abs = np.abs(u).reshape(sequence_length, vocab_size, n_dirs)
    mass_per_pos = t_abs.sum(axis=1)
    pos_per_dir = mass_per_pos.argmax(axis=0)
    dir_ix = np.arange(n_dirs)
    rows = t_abs[pos_per_dir, :, dir_ix]
    mask = rows > token_threshold
    dir_hit, vocab_ix = np.where(mask)
    pos_ix = pos_per_dir[dir_hit].astype(np.int64, copy=False)
    vocab_ix = vocab_ix.astype(np.int64, copy=False)

    if pos_ix.size == 0:
        return {}

    # Dedupe (position, vocab), sort by position then vocab, split vocab ids per position.
    pairs = np.column_stack((pos_ix, vocab_ix))
    pairs = np.unique(pairs, axis=0)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    positions = pairs[:, 0]
    vocabs = pairs[:, 1]
    uniq_pos, starts = np.unique(positions, return_index=True)
    out: dict[int, tuple[int, ...]] = {}
    for i, p in enumerate(uniq_pos):
        start = int(starts[i])
        end = int(starts[i + 1]) if i + 1 < len(starts) else len(vocabs)
        out[int(p)] = tuple(int(x) for x in vocabs[start:end])
    return out


def substitutions_batch_from_jacobian(
    jac_batch: torch.Tensor | np.ndarray,
    *,
    sequence_length: int,
    vocab_size: int,
    direction_significance_threshold: float = 1e-3,
    min_number_of_directions: int = 5,
    token_threshold: float = 0.1,
) -> list[dict[int, tuple[int, ...]]]:
    """One substitution dict per batch row from Jacobians shaped ``(batch, ambient, latent)``."""
    if sequence_length < 1 or vocab_size < 2:
        # Need at least two tokens so vocab indexing is non-degenerate for callers.
        raise ValueError("sequence_length must be >= 1 and vocab_size must be >= 2")

    if isinstance(jac_batch, torch.Tensor):
        jac_np = np.asarray(jac_batch.detach().cpu().numpy(), dtype=np.float64)
    else:
        jac_np = np.asarray(jac_batch, dtype=np.float64)
    if jac_np.ndim != 3:
        raise ValueError(f"jacobian batch must be a 3D array, got shape {jac_np.shape}")

    batch, ambient, _ = jac_np.shape
    expected_ambient = sequence_length * vocab_size
    if ambient != expected_ambient:
        raise ValueError(
            "jacobian batch has wrong shape: expected "
            f"(batch, {expected_ambient}, latent), got {jac_np.shape}"
        )

    u_b, s_b, _vh = np.linalg.svd(jac_np, full_matrices=False)

    out: list[dict[int, tuple[int, ...]]] = []
    k = int(s_b.shape[1])
    for b in range(batch):
        s_row = s_b[b]
        n_above_threshold = int((s_row > direction_significance_threshold).sum())
        n_dirs = int(min(max(n_above_threshold, min_number_of_directions), k))
        out.append(
            _substitutions_from_u(
                u_b[b, :, :n_dirs],
                sequence_length=sequence_length,
                vocab_size=vocab_size,
                token_threshold=token_threshold,
            )
        )
    return out


def substitutions_from_encoded_batch(
    input_ids: torch.Tensor,
    encode: Callable[[torch.Tensor], torch.Tensor],
    jacobian_batch_fn: Callable[[torch.Tensor], torch.Tensor | np.ndarray],
    *,
    sequence_length: int,
    vocab_size: int,
    direction_significance_threshold: float = 1e-3,
    min_number_of_directions: int = 5,
    token_threshold: float = 0.1,
) -> list[dict[int, tuple[int, ...]]]:
    """Encode token ids to latents, Jacobian per batch row, then substitution dicts.

    ``sequence_length`` and ``vocab_size`` must match how ``jacobian_batch_fn`` flattens decoder
    outputs (same convention as the module docstring).
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}"
        )

    z = encode(input_ids)
    if z.ndim != 2:
        raise ValueError(
            f"encode must return [batch, latent], got shape {tuple(z.shape)}"
        )
    batch = input_ids.shape[0]
    if z.shape[0] != batch:
        raise ValueError(
            f"encode batch size {z.shape[0]} does not match input_ids batch {batch}"
        )
    jac = jacobian_batch_fn(z.detach())
    z_dim = int(z.shape[1])
    jac_last = (
        int(jac.shape[-1])
        if isinstance(jac, torch.Tensor)
        else int(np.asarray(jac).shape[-1])
    )
    if jac_last != z_dim:
        raise ValueError(
            f"jacobian trailing dimension {jac_last} does not match encode width {z_dim}"
        )

    subst_kw = dict(
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        direction_significance_threshold=direction_significance_threshold,
        min_number_of_directions=min_number_of_directions,
        token_threshold=token_threshold,
    )
    return substitutions_batch_from_jacobian(jac, **subst_kw)
