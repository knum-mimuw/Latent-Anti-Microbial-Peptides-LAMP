"""Tests for tangent-space substitution helpers (batched Jacobian + SVD)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from einops import rearrange

from pep_compass_jr.tangent_space_mutation_proposer import (
    substitutions_batch_from_jacobian,
    substitutions_from_encoded_batch,
)
from pep_compass_jr.utils import softmax_probs_jacobian_fn


def _linear_weight(*, latent_dim: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(latent_dim, seq_len * vocab_size, generator=g) * 0.05


def _assert_substitution_dict(
    d: dict[int, tuple[int, ...]],
    *,
    sequence_length: int,
    require_nonempty: bool,
) -> None:
    if require_nonempty:
        assert len(d) >= 1
    for pos, toks in d.items():
        assert isinstance(pos, int)
        assert 0 <= pos < sequence_length
        assert isinstance(toks, tuple)
        assert len(toks) >= 1
        assert all(isinstance(x, int) for x in toks)
        assert list(toks) == sorted(toks)


def test_substitutions_batch_returns_per_position_dicts() -> None:
    batch, seq_in, latent_dim = 2, 5, 3
    s_len, v_size = 2, 4
    w = _linear_weight(latent_dim=latent_dim, seq_len=s_len, vocab_size=v_size)

    def encode(ids: torch.Tensor) -> torch.Tensor:
        assert ids.shape == (batch, seq_in)
        return ids[:, :latent_dim].to(dtype=torch.float32)

    def decode_logits(z: torch.Tensor) -> torch.Tensor:
        flat = z @ w.to(z.device, dtype=z.dtype)
        return rearrange(flat, "b (s v) -> b s v", s=s_len, v=v_size)

    jac_batch_fn = softmax_probs_jacobian_fn(
        decode_logits,
        sequence_length=s_len,
        vocab_size=v_size,
        jacobian_mode="approx",
        jacobian_eps=1e-3,
    )
    input_ids = torch.randint(0, v_size, (batch, seq_in), dtype=torch.long)
    out = substitutions_from_encoded_batch(
        input_ids,
        encode,
        jac_batch_fn,
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
    )
    assert len(out) == batch
    for proposals in out:
        _assert_substitution_dict(proposals, sequence_length=s_len, require_nonempty=True)


def test_substitutions_differ_for_distinct_rank_one_jacobians() -> None:
    """Rank-one Jacobians with left singular vectors on different ambient rows → different positions."""
    s_len, v_size, latent = 2, 3, 2
    ambient = s_len * v_size

    j_a = np.zeros((ambient, latent), dtype=np.float64)
    j_a[1, 0] = 1.0
    j_b = np.zeros((ambient, latent), dtype=np.float64)
    j_b[4, 0] = 1.0

    kw = dict(
        sequence_length=s_len,
        vocab_size=v_size,
        min_number_of_directions=1,
        token_threshold=0.0,
    )
    stacked = np.stack([j_a, j_b], axis=0)
    outs = substitutions_batch_from_jacobian(stacked, **kw)
    assert outs[0] != outs[1]
    assert set(outs[0].keys()) != set(outs[1].keys())


def test_raises_when_jacobian_width_mismatches_encode() -> None:
    def encode(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.shape[0], 3, dtype=torch.float32)

    def jac_batch_fn(z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], 2, 7, dtype=torch.float32)

    with pytest.raises(ValueError, match="trailing dimension"):
        substitutions_from_encoded_batch(
            torch.zeros((1, 4), dtype=torch.long),
            encode,
            jac_batch_fn,
            sequence_length=1,
            vocab_size=2,
        )


def test_raises_on_bad_jacobian_shape() -> None:
    bad = np.zeros((1, 3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="wrong shape"):
        substitutions_batch_from_jacobian(
            bad,
            sequence_length=1,
            vocab_size=2,
        )


def test_constant_jacobian_numpy() -> None:
    """Deterministic path: fixed Jacobian matrix per batch row."""

    def encode(ids: torch.Tensor) -> torch.Tensor:
        _ = ids
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)

    ambient, latent = 4, 2
    j_fix = np.zeros((ambient, latent), dtype=np.float64)
    j_fix[:, 0] = 1.0

    def jac_batch_fn(z: torch.Tensor) -> np.ndarray:
        b = z.shape[0]
        return np.stack([j_fix] * b, axis=0)

    out = substitutions_from_encoded_batch(
        torch.tensor([[0, 1]], dtype=torch.long),
        encode,
        jac_batch_fn,
        sequence_length=2,
        vocab_size=2,
        min_number_of_directions=1,
        token_threshold=0.0,
    )
    assert len(out) == 1
    _assert_substitution_dict(out[0], sequence_length=2, require_nonempty=True)
