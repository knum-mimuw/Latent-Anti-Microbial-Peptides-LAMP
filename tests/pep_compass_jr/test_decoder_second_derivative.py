"""Tests for decoder_second_derivative (approx + strict)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pep_compass_jr.utils import (
    decoder_second_derivative,
    decoder_second_derivative_approx,
    decoder_second_derivative_strict,
)

LATENT_DIM = 4
AMBIENT_DIM = 6


def _quadratic_matrix(*, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(LATENT_DIM, AMBIENT_DIM, generator=g) * 0.3


def _quadratic_decoder(A: torch.Tensor):
    """f(z) = z^T A  applied element-wise as  sum_k z_k * A[k, :],
    but quadratic: f_j(z) = sum_{k,l} z_k * B[k,l,j] * z_l.
    We use f(z) = (Az)^2 element-wise so D''(z)[v1,v2] = 2 * (A^T v1) . (A^T v2)
    entry-wise product broadcast -- but that gives scalar not ambient.

    Simpler: f(z) = z @ A  gives f''= 0 (linear).

    Use truly quadratic: f_j(z) = z^T M_j z  for each output dim j.
    Then D''(z)[v1,v2]_j = v1^T (M_j + M_j^T) v2.
    """
    pass


class QuadraticDecoder:
    """f_j(z) = z^T M_j z.  D''[v1,v2]_j = v1^T (M_j + M_j^T) v2."""

    def __init__(self, latent_dim: int, ambient_dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.M = torch.randn(ambient_dim, latent_dim, latent_dim, generator=g) * 0.1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim)
        # out_j = z^T M_j z = einsum("bi, jik, bk -> bj", z, M, z)
        return torch.einsum("bi, jik, bk -> bj", z, self.M, z)

    def analytic_second_deriv(
        self, v1: torch.Tensor, v2: torch.Tensor
    ) -> torch.Tensor:
        # D''[v1,v2]_j = v1^T (M_j + M_j^T) v2
        M_sym = self.M + self.M.transpose(-2, -1)  # (ambient, latent, latent)
        # v1: (batch, latent), v2: (batch, latent)
        return torch.einsum("bi, jik, bk -> bj", v1, M_sym, v2)


@pytest.fixture()
def quad() -> QuadraticDecoder:
    return QuadraticDecoder(LATENT_DIM, AMBIENT_DIM)


@pytest.fixture()
def z_single() -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    return torch.randn(1, LATENT_DIM, generator=g)


@pytest.fixture()
def z_batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(42)
    return torch.randn(3, LATENT_DIM, generator=g)


@pytest.fixture()
def v1_single() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return torch.randn(1, LATENT_DIM, generator=g)


@pytest.fixture()
def v2_single() -> torch.Tensor:
    g = torch.Generator().manual_seed(13)
    return torch.randn(1, LATENT_DIM, generator=g)


def test_approx_matches_analytic(
    quad: QuadraticDecoder,
    z_single: torch.Tensor,
    v1_single: torch.Tensor,
    v2_single: torch.Tensor,
) -> None:
    expected = quad.analytic_second_deriv(v1_single, v2_single)
    got = decoder_second_derivative_approx(
        quad.forward, z_single, v1_single, v2_single, field_eps=1e-2
    )
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


def test_approx_field_derivative_matches_analytic(
    quad: QuadraticDecoder,
    z_single: torch.Tensor,
    v1_single: torch.Tensor,
) -> None:
    """v2 == v1 case (field derivative / acceleration)."""
    expected = quad.analytic_second_deriv(v1_single, v1_single)
    got = decoder_second_derivative_approx(
        quad.forward, z_single, v1_single, v1_single, field_eps=1e-2
    )
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)


def test_strict_matches_analytic(
    quad: QuadraticDecoder,
    z_single: torch.Tensor,
    v1_single: torch.Tensor,
    v2_single: torch.Tensor,
) -> None:
    expected = quad.analytic_second_deriv(v1_single, v2_single)
    got = decoder_second_derivative_strict(
        quad.forward, z_single, v1_single, v2_single
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_approx_matches_strict_nonlinear(
    z_single: torch.Tensor,
    v1_single: torch.Tensor,
    v2_single: torch.Tensor,
) -> None:
    """Non-linear decoder: approx and strict should agree within tolerance."""
    torch.manual_seed(99)
    mlp = nn.Sequential(
        nn.Linear(LATENT_DIM, 16),
        nn.Tanh(),
        nn.Linear(16, AMBIENT_DIM),
    )

    def fwd(z: torch.Tensor) -> torch.Tensor:
        return mlp(z)

    v1_unit = v1_single / torch.linalg.norm(v1_single)
    v2_unit = v2_single / torch.linalg.norm(v2_single)
    got_approx = decoder_second_derivative_approx(
        fwd, z_single, v1_unit, v2_unit, field_eps=1e-3
    )
    got_strict = decoder_second_derivative_strict(
        fwd, z_single, v1_unit, v2_unit
    )
    torch.testing.assert_close(got_approx, got_strict, atol=5e-2, rtol=5e-2)


def test_v2_none_defaults_to_v1(
    quad: QuadraticDecoder,
    z_single: torch.Tensor,
    v1_single: torch.Tensor,
) -> None:
    got_none = decoder_second_derivative(
        quad.forward, z_single, v1_single, v2=None, mode="approx"
    )
    got_explicit = decoder_second_derivative(
        quad.forward, z_single, v1_single, v2=v1_single, mode="approx"
    )
    torch.testing.assert_close(got_none, got_explicit)


def test_batched_approx(quad: QuadraticDecoder, z_batch: torch.Tensor) -> None:
    g = torch.Generator().manual_seed(55)
    v1 = torch.randn(3, LATENT_DIM, generator=g)
    v2 = torch.randn(3, LATENT_DIM, generator=g)
    expected = quad.analytic_second_deriv(v1, v2)
    got = decoder_second_derivative_approx(
        quad.forward, z_batch, v1, v2, field_eps=1e-2
    )
    torch.testing.assert_close(got, expected, atol=1e-2, rtol=1e-2)
