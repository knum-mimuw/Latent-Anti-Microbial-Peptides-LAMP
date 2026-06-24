from collections.abc import Callable
from typing import Any, Literal

import torch
from einops import rearrange


def decoder_jacobian(
    decoder_forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    jacobian_fn_mode: Literal["strict", "approx"],
    jacobian_fn_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    r"""input shape: (batch_dim, latent_dim), output shape: (batch_dim, ambient_dim, latent_dim)"""
    assert x.ndim == 2, ValueError(f"x should be 2D, got {x.ndim}D instead.")
    jacobian_fn_kwargs = jacobian_fn_kwargs or {}
    if jacobian_fn_mode == "strict":
        return decoder_jacobian_strict(decoder_forward, x, **jacobian_fn_kwargs)
    elif jacobian_fn_mode == "approx":
        return decoder_jacobian_approx(decoder_forward, x, **jacobian_fn_kwargs)


def decoder_jacobian_strict(
    decoder_forward: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    return rearrange(
        torch.autograd.functional.jacobian(decoder_forward, x)[0],
        "a b d -> b a d",
    )


def decoder_jacobian_approx(
    decoder_forward: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    jacobian_eps: float,
) -> torch.Tensor:
    assert x.ndim == 2, ValueError(f"x should be 2D, got {x.ndim}D instead.")
    latent_dim = x.shape[1]
    decoder_input_delta = torch.cat(
        [
            torch.eye(latent_dim, device=x.device) * jacobian_eps,
            torch.zeros((1, latent_dim), device=x.device),
        ],
        dim=0,
    )
    decoder_input = rearrange(
        rearrange(x, "b d -> b 1 d")
        + rearrange(decoder_input_delta, "d_plus_1 d -> 1 d_plus_1 d"),
        "b d_plus_1 d -> (b d_plus_1) d",
    )
    with torch.no_grad():
        decoder_output = decoder_forward(decoder_input)

    decoder_output = rearrange(
        decoder_output, "(b d_plus_1) d -> b d_plus_1 d", b=x.shape[0]
    )
    approx_jac = (decoder_output[:, :-1, :] - decoder_output[:, [-1], :]) / jacobian_eps

    return rearrange(approx_jac, "b d a -> b a d")


def decoder_second_derivative(
    decoder_forward: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor | None = None,
    mode: Literal["strict", "approx"] = "approx",
    kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    r"""Second directional derivative D''(z)[v1, v2].

    If v2 is None, computes D''(z)[v1, v1].
    Input shapes: z, v1, v2: (batch, latent_dim).
    Output shape: (batch, ambient_dim).
    """
    assert z.ndim == 2, ValueError(f"z should be 2D, got {z.ndim}D instead.")
    if v2 is None:
        v2 = v1
    kwargs = kwargs or {}
    if mode == "strict":
        return decoder_second_derivative_strict(decoder_forward, z, v1, v2)
    elif mode == "approx":
        return decoder_second_derivative_approx(decoder_forward, z, v1, v2, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


def decoder_second_derivative_approx(
    decoder_forward: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    field_eps: float = 1e-3,
) -> torch.Tensor:
    r"""Finite-difference approximation of D''(z)[v1, v2].

    Uses the mixed-derivative stencil (works for v1 == v2 as well):
    D''(z)[v1, v2] ~ (f(z+e*v1+e*v2) - f(z+e*v1) - f(z+e*v2) + f(z)) / e^2
    """
    assert z.ndim == 2, ValueError(f"z should be 2D, got {z.ndim}D instead.")
    batch = z.shape[0]
    ev1 = field_eps * v1
    ev2 = field_eps * v2
    stacked = torch.cat([z + ev1 + ev2, z + ev1, z + ev2, z], dim=0)
    with torch.no_grad():
        out = decoder_forward(stacked)
    out = out.view(4, batch, -1)
    return (out[0] - out[1] - out[2] + out[3]) / (field_eps**2)


def decoder_second_derivative_strict(
    decoder_forward: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> torch.Tensor:
    r"""Exact second directional derivative via nested autograd JVP."""

    def inner_jvp(z_: torch.Tensor) -> torch.Tensor:
        _, jvp_val = torch.autograd.functional.jvp(
            decoder_forward, z_, v1, create_graph=True
        )
        return jvp_val

    _, second_jvp = torch.autograd.functional.jvp(inner_jvp, z, v2)
    return second_jvp
