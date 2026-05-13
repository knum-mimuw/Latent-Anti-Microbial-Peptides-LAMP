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
    jacobian_fn_kwargs = jacobian_fn_kwargs if jacobian_fn_kwargs is not None else {}
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


def softmax_probs_jacobian_fn(
    decode_logits: Callable[[torch.Tensor], torch.Tensor],
    *,
    sequence_length: int,
    vocab_size: int,
    jacobian_mode: Literal["strict", "approx"] = "approx",
    jacobian_eps: float = 1e-4,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build ``jacobian_batch_fn(z)`` for ``substitutions_from_encoded_batch``.

    ``z`` must have shape ``[batch, latent_dim]`` with ``batch >= 1``. Returns the Jacobian of
    ``softmax(decode_logits(z), dim=-1)`` with respect to ``z``, with shape
    ``[batch, sequence_length * vocab_size, latent_dim]``.
    """
    ambient = sequence_length * vocab_size

    def decoder_probs_flat(z_in: torch.Tensor) -> torch.Tensor:
        logits = decode_logits(z_in)
        if logits.ndim != 3 or logits.shape[1:] != (sequence_length, vocab_size):
            raise ValueError(
                f"decode_logits must return [batch, {sequence_length}, {vocab_size}], "
                f"got {tuple(logits.shape)}"
            )
        probs = torch.softmax(logits, dim=-1)
        return rearrange(probs, "batch seq vocab -> batch (seq vocab)")

    def jac_fn(z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[0] < 1:
            raise ValueError("z must have shape [batch, latent_dim] with batch >= 1")
        if jacobian_mode == "approx":
            z_in = z.detach().clone()
            jac = decoder_jacobian(
                decoder_probs_flat,
                z_in,
                "approx",
                {"jacobian_eps": jacobian_eps},
            )
        else:
            z_in = z.detach().clone().requires_grad_(True)
            jac = decoder_jacobian(decoder_probs_flat, z_in, "strict", None)
        if jac.ndim != 3 or jac.shape[1] != ambient:
            raise RuntimeError(
                f"Jacobian ambient dim {jac.shape[1]} != {ambient} "
                "(sequence_length * vocab_size)"
            )
        return jac

    return jac_fn
