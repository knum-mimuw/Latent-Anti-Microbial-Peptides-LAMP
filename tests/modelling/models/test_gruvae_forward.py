"""Smoke tests for GRUVAE training forward (latent + positional decoder)."""

import pytest
import torch

from modelling.src.models.aes.grugru import GRUVAE, GRUVAEConfig


def _tiny_gruvae() -> tuple[GRUVAE, GRUVAEConfig]:
    cfg = GRUVAEConfig(
        vocab_size=29,
        embedding_dim=100,
        latent_dim=16,
        encoder_hidden_size=32,
        encoder_num_layers=1,
        encoder_bidirectional=False,
        decoder_hidden_size=32,
        decoder_num_layers=1,
    )
    return GRUVAE(cfg).eval(), cfg


def test_gruvae_forward_shapes() -> None:
    model, cfg = _tiny_gruvae()

    b, s = 2, 12
    input_ids = torch.randint(1, cfg.vocab_size, (b, s))
    out = model(input_ids)

    v = cfg.vocab_size
    assert out["logits"].shape == (b, v, s - 1)
    assert out["target"].shape == (b, s - 1)
    assert out["mean"].shape == (b, cfg.latent_dim)
    assert out["log_std"].shape == (b, cfg.latent_dim)


def test_decoder_forward_latent_positions_is_token_free() -> None:
    """Training decode path feeds PE only; for fixed ``z`` and length, logits
    do not depend on any token sequence (only on ``z`` and step index).
    """
    model, cfg = _tiny_gruvae()

    z = torch.randn(2, cfg.latent_dim)
    num_steps = 7
    with torch.no_grad():
        once = model.decoder.forward_latent_positions(z, num_steps=num_steps).logits
        twice = model.decoder.forward_latent_positions(z, num_steps=num_steps).logits
    assert torch.allclose(once, twice)


def test_decoder_generate_runs_with_latent() -> None:
    model, cfg = _tiny_gruvae()

    b = 2
    bos_id = 1
    bos = torch.full((b, 1), bos_id, dtype=torch.long)
    z = torch.randn(b, cfg.latent_dim)

    with torch.no_grad():
        out = model.decoder.generate(
            bos,
            z=z,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=cfg.padding_idx,
            bos_token_id=bos_id,
        )

    assert out.dtype == torch.long
    assert out.shape == (b, 1 + 5)


@pytest.mark.parametrize(
    ("gen_kwargs", "expected_batch"),
    [
        ({"do_sample": False, "num_beams": 3}, 2),
        (
            {
                "do_sample": False,
                "num_beams": 2,
                "repetition_penalty": 1.2,
                "length_penalty": 1.5,
            },
            2,
        ),
        (
            {
                "do_sample": True,
                "temperature": 0.9,
                "top_k": 10,
            },
            2,
        ),
        (
            {
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 20,
                "temperature": 1.0,
            },
            2,
        ),
        (
            {
                "do_sample": False,
                "num_beams": 4,
                "num_return_sequences": 2,
            },
            4,
        ),
        (
            {
                "do_sample": False,
                "num_beams": 2,
                "no_repeat_ngram_size": 2,
            },
            2,
        ),
    ],
)
def test_decoder_generate_hf_options(
    gen_kwargs: dict,
    expected_batch: int,
) -> None:
    """HuggingFace ``generate`` paths used in practice (beam, penalties, sampling)."""
    model, cfg = _tiny_gruvae()
    b = 2
    bos_id = 1
    max_new = 4
    bos = torch.full((b, 1), bos_id, dtype=torch.long)
    torch.manual_seed(42)
    z = torch.randn(b, cfg.latent_dim)

    with torch.no_grad():
        out = model.decoder.generate(
            bos,
            z=z,
            max_new_tokens=max_new,
            pad_token_id=cfg.padding_idx,
            eos_token_id=2,
            bos_token_id=bos_id,
            **gen_kwargs,
        )

    assert out.dtype == torch.long
    assert out.shape == (expected_batch, 1 + max_new)
