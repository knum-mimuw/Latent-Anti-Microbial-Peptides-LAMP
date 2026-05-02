"""Hugging Face model wrapper for HydrAMP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from .config import HydrAMPConfig
from .hydramp import HydrAMPDecoder, HydrAMPEncoder


@dataclass
class HydrAMPOutput(ModelOutput):
    """HydrAMP forward outputs."""

    logits: torch.Tensor | None = None
    mean: torch.Tensor | None = None
    log_std: torch.Tensor | None = None


class HydrAMPModel(PreTrainedModel):
    """HydrAMP model with HF `AutoModel` compatibility."""

    config_class = HydrAMPConfig
    base_model_prefix = "hydramp"

    def __init__(self, config: HydrAMPConfig) -> None:
        super().__init__(config)
        if len(config.default_condition) != config.condition_dim:
            raise ValueError(
                f"default_condition must contain {config.condition_dim} values, got {len(config.default_condition)}."
            )

        self.encoder = HydrAMPEncoder(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            latent_dim=config.latent_dim,
            sequence_length=config.sequence_length,
            gru_hidden_size=config.encoder_hidden_size,
        )
        self.decoder = HydrAMPDecoder(
            sequence_length=config.sequence_length,
            latent_dim=config.latent_dim,
            condition_dim=config.condition_dim,
            hidden_size=config.decoder_hidden_size,
            vocab_size=config.vocab_size,
        )
        self.register_buffer(
            "default_condition",
            torch.tensor(config.default_condition, dtype=torch.float32),
            persistent=False,
        )
        self.post_init()

    def forward_latent_positions(
        self,
        z: torch.Tensor,
        num_steps: int | None = None,
        condition: torch.Tensor | None = None,
        *,
        return_logits: bool = True,
    ) -> CausalLMOutputWithPast:
        """Decode latent vectors to sequence distributions (GRUVAE-style API).

        Output length is fixed to ``config.sequence_length``. If ``num_steps`` is
        passed, it must equal that value.
        """
        fixed = self.config.sequence_length
        if num_steps is None:
            num_steps = fixed
        elif num_steps != fixed:
            msg = f"HydrAMP decoder length is fixed at {fixed}; got num_steps={num_steps}."
            raise ValueError(msg)

        if condition is None:
            condition = self.default_condition.unsqueeze(0).expand(z.shape[0], -1)
        condition = condition.to(device=z.device, dtype=z.dtype)
        decoder_input = torch.cat([z, condition], dim=-1)
        out = self.decoder(
            decoder_input,
            return_logits=return_logits,
            gumbel_temperature=self.config.temperature,
        )
        return CausalLMOutputWithPast(logits=out, past_key_values=None)

    def decode_to_token_ids(
        self,
        z: torch.Tensor,
        num_steps: int | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy token IDs from latent ``z`` (argmax over vocabulary per position)."""
        logits = self.forward_latent_positions(
            z, num_steps=num_steps, condition=condition, return_logits=True
        ).logits
        assert logits is not None
        return logits.argmax(dim=-1)

    def forward(self, input_ids: torch.Tensor, **_: object) -> HydrAMPOutput:
        """Run encode + deterministic decode for reconstruction."""
        mean, log_std = self.encoder.encode(input_ids)
        logits = self.forward_latent_positions(mean, return_logits=True).logits
        return HydrAMPOutput(logits=logits, mean=mean, log_std=log_std)
