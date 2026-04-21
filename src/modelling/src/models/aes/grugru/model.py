import math

import torch
import torch.nn as nn
from einops import rearrange
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .config import GRUVAEConfig


class GRUVAEEncoder(PreTrainedModel):
    """GRU-based VAE encoder. Independently loadable via ``from_pretrained``."""

    config_class = GRUVAEConfig

    def __init__(self, config: GRUVAEConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )

        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.encoder_hidden_size,
            num_layers=config.encoder_num_layers,
            batch_first=True,
            bidirectional=config.encoder_bidirectional,
            dropout=config.encoder_dropout if config.encoder_num_layers > 1 else 0,
        )

        encoder_output_dim = config.encoder_hidden_size * (
            2 if config.encoder_bidirectional else 1
        )
        self.mean_linear = nn.Linear(encoder_output_dim, config.latent_dim)
        self.log_std_linear = nn.Linear(encoder_output_dim, config.latent_dim)

        self.post_init()

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode token sequences to Gaussian parameters.

        Args:
            input_ids: ``[batch, seq_len]``

        Returns:
            ``(mean, log_std)`` each ``[batch, latent_dim]``.
        """
        embeddings = self.embedding(input_ids)
        _output, hidden = self.gru(embeddings)

        if self.config.encoder_bidirectional:
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]

        return self.mean_linear(last_hidden), self.log_std_linear(last_hidden)


class GRUVAEDecoder(PreTrainedModel, GenerationMixin):
    """GRU-based VAE decoder with HuggingFace ``.generate()`` support.

    Independently loadable via ``from_pretrained``.  Generation example::

        dec = GRUVAEDecoder.from_pretrained(...)
        bos = torch.full((B, 1), bos_id, device=device)
        out = dec.generate(bos, z=z, max_new_tokens=50, do_sample=True)
    """

    config_class = GRUVAEConfig

    def __init__(self, config: GRUVAEConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )

        self.latent_proj = nn.Linear(
            config.latent_dim,
            config.decoder_hidden_size * config.decoder_num_layers,
        )

        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.decoder_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=config.decoder_dropout if config.decoder_num_layers > 1 else 0,
        )

        self.output_proj = nn.Linear(config.decoder_hidden_size, config.vocab_size)

        self.post_init()

    # ------------------------------------------------------------------
    # Latent → GRU hidden state
    # ------------------------------------------------------------------

    def initial_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """Build GRU initial hidden state from latent ``z``.

        Args:
            z: ``[batch, latent_dim]``

        Returns:
            ``[decoder_num_layers, batch, decoder_hidden_size]``
        """
        h_0_flat = self.latent_proj(z)
        return rearrange(
            h_0_flat,
            "b (l h) -> l b h",
            l=self.config.decoder_num_layers,
            h=self.config.decoder_hidden_size,
        ).contiguous()

    def forward_latent_positions(
        self,
        z: torch.Tensor,
        num_steps: int,
    ) -> CausalLMOutputWithPast:
        """Decode a fixed-length sequence without teacher forcing.

        The GRU is fed sinusoidal positional encodings only; ``z`` initializes
        the hidden state via :meth:`initial_hidden`. Training via
        :class:`GRUVAE` uses this path; :meth:`forward` remains for
        token-conditioned generation (e.g. ``.generate()``).

        Args:
            z: ``[batch, latent_dim]``
            num_steps: Number of output positions ``T`` (GRU sequence length).

        Returns:
            ``CausalLMOutputWithPast`` with ``logits`` ``[batch, T, vocab_size]``.
        """
        batch = z.shape[0]
        hidden = self.initial_hidden(z)
        if num_steps < 1:
            logits = z.new_zeros(batch, 0, self.config.vocab_size)
            return CausalLMOutputWithPast(logits=logits, past_key_values=hidden)

        pe = _sinusoidal_position_encoding(
            num_steps,
            self.config.embedding_dim,
            device=z.device,
            dtype=z.dtype,
        ).expand(batch, -1, -1)
        output, new_hidden = self.gru(pe, hidden)
        logits = self.output_proj(output)
        return CausalLMOutputWithPast(logits=logits, past_key_values=new_hidden)

    # ------------------------------------------------------------------
    # One-step decoding (manual autoregressive loops)
    # ------------------------------------------------------------------

    def decode_step(
        self,
        hidden: torch.Tensor,
        input_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the decoder one timestep given a precomputed embedding.

        Args:
            hidden: ``[decoder_num_layers, batch, decoder_hidden_size]``
            input_embedding: ``[batch, embedding_dim]`` or ``[batch, 1, embedding_dim]``

        Returns:
            ``(logits, new_hidden)`` where logits is ``[batch, vocab_size]``.
        """
        if input_embedding.dim() == 2:
            x = input_embedding.unsqueeze(1)
        elif input_embedding.dim() == 3:
            if input_embedding.shape[1] != 1:
                msg = (
                    "decode_step expects sequence length 1 when input is 3D, "
                    f"got shape {tuple(input_embedding.shape)}"
                )
                raise ValueError(msg)
            x = input_embedding
        else:
            msg = f"input_embedding must be 2D or 3D, got dim {input_embedding.dim()}"
            raise ValueError(msg)

        output, new_hidden = self.gru(x, hidden)
        logits = self.output_proj(output[:, -1, :])
        return logits, new_hidden

    # ------------------------------------------------------------------
    # HuggingFace forward (generation-compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        z: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Decode tokens conditioned on latent ``z`` or cached hidden state.

        On the first generation step ``z`` is provided and ``past_key_values``
        is ``None``; on subsequent steps the GRU hidden state round-trips via
        ``past_key_values``.

        Args:
            input_ids: ``[batch, seq_len]``
            z: ``[batch, latent_dim]`` (first step only).
            past_key_values: GRU hidden ``[layers, batch, hidden]`` from the
                previous step.

        Returns:
            ``CausalLMOutputWithPast`` with ``.logits`` ``[batch, seq_len, vocab_size]``
            and ``.past_key_values`` (new GRU hidden state).
        """
        if past_key_values is not None:
            hidden = past_key_values
        else:
            hidden = self.initial_hidden(z)

        embeddings = self.embedding(input_ids)
        output, new_hidden = self.gru(embeddings, hidden)
        logits = self.output_proj(output)

        return CausalLMOutputWithPast(logits=logits, past_key_values=new_hidden)

    # ------------------------------------------------------------------
    # GenerationMixin hooks
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "past_key_values": past_key_values, "z": z}

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False

    @staticmethod
    def _reorder_cache(
        past_key_values: torch.Tensor, beam_idx: torch.LongTensor
    ) -> torch.Tensor:
        """Reindex the GRU hidden state for beam search."""
        return past_key_values.index_select(1, beam_idx)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _sinusoidal_position_encoding(
    length: int,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Transformer-style sinusoidal positions ``[1, length, d_model]``.

    ``d_model`` must be even (sin on even indices, cos on odd).
    """
    if d_model % 2 != 0:
        msg = f"sinusoidal PE requires even d_model, got {d_model}"
        raise ValueError(msg)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _sample_gaussian(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mean)
    return mean + eps * std


# ----------------------------------------------------------------------
# Composite VAE (training entry-point)
# ----------------------------------------------------------------------


class GRUVAE(PreTrainedModel):
    """Composite GRU VAE for training.

    Composes :class:`GRUVAEEncoder` and :class:`GRUVAEDecoder`.  The
    ``forward`` method runs the full encode → sample → decode pipeline and
    returns the training-oriented dict consumed by ``MetaModule``.

    For generation, use the decoder directly::

        vae = GRUVAE.from_pretrained(...)
        mean, log_std = vae.encoder(input_ids)
        z = ...
        generated = vae.decoder.generate(bos, z=z, max_new_tokens=50)
    """

    config_class = GRUVAEConfig
    base_model_prefix = "gruvae"

    def __init__(self, config: GRUVAEConfig):
        super().__init__(config)
        self.encoder = GRUVAEEncoder(config)
        self.decoder = GRUVAEDecoder(config)
        self.post_init()

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience delegate to ``self.encoder(input_ids)``."""
        return self.encoder(input_ids)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Full VAE training forward.

        The decoder is unrolled for ``S-1`` steps with **no teacher forcing**:
        each step's GRU input is a sinusoidal positional encoding; the latent
        ``z`` initializes the GRU hidden state. Token-conditioned decoding for
        sampling remains on :meth:`GRUVAEDecoder.forward` / ``.generate()``.

        Args:
            input_ids: ``[batch, seq_len]`` with format
                ``[BOS, t1, ..., tn, EOS, PAD...]``
            **kwargs: Ignored (e.g. ``attention_mask``).

        Returns:
            Dict with keys ``logits`` ``[B, V, S-1]``, ``target`` ``[B, S-1]``,
            ``mean`` ``[B, latent_dim]``, ``log_std`` ``[B, latent_dim]``.
        """
        mean, log_std = self.encode(input_ids)
        z = _sample_gaussian(mean, torch.exp(log_std))

        num_steps = input_ids.shape[1] - 1
        target = input_ids[:, 1:]

        decoder_out = self.decoder.forward_latent_positions(z=z, num_steps=num_steps)
        logits = rearrange(decoder_out.logits, "b s v -> b v s")

        return {
            "logits": logits,
            "target": target,
            "mean": mean,
            "log_std": log_std,
        }
