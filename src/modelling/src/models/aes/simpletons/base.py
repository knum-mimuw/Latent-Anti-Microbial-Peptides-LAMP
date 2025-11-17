# TODO: Add configurable encoder representation wrapper that extracts the desired hidden state
#       before projection to latent space (e.g. CLS, BOS/EOS, pooled, custom extractor).
# TODO: Introduce a Hugging Face tokenizer subclass tailored for amino-acid sequences with
#       optional CLS/BOS/EOS/PAD tokens, and wire it into the configuration/export flow.
# TODO: Externalize the tokenizer/repr configuration so exported Hugging Face packages can
#       reconstruct the exact preprocessing and embedding logic independently of Lightning.

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from ...utils.importing import load_model_from_huggingface


class HFBackboneConfig(BaseModel):
    """Configuration wrapper for a single Hugging Face backbone."""

    model_class_path: str = Field(
        ..., description="Import path to the model class (e.g. transformers.AutoModel)"
    )
    pretrained_model_name_or_path: Optional[str] = Field(
        None, description="Hugging Face model identifier or local path"
    )
    config_class_path: Optional[str] = Field(
        None, description="Optional import path to config class"
    )
    load_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments forwarded to from_pretrained",
    )
    config_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded when instantiating configs without pretrained weights.",
    )
    load_pretrained: bool = Field(
        default=True,
        description="Whether to load pretrained weights. If False, instantiate model from config.",
    )

    model_config = ConfigDict(extra="allow")


class VAEConfig(BaseModel):
    """Top-level configuration for the VAE."""

    encoder: Optional[HFBackboneConfig] = Field(
        None,
        description="Optional encoder configuration. If omitted, the decoder config is reused.",
    )
    decoder: HFBackboneConfig = Field(
        ...,
        description="Decoder configuration (also used for encoder when encoder is omitted).",
    )
    d_decoder: int = Field(
        ...,
        description="Hidden size of the decoder backbone.",
    )
    d_encoder: Optional[int] = Field(
        None,
        description="Hidden size of the encoder backbone. Defaults to d_decoder when encoder config is not provided.",
    )
    latent_dim: int
    kl_weight: float = 1.0
    temperature: float = 1.0
    encoder_n_pooling_tokens: int = 1

    model_config = ConfigDict(extra="allow")


class VAE(nn.Module):
    """Simple VAE using encoder and decoder from Hugging Face."""

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.cfg = config

        self.d_decoder = config.d_decoder
        self.d_encoder = config.d_encoder or config.d_decoder
        self.latent_dim = config.latent_dim
        self.kl_weight = config.kl_weight
        self.temperature = config.temperature
        self.encoder_n_pooling_tokens = config.encoder_n_pooling_tokens

        # Determine encoder config (fallback to decoder config if not provided)
        encoder_cfg = config.encoder or config.decoder

        # Load encoder from Hugging Face
        self.encoder = load_model_from_huggingface(
            model_class_path=encoder_cfg.model_class_path,
            pretrained_model_name_or_path=encoder_cfg.pretrained_model_name_or_path,
            config_class_path=encoder_cfg.config_class_path,
            load_pretrained=encoder_cfg.load_pretrained,
            config_kwargs=encoder_cfg.config_kwargs,
            **encoder_cfg.load_kwargs,
        )

        # Load decoder from Hugging Face
        self.decoder = load_model_from_huggingface(
            model_class_path=config.decoder.model_class_path,
            pretrained_model_name_or_path=config.decoder.pretrained_model_name_or_path,
            config_class_path=config.decoder.config_class_path,
            load_pretrained=config.decoder.load_pretrained,
            config_kwargs=config.decoder.config_kwargs,
            **config.decoder.load_kwargs,
        )

        # VAE encoder: maps encoder output to latent mean and log_std
        self.latent_encoder = nn.Linear(
            self.d_encoder * self.encoder_n_pooling_tokens, self.latent_dim * 2
        )

        # VAE decoder: maps latent to decoder input dimension
        self.latent_decoder = nn.Linear(self.latent_dim, self.d_decoder)

        # Pooling tokens for encoder
        self.encoder_pooling_tokens = nn.Parameter(
            torch.empty(self.encoder_n_pooling_tokens, self.d_encoder)
        )
        nn.init.normal_(self.encoder_pooling_tokens, mean=0.0, std=0.02)

    def _sample_gaussian(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample a single latent vector from Gaussian distribution with temperature scaling."""
        std = torch.exp(log_std) * temperature
        eps = torch.randn_like(mean)
        return mean + eps * std

    def _compute_kl_divergence(
        self, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and unit Gaussian prior."""
        std = torch.exp(log_std)
        kl = 0.5 * (mean.pow(2) + std.pow(2) - 1 - 2 * log_std).sum(dim=-1)
        return kl

    def _encoder_forward(
        self, encoder_input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder."""
        # HF models typically return a tuple or object with last_hidden_state
        output = self.encoder(
            inputs_embeds=encoder_input, attention_mask=attention_mask
        )
        # Extract hidden states
        if isinstance(output, tuple):
            return output[0]
        elif hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        else:
            return output

    def _decoder_forward(
        self,
        decoder_input: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through decoder."""
        # Try to call with encoder_hidden_states if available
        try:
            output = self.decoder(
                inputs_embeds=decoder_input,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )
        except TypeError:
            # Fallback to just inputs_embeds
            output = self.decoder(
                inputs_embeds=decoder_input, attention_mask=attention_mask
            )
        # Extract hidden states
        if isinstance(output, tuple):
            return output[0]
        elif hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        else:
            return output

    def encode(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent representation.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            temperature: Optional temperature for sampling (defaults to self.temperature)

        Returns:
            Tuple of (latent samples, mean, log_std)
        """
        if temperature is None:
            temperature = self.temperature

        batch_size = x.shape[0]

        # Add pooling tokens
        pooling_tokens = self.encoder_pooling_tokens.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        encoder_input = torch.cat([pooling_tokens, x], dim=1)

        # Create attention mask for pooling tokens (not masked)
        if attention_mask is not None:
            pooling_mask = torch.ones(
                batch_size,
                self.encoder_n_pooling_tokens,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_attention_mask = torch.cat([pooling_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None

        # Encode through HF encoder
        encoder_out = self._encoder_forward(encoder_input, full_attention_mask)

        # Extract pooling token outputs
        pooling_output = encoder_out[:, : self.encoder_n_pooling_tokens]

        # Flatten pooling tokens
        pooling_flat = pooling_output.view(batch_size, -1)

        # Map to latent space
        latent_params = self.latent_encoder(pooling_flat)
        mean, log_std = latent_params.chunk(2, dim=-1)

        # Sample from latent
        z = self._sample_gaussian(mean, log_std, temperature)

        return z, mean, log_std

    def decode(
        self,
        z: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent representation to output.

        Args:
            z: Latent tensor of shape [batch_size, n_samples, latent_dim] or [batch_size, latent_dim]
            decoder_input: Optional decoder input embeddings
            attention_mask: Optional attention mask

        Returns:
            Decoded output embeddings
        """
        # Map latent to decoder dimension
        if z.ndim == 2:
            latent_decoded = self.latent_decoder(z)
        else:
            # Flatten batch and sample dimensions
            batch_size, n_samples, latent_dim = z.shape
            z_flat = z.view(-1, latent_dim)
            latent_decoded_flat = self.latent_decoder(z_flat)
            latent_decoded = latent_decoded_flat.view(batch_size, n_samples, -1)

        # If decoder_input is provided, use it; otherwise use latent_decoded as initial input
        if decoder_input is not None:
            # Concatenate latent with decoder input
            if latent_decoded.ndim == 2:
                decoder_input_with_latent = torch.cat(
                    [latent_decoded.unsqueeze(1), decoder_input], dim=1
                )
            else:
                # Repeat decoder_input for each latent sample
                decoder_input_expanded = decoder_input.unsqueeze(1).repeat(
                    1, latent_decoded.shape[1], 1, 1
                )
                decoder_input_expanded = decoder_input_expanded.view(
                    -1, decoder_input.shape[1], decoder_input.shape[2]
                )
                latent_decoded_expanded = latent_decoded.view(
                    -1, 1, latent_decoded.shape[2]
                )
                decoder_input_with_latent = torch.cat(
                    [latent_decoded_expanded, decoder_input_expanded], dim=1
                )
        else:
            decoder_input_with_latent = latent_decoded

        # Decode through HF decoder
        decoder_out = self._decoder_forward(
            decoder_input_with_latent,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
        )
        return decoder_out

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through VAE.

        Args:
            x: Input embeddings of shape [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask for input
            decoder_input: Optional decoder input embeddings

        Returns:
            Tuple of (reconstructed output, loss dictionary)
        """
        # Encode (use temperature=1.0 for training to maintain proper KL)
        z, mean, log_std = self.encode(x, attention_mask, temperature=1.0)

        # Decode
        x_reconstructed = self.decode(z, decoder_input, attention_mask)

        # Compute KL divergence loss
        kl_per_sample = self._compute_kl_divergence(mean, log_std)
        kl_loss = kl_per_sample.mean()

        return x_reconstructed, {
            "kl_loss": kl_loss,
            "total_latent": kl_loss * self.kl_weight,
        }
