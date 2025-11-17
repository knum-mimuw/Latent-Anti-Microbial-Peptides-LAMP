"""Simple GRU encoder-decoder VAE for sequence modeling."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from einops import rearrange


class GRUConfig(BaseModel):
    batch_first: bool = Field(True, description="Whether to use batch first indexing")
    hidden_size: int = Field(128, description="GRU hidden dimension")
    num_layers: int = Field(2, description="Number of GRU layers")
    bidirectional: bool = Field(True, description="Use bidirectional encoder")
    dropout: float = Field(0.1, description="Dropout rate")

    model_config = ConfigDict(extra="allow")


class EmbeddingConfig(BaseModel):
    num_embeddings: int = Field(..., description="Number of embedding embeddings")
    embedding_dim: int = Field(100, description="Embedding dimension")
    padding_idx: int = Field(0, description="Padding index")

    model_config = ConfigDict(extra="allow")


class GRUVAEConfig(BaseModel):
    """Configuration for GRU-based VAE."""

    embedding: EmbeddingConfig = Field(..., description="Embedding configuration")
    latent_dim: int = Field(64, description="Latent space dimension")
    encoder: GRUConfig = Field(..., description="Encoder configuration")
    decoder: GRUConfig = Field(..., description="Decoder configuration")


class GRUEncoder(nn.Module):
    """GRU-based encoder for VAE."""

    def __init__(self, config: GRUVAEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(**config.embedding.model_dump())
        self.gru = nn.GRU(**config.encoder.model_dump())

        # Output dimension depends on bidirectional
        encoder_output_dim = config.encoder.hidden_size * (
            2 if config.encoder.bidirectional else 1
        )

        # Project to latent mean and log_std
        self.mean_linear = nn.Linear(encoder_output_dim, config.latent_dim)
        self.log_std_linear = nn.Linear(encoder_output_dim, config.latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent parameters.

        Args:
            x: Input token sequence [batch_size, seq_len]

        Returns:
            mean: Latent mean [batch_size, latent_dim]
            log_std: Latent log standard deviation [batch_size, latent_dim]
        """
        # Embed input
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Encode through GRU
        output, hidden = self.gru(embeddings)

        # Use last hidden state
        # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
        if self.config.encoder.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            forward_hidden = hidden[-2]  # [batch_size, hidden_dim]
            backward_hidden = hidden[-1]  # [batch_size, hidden_dim]
            last_hidden = torch.cat(
                [forward_hidden, backward_hidden], dim=-1
            )  # [batch_size, hidden_dim*2]
        else:
            last_hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Project to latent parameters
        mean = self.mean_linear(last_hidden)
        log_std = self.log_std_linear(last_hidden)

        return mean, log_std


class GRUDecoder(nn.Module):
    """GRU-based decoder for VAE."""

    def __init__(self, config: GRUVAEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(**config.embedding.model_dump())

        # Project latent to initial hidden state
        self.latent_proj = nn.Linear(
            config.latent_dim,
            config.decoder.hidden_size * config.decoder.num_layers,
        )

        self.gru = nn.GRU(**config.decoder.model_dump())

        self.output_proj = nn.Linear(
            config.decoder.hidden_size, config.embedding.num_embeddings
        )

    def forward(self, z: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to sequence.

        Args:
            z: Latent vector [batch_size, latent_dim]
            input: Input sequence [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size = z.shape[0]

        # Project latent to initial hidden state
        h_0_flat = self.latent_proj(z)  # [batch_size, hidden_dim * num_layers]
        h_0 = rearrange(
            h_0_flat,
            "b (l h) -> l b h",
            l=self.config.decoder.num_layers,
            h=self.config.decoder.hidden_size,
        )  # [num_layers, batch_size, hidden_dim]

        # Teacher forcing: use input sequence
        embeddings = self.embedding(input)  # [batch_size, seq_len, embedding_dim]
        output, _ = self.gru(embeddings, h_0)  # [batch_size, seq_len, hidden_dim]

        logits = self.output_proj(output)  # [batch_size, seq_len, vocab_size]
        return logits


class GRUVAE(nn.Module):
    """GRU encoder-decoder VAE."""

    def __init__(self, config: GRUVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = GRUEncoder(config)
        self.decoder = GRUDecoder(config)

    def _sample_gaussian(
        self, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        """Sample from Gaussian distribution.

        Args:
            mean: Mean [batch_size, latent_dim]
            log_std: Log standard deviation [batch_size, latent_dim]

        Returns:
            z: Sampled latent vector [batch_size, latent_dim]
        """
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        z = mean + eps * std
        return z

    def forward(
        self, input_ids: torch.Tensor, input: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            input: Optional input sequence for decoder teacher forcing [batch_size, seq_len]
                   If None, uses input_ids

        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed logits [batch_size, seq_len, vocab_size]
                - mean: Latent mean [batch_size, latent_dim]
                - log_std: Latent log standard deviation [batch_size, latent_dim]
        """
        if input is None:
            input = input_ids

        # Encode to latent parameters
        mean, log_std = self.encoder(input_ids)

        # Sample from latent
        z = self._sample_gaussian(mean, log_std)

        # Decode
        reconstruction = self.decoder(z, input=input)

        return {
            "reconstruction": reconstruction,
            "mean": mean,
            "log_std": log_std,
        }
