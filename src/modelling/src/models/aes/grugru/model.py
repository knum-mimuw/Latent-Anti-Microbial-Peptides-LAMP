import torch
import torch.nn as nn
from typing import Dict
from einops import rearrange
from transformers import PreTrainedModel

from .config import GRUVAEConfig


class GRUEncoder(nn.Module):
    """GRU-based encoder for VAE."""

    def __init__(self, config: GRUVAEConfig):
        """Initialize encoder with embedding and GRU layers."""
        super().__init__()
        self.config = config

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent parameters."""
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        output, hidden = self.gru(embeddings)

        if self.config.encoder_bidirectional:
            forward_hidden = hidden[-2]  # [batch_size, hidden_dim]
            backward_hidden = hidden[-1]  # [batch_size, hidden_dim]
            last_hidden = torch.cat(
                [forward_hidden, backward_hidden], dim=-1
            )  # [batch_size, hidden_dim*2]
        else:
            last_hidden = hidden[-1]  # [batch_size, hidden_dim]

        mean = self.mean_linear(last_hidden)
        log_std = self.log_std_linear(last_hidden)

        return mean, log_std


class GRUDecoder(nn.Module):
    """GRU-based decoder for VAE."""

    def __init__(self, config: GRUVAEConfig):
        """Initialize decoder with embedding, GRU, and projection layers."""
        super().__init__()
        self.config = config

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

    def forward(self, z: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to sequence."""

        h_0_flat = self.latent_proj(z)  # [batch_size, hidden_dim * num_layers]
        h_0 = rearrange(
            h_0_flat,
            "b (l h) -> l b h",
            l=self.config.decoder_num_layers,
            h=self.config.decoder_hidden_size,
        ).contiguous()  # [num_layers, batch_size, hidden_dim]

        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        output, _ = self.gru(embeddings, h_0)  # [batch_size, seq_len, hidden_dim]

        logits = self.output_proj(output)  # [batch_size, seq_len, vocab_size]
        return logits


def _sample_gaussian(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Sample from Gaussian distribution."""
    eps = torch.randn_like(mean)
    z = mean + eps * std
    return z


class GRUVAE(PreTrainedModel):
    """GRU encoder-decoder VAE with HuggingFace compatibility.

    This model can be saved/loaded via HuggingFace Hub:
        model.save_pretrained("./my_model")
        model.push_to_hub("username/model-name")
        model = GRUVAE.from_pretrained("username/model-name")

    It also works seamlessly with PyTorch Lightning since PreTrainedModel
    inherits from nn.Module.
    """

    config_class = GRUVAEConfig
    base_model_prefix = "gruvae"

    def __init__(self, config: GRUVAEConfig):
        """Initialize VAE with encoder and decoder."""
        super().__init__(config)
        self.encoder = GRUEncoder(config)
        self.decoder = GRUDecoder(config)

        # Initialize weights using HuggingFace's mechanism
        self.post_init()

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            input_ids: Tokenized input sequences [batch_size, seq_len]
            **kwargs: Additional batch keys (e.g., attention_mask) - ignored

        Returns:
            Dictionary with logits, mean, and log_std tensors.
        """
        mean, log_std = self.encoder(input_ids)
        z = _sample_gaussian(mean, torch.exp(log_std))
        logits = self.decoder(z, input_ids)

        logits = rearrange(logits, "b s v -> b v s")

        return {
            "logits": logits,
            "mean": mean,
            "log_std": log_std,
        }
