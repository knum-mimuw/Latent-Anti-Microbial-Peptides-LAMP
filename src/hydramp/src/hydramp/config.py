"""Hugging Face config for HydrAMP."""

from transformers import PretrainedConfig


class HydrAMPConfig(PretrainedConfig):
    """Configuration for HydrAMP encoder/decoder model."""

    model_type = "hydramp"
    auto_map = {
        "AutoConfig": "config.HydrAMPConfig",
        "AutoModel": "model.HydrAMPModel",
    }

    def __init__(
        self,
        vocab_size: int = 21,
        sequence_length: int = 25,
        latent_dim: int = 64,
        condition_dim: int = 2,
        embedding_dim: int = 100,
        encoder_hidden_size: int = 128,
        decoder_hidden_size: int = 100,
        default_condition: list[float] | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.embedding_dim = embedding_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.default_condition = default_condition or [1.0, 1.0]
        self.temperature = temperature
        self.auto_map = {
            "AutoConfig": "config.HydrAMPConfig",
            "AutoModel": "model.HydrAMPModel",
        }

