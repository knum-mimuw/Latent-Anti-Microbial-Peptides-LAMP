from transformers import PretrainedConfig


class GRUVAEConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for GRU-based VAE.

    Uses flat parameters for maximum compatibility with HuggingFace Hub.
    """

    model_type = "gruvae"

    # Enable trust_remote_code loading via AutoModel
    auto_map = {
        "AutoConfig": "config.GRUVAEConfig",
        "AutoModel": "model.GRUVAE",
    }

    def __init__(
        self,
        # Embedding parameters
        vocab_size: int = 29,
        embedding_dim: int = 100,
        padding_idx: int = 0,
        # Latent space
        latent_dim: int = 64,
        # Encoder parameters
        encoder_hidden_size: int = 128,
        encoder_num_layers: int = 2,
        encoder_bidirectional: bool = True,
        encoder_dropout: float = 0.1,
        # Decoder parameters
        decoder_hidden_size: int = 128,
        decoder_num_layers: int = 2,
        decoder_dropout: float = 0.1,
        # Training loss (used by Hugging Face Trainer; inference leaves ``labels`` unset)
        kl_beta: float = 1e-3,
        ignore_index: int = 0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        """Initialize GRU VAE configuration.

        Args:
            vocab_size: Number of tokens in vocabulary (amino acids + special tokens).
            embedding_dim: Dimension of token embeddings (must be even for the decoder's
                sinusoidal positional inputs used in training).
            padding_idx: Index used for padding tokens.
            latent_dim: Dimension of the latent space.
            encoder_hidden_size: Hidden size of encoder GRU.
            encoder_num_layers: Number of layers in encoder GRU.
            encoder_bidirectional: Whether encoder GRU is bidirectional.
            encoder_dropout: Dropout rate for encoder GRU.
            decoder_hidden_size: Hidden size of decoder GRU.
            decoder_num_layers: Number of layers in decoder GRU.
            decoder_dropout: Dropout rate for decoder GRU.
            kl_beta: Weight on KL(q(z|x) || N(0,I)) in the training loss.
            ignore_index: Passed to cross-entropy for padded / masked positions.
            label_smoothing: Cross-entropy label smoothing (0 disables).
            **kwargs: Additional arguments passed to PretrainedConfig.
        """
        super().__init__(**kwargs)

        # Embedding
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Latent
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.encoder_dropout = encoder_dropout

        # Decoder
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.decoder_dropout = decoder_dropout

        self.kl_beta = kl_beta
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
