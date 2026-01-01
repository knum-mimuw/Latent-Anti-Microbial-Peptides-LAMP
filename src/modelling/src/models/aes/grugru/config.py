from pydantic import BaseModel, ConfigDict, Field


class GRUConfig(BaseModel):
    batch_first: bool = Field(True, description="Whether to use batch first indexing")
    hidden_size: int = Field(..., description="GRU hidden dimension")
    num_layers: int = Field(..., description="Number of GRU layers")
    bidirectional: bool = Field(..., description="Use bidirectional encoder")
    dropout: float = Field(..., description="Dropout rate")

    model_config = ConfigDict(extra="allow")


class EmbeddingConfig(BaseModel):
    num_embeddings: int = Field(..., description="Number of embeddings")
    embedding_dim: int = Field(..., description="Embedding dimension")
    padding_idx: int = Field(0, description="Padding index")

    model_config = ConfigDict(extra="allow")


class GRUVAEConfig(BaseModel):
    """Configuration for GRU-based VAE."""

    embedding: EmbeddingConfig = Field(..., description="Embedding configuration")
    latent_dim: int = Field(64, description="Latent space dimension")
    encoder: GRUConfig = Field(..., description="Encoder configuration")
    decoder: GRUConfig = Field(..., description="Decoder configuration")
