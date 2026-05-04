from typing import Any

from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerCollateConfig(BaseModel):
    """Configuration for tokenizer collate function."""

    tokenizer_path: str = Field(..., description="HF tokenizer path or local path")
    sequence_column: str = Field(..., description="Column containing sequences")
    tokenizer_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"padding": "longest", "return_tensors": "pt"},
        description="Kwargs passed to tokenizer.__call__() (padding, max_length, truncation, etc.)",
    )
    preserve_columns: list[str] = Field(
        default_factory=list,
        description="Additional columns to preserve from original batch (as lists)",
    )
    add_shifted_labels: bool = Field(
        True,
        description="If True, add labels=input_ids[:, 1:] for causal VAE / Trainer loss.",
    )


class TokenizerCollate:
    """Collate function that tokenizes sequences in batches.

    Designed to be used as a collate_fn in PyTorch DataLoaders.
    Tokenizes sequences from a specified column and returns a batch
    with input_ids, attention_mask, and optionally preserved columns.
    """

    def __init__(self, config: TokenizerCollateConfig):
        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            config.tokenizer_path
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate and tokenize a batch of items.

        Args:
            batch: List of dictionaries from the dataset

        Returns:
            Dictionary with tokenized tensors (input_ids, attention_mask)
            and any preserved columns as lists.
        """
        # Aggregate sequences from the batch
        sequences = [item[self.config.sequence_column] for item in batch]

        # Aggregate preserved columns
        output_batch: dict[str, Any] = {}
        for col in self.config.preserve_columns:
            output_batch[col] = [item[col] for item in batch]

        tokenized = self.tokenizer(sequences, **self.config.tokenizer_kwargs)
        output_batch.update(tokenized)

        if self.config.add_shifted_labels and "input_ids" in output_batch:
            ids = output_batch["input_ids"]
            output_batch["labels"] = ids[:, 1:].clone()

        return output_batch
