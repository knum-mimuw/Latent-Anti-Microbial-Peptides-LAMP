"""Sequence encoding helpers for JAX search genomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN: str = "_"


@dataclass(frozen=True)
class AlphabetCodec:
    """Encode/decode peptide sequences to fixed-length integer genomes."""

    max_length: int
    alphabet: str = AMINO_ACIDS
    pad_token: str = PAD_TOKEN

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet contains duplicate symbols.")
        if self.pad_token in self.alphabet:
            raise ValueError("pad_token must not be part of alphabet.")

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.alphabet) + (self.pad_token,)

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def pad_idx(self) -> int:
        return self.vocab_size - 1

    def encode(self, sequence: str) -> np.ndarray:
        """Encode sequence to fixed-length integer vector."""
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError("sequence must be a non-empty string.")
        tokens = list(sequence.strip())
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        token_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        arr = np.full(self.max_length, self.pad_idx, dtype=np.int32)
        for idx, token in enumerate(tokens):
            try:
                arr[idx] = token_to_idx[token]
            except KeyError as exc:
                raise ValueError(f"Unknown amino-acid token: '{token}'.") from exc
        return arr

    def decode(self, genome: np.ndarray) -> str:
        """Decode integer vector back to peptide sequence."""
        arr = np.asarray(genome, dtype=np.int32)
        if arr.ndim != 1 or arr.shape[0] != self.max_length:
            raise ValueError(f"genome must have shape ({self.max_length},).")
        symbols = self.symbols
        chars: list[str] = []
        for idx in arr.tolist():
            if idx < 0 or idx >= self.vocab_size:
                raise ValueError(f"Genome index '{idx}' out of bounds.")
            token = symbols[idx]
            if token == self.pad_token:
                break
            chars.append(token)
        if not chars:
            # Keep empty sequences from reaching APEX internals.
            raise ValueError("Decoded sequence is empty.")
        return "".join(chars)
