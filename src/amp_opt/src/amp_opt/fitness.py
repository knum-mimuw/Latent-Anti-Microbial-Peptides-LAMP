"""Black-box fitness adapter from genome vectors to APEX panel scores."""

from __future__ import annotations

import numpy as np

from .encoding import AlphabetCodec
from .oracle import ApexPanelOracle


class GenomeFitness:
    """Evaluate evosax genomes by decoding and scoring with APEX."""

    def __init__(
        self,
        *,
        codec: AlphabetCodec,
        oracle: ApexPanelOracle,
        use_log_mic: bool = False,
    ) -> None:
        self.codec = codec
        self.oracle = oracle
        self.use_log_mic = use_log_mic

    def evaluate_one(self, genome: np.ndarray) -> float:
        sequence = self.codec.decode(np.asarray(genome))
        mic = self.oracle.score_sequence(sequence)
        if self.use_log_mic:
            return float(np.log10(mic))
        return float(mic)

    def evaluate_batch(self, genomes: np.ndarray) -> np.ndarray:
        arr = np.asarray(genomes)
        if arr.ndim != 2:
            raise ValueError("genomes must have shape [population, genome_len].")
        sequences = [self.codec.decode(row) for row in arr]
        mic = self.oracle.score_sequences(sequences)
        if self.use_log_mic:
            mic = np.log10(mic)
        return np.asarray(mic, dtype=np.float64)
