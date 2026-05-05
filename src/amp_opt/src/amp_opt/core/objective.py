"""Sequence objective protocol and APEX panel implementation."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

from ..oracle import ApexPanelOracle, ObjectivePanel

ScoreTransform = Literal["raw_mic", "log10_mic"]


@runtime_checkable
class SequenceObjective(Protocol):
    """Protocol for scoring peptide sequences. Lower score is better."""

    @property
    def name(self) -> str: ...

    def score_sequences(self, sequences: list[str]) -> np.ndarray: ...


class ApexPanelObjective:
    """APEX panel oracle wrapped as a SequenceObjective."""

    def __init__(
        self,
        *,
        oracle: ApexPanelOracle,
        score_transform: ScoreTransform = "raw_mic",
    ) -> None:
        self._oracle = oracle
        self._score_transform = score_transform

    @property
    def name(self) -> str:
        return f"apex_panel_{self._oracle.panel}"

    @property
    def oracle(self) -> ApexPanelOracle:
        return self._oracle

    @property
    def panel(self) -> ObjectivePanel:
        return self._oracle.panel

    def score_sequences(self, sequences: list[str]) -> np.ndarray:
        """Score sequences. Lower is better."""
        mic = self._oracle.score_sequences(sequences)
        if self._score_transform == "log10_mic":
            return np.log10(mic)
        return mic
