"""Shared data types for the optimization stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """A single evaluated peptide candidate."""

    sequence: str
    score: float
    generation: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationStats:
    """Summary statistics for one generation."""

    best_score: float
    mean_score: float
    std_score: float
    population_size: int


@dataclass(frozen=True)
class GenerationRecord:
    """Full record for one generation of optimization."""

    generation: int
    candidates: tuple[Candidate, ...]
    best: Candidate
    stats: GenerationStats


@dataclass(frozen=True)
class OptimizationResult:
    """Final output of an optimization run."""

    best: Candidate
    top_k: tuple[Candidate, ...]
    history: tuple[GenerationRecord, ...]
    config: dict[str, Any]
