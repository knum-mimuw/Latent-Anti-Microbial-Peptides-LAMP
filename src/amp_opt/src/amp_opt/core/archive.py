"""Optimization archive: stores evaluated candidates with deduplication."""

from __future__ import annotations

from .types import Candidate


class OptimizationArchive:
    """Stores and deduplicates evaluated candidates across generations."""

    def __init__(self) -> None:
        self._by_sequence: dict[str, Candidate] = {}

    def add(self, candidates: list[Candidate] | tuple[Candidate, ...]) -> None:
        """Add candidates, keeping only the best score per unique sequence."""
        for candidate in candidates:
            existing = self._by_sequence.get(candidate.sequence)
            if existing is None or candidate.score < existing.score:
                self._by_sequence[candidate.sequence] = candidate

    @property
    def best(self) -> Candidate | None:
        """Return the candidate with the lowest score, or None if empty."""
        if not self._by_sequence:
            return None
        return min(self._by_sequence.values(), key=lambda c: c.score)

    def top_k(self, k: int) -> list[Candidate]:
        """Return the k candidates with lowest scores."""
        return sorted(self._by_sequence.values(), key=lambda c: c.score)[:k]

    def all_sequences(self) -> list[str]:
        """Return all unique sequences in the archive."""
        return list(self._by_sequence.keys())

    def all_candidates(self) -> list[Candidate]:
        """Return all candidates sorted by score."""
        return sorted(self._by_sequence.values(), key=lambda c: c.score)

    @property
    def size(self) -> int:
        return len(self._by_sequence)
