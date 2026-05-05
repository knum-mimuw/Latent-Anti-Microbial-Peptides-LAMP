"""Algorithm-agnostic optimization runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from .archive import OptimizationArchive
from .method import OptimizerMethod
from .objective import SequenceObjective
from .types import Candidate, GenerationRecord, GenerationStats, OptimizationResult


class OptimizationRunner:
    """Runs the optimization loop: propose -> evaluate -> archive -> update."""

    def __init__(
        self,
        *,
        objective: SequenceObjective,
        method: OptimizerMethod,
        archive: OptimizationArchive | None = None,
    ) -> None:
        self.objective = objective
        self.method = method
        self.archive = archive or OptimizationArchive()

    def run(
        self,
        *,
        starting_sequences: list[str],
        generations: int,
        top_k: int = 100,
    ) -> OptimizationResult:
        """Execute the full optimization loop."""
        if not starting_sequences:
            raise ValueError("starting_sequences must not be empty.")
        if generations <= 0:
            raise ValueError("generations must be positive.")

        scores = self.objective.score_sequences(starting_sequences)
        initial_candidates = [
            Candidate(sequence=seq, score=float(s), generation=0)
            for seq, s in zip(starting_sequences, scores, strict=False)
        ]
        self.archive.add(initial_candidates)

        state = self.method.initialize(starting_sequences, self.archive)
        history: list[GenerationRecord] = []

        for gen in range(1, generations + 1):
            proposed = self.method.propose(state, self.archive)
            if not proposed:
                break
            gen_scores = self.objective.score_sequences(proposed)
            candidates = [
                Candidate(sequence=seq, score=float(s), generation=gen)
                for seq, s in zip(proposed, gen_scores, strict=False)
            ]
            self.archive.add(candidates)
            state = self.method.update(state, candidates, self.archive)

            best = self.archive.best
            assert best is not None
            stats = GenerationStats(
                best_score=best.score,
                mean_score=float(np.mean(gen_scores)),
                std_score=float(np.std(gen_scores)),
                population_size=len(proposed),
            )
            record = GenerationRecord(
                generation=gen,
                candidates=tuple(candidates),
                best=best,
                stats=stats,
            )
            history.append(record)

        best = self.archive.best
        assert best is not None
        config = self._build_config()
        return OptimizationResult(
            best=best,
            top_k=tuple(self.archive.top_k(top_k)),
            history=tuple(history),
            config=config,
        )

    def _build_config(self) -> dict[str, Any]:
        return {
            "objective": self.objective.name,
            "method": self.method.name,
            "method_config": self.method.get_config(),
        }
