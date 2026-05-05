"""Tests for OptimizationRunner with mock objective and method."""

from __future__ import annotations

from typing import Any

import numpy as np

from amp_opt.core.archive import OptimizationArchive
from amp_opt.core.method import MethodState
from amp_opt.core.runner import OptimizationRunner
from amp_opt.core.types import Candidate


class _MockState(MethodState):
    def __init__(self) -> None:
        self.call_count = 0


class _MockObjective:
    @property
    def name(self) -> str:
        return "mock_objective"

    def score_sequences(self, sequences: list[str]) -> np.ndarray:
        return np.array([len(s) for s in sequences], dtype=float)


class _MockMethod:
    @property
    def name(self) -> str:
        return "mock_method"

    def get_config(self) -> dict[str, Any]:
        return {"type": "mock"}

    def initialize(
        self, starting_sequences: list[str], archive: OptimizationArchive
    ) -> _MockState:
        return _MockState()

    def propose(self, state: MethodState, archive: OptimizationArchive) -> list[str]:
        assert isinstance(state, _MockState)
        return ["A", "AA", "AAA"]

    def update(
        self,
        state: MethodState,
        candidates: list[Candidate],
        archive: OptimizationArchive,
    ) -> _MockState:
        assert isinstance(state, _MockState)
        state.call_count += 1
        return state


def test_runner_executes_generations() -> None:
    runner = OptimizationRunner(objective=_MockObjective(), method=_MockMethod())
    result = runner.run(starting_sequences=["HELLO", "HI"], generations=3, top_k=5)
    assert result.best.sequence == "A"
    assert result.best.score == 1.0
    assert len(result.history) == 3
    assert result.history[0].generation == 1
    assert result.history[2].generation == 3


def test_runner_archive_deduplicates() -> None:
    runner = OptimizationRunner(objective=_MockObjective(), method=_MockMethod())
    result = runner.run(starting_sequences=["AAA"], generations=2, top_k=10)
    seqs = [c.sequence for c in result.top_k]
    assert len(seqs) == len(set(seqs))
