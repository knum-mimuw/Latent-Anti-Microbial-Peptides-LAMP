"""Optimizer method protocol: the interface every optimization algorithm implements."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .archive import OptimizationArchive
from .types import Candidate


class MethodState:
    """Base for method-specific mutable state. Subclass per method."""


@runtime_checkable
class OptimizerMethod(Protocol):
    """Protocol for pluggable optimization algorithms."""

    @property
    def name(self) -> str: ...

    def initialize(
        self,
        starting_sequences: list[str],
        archive: OptimizationArchive,
    ) -> MethodState: ...

    def propose(
        self,
        state: MethodState,
        archive: OptimizationArchive,
    ) -> list[str]: ...

    def update(
        self,
        state: MethodState,
        candidates: list[Candidate],
        archive: OptimizationArchive,
    ) -> MethodState: ...

    def get_config(self) -> dict[str, Any]: ...
