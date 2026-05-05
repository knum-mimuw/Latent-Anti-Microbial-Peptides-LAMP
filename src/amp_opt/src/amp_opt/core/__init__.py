"""Core optimization abstractions: types, objective, archive, method protocol, runner."""

from .archive import OptimizationArchive
from .method import MethodState, OptimizerMethod
from .objective import ApexPanelObjective, ScoreTransform, SequenceObjective
from .runner import OptimizationRunner
from .types import Candidate, GenerationRecord, GenerationStats, OptimizationResult

__all__ = [
    "ApexPanelObjective",
    "Candidate",
    "GenerationRecord",
    "GenerationStats",
    "MethodState",
    "OptimizationArchive",
    "OptimizationResult",
    "OptimizationRunner",
    "OptimizerMethod",
    "ScoreTransform",
    "SequenceObjective",
]
