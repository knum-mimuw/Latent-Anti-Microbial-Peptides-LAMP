"""Configuration for the random-mutation search method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ParentPool = Literal["starting", "archive_top_k", "archive_all", "best"]


@dataclass(frozen=True)
class RandomSearchConfig:
    """Parameters for evosax-backed random mutation search."""

    population_size: int = 64
    mutation_count: int = 1
    seed: int = 0
    parent_pool: ParentPool = "archive_top_k"
    parent_pool_top_k: int = 50
