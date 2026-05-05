"""Mutable state for random-mutation search."""

from __future__ import annotations

import jax

from ...core.method import MethodState
from .config import RandomSearchConfig


class RandomSearchState(MethodState):
    """Holds RNG key and generation counter for random search."""

    def __init__(self, *, rng_key: jax.Array, generation: int, config: RandomSearchConfig) -> None:
        self.rng_key = rng_key
        self.generation = generation
        self.config = config
