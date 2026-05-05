"""Random-mutation search method implementing OptimizerMethod."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.distribution_based.random_search import RandomSearch

from ...core.archive import OptimizationArchive
from ...core.method import MethodState
from ...core.types import Candidate
from ...encoding import AlphabetCodec
from .config import RandomSearchConfig
from .sampler import build_random_mutation_sampling_fn
from .state import RandomSearchState


class RandomSearchMethod:
    """evosax RandomSearch with archive-aware parent sampling."""

    def __init__(self, *, config: RandomSearchConfig, codec: AlphabetCodec) -> None:
        self._config = config
        self._codec = codec

    @property
    def name(self) -> str:
        return "random_search"

    def get_config(self) -> dict[str, Any]:
        return {
            "population_size": self._config.population_size,
            "mutation_count": self._config.mutation_count,
            "seed": self._config.seed,
            "parent_pool": self._config.parent_pool,
            "parent_pool_top_k": self._config.parent_pool_top_k,
            "max_length": self._codec.max_length,
        }

    def initialize(
        self,
        starting_sequences: list[str],
        archive: OptimizationArchive,
    ) -> RandomSearchState:
        key = jax.random.PRNGKey(self._config.seed)
        return RandomSearchState(rng_key=key, generation=0, config=self._config)

    def propose(
        self,
        state: MethodState,
        archive: OptimizationArchive,
    ) -> list[str]:
        assert isinstance(state, RandomSearchState)
        parent_genomes = self._get_parent_genomes(archive, state)
        sampling_fn = build_random_mutation_sampling_fn(
            parent_genomes=parent_genomes,
            vocab_size=self._codec.vocab_size,
            mutation_count=state.config.mutation_count,
        )
        algo = RandomSearch(
            population_size=state.config.population_size,
            solution=jnp.asarray(parent_genomes[0], dtype=jnp.int32),
            sampling_fn=sampling_fn,
        )
        params = algo.default_params
        key, init_key, ask_key = jax.random.split(state.rng_key, 3)
        init_state = algo.init(init_key, jnp.asarray(parent_genomes[0], dtype=jnp.int32), params)
        population, _ = algo.ask(ask_key, init_state, params)
        population_np = np.asarray(population, dtype=np.int32)

        state.rng_key = key
        sequences: list[str] = []
        for row in population_np:
            try:
                sequences.append(self._codec.decode(row))
            except ValueError:
                continue
        return sequences

    def update(
        self,
        state: MethodState,
        candidates: list[Candidate],
        archive: OptimizationArchive,
    ) -> RandomSearchState:
        assert isinstance(state, RandomSearchState)
        state.generation += 1
        return state

    def _get_parent_genomes(
        self,
        archive: OptimizationArchive,
        state: RandomSearchState,
    ) -> np.ndarray:
        pool = state.config.parent_pool
        if pool == "best":
            best = archive.best
            if best is not None:
                return np.stack([self._codec.encode(best.sequence)])
        elif pool == "archive_top_k":
            top = archive.top_k(state.config.parent_pool_top_k)
            if top:
                return np.stack([self._codec.encode(c.sequence) for c in top])
        elif pool == "archive_all":
            all_c = archive.all_candidates()
            if all_c:
                return np.stack([self._codec.encode(c.sequence) for c in all_c])

        # Fallback to whatever is in archive (starting sequences were added)
        all_c = archive.all_candidates()
        if all_c:
            return np.stack([self._codec.encode(c.sequence) for c in all_c])
        raise ValueError("Archive is empty; cannot build parent pool.")
