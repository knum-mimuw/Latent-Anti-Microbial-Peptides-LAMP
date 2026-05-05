"""evosax RandomSearch runner for AMP panel optimization."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.distribution_based.random_search import RandomSearch

from .fitness import GenomeFitness
from .sampling import SamplingFn, build_random_mutation_sampling_fn


@dataclass(frozen=True)
class RandomSearchConfig:
    """Configuration for evosax RandomSearch execution."""

    seed: int = 0
    population_size: int = 64
    generations: int = 50
    mutation_count: int = 1


@dataclass(frozen=True)
class RandomSearchResult:
    """Optimization output with best found sequence and score."""

    best_sequence: str
    best_fitness: float
    best_mic: float
    unresolved_strains: tuple[str, ...]
    resolved_apex_columns: tuple[str, ...]
    best_genome: tuple[int, ...]


def run_random_search(
    *,
    fitness: GenomeFitness,
    initial_sequences: list[str],
    search_config: RandomSearchConfig,
    sampling_fn: SamplingFn | None = None,
) -> RandomSearchResult:
    """Optimize peptides with evosax RandomSearch."""
    if not initial_sequences:
        raise ValueError("initial_sequences must not be empty.")
    codec = fitness.codec
    reference_population = np.stack([codec.encode(seq) for seq in initial_sequences], axis=0)

    effective_sampling_fn = sampling_fn or build_random_mutation_sampling_fn(
        reference_population=reference_population,
        vocab_size=codec.vocab_size,
        mutation_count=search_config.mutation_count,
    )

    algo = RandomSearch(
        population_size=search_config.population_size,
        solution=jnp.asarray(reference_population[0], dtype=jnp.int32),
        sampling_fn=effective_sampling_fn,
    )
    params = algo.default_params

    key = jax.random.PRNGKey(search_config.seed)
    key, init_key = jax.random.split(key)
    init_mean = jnp.asarray(reference_population[0], dtype=jnp.int32)
    state = algo.init(init_key, init_mean, params)

    best_genome = reference_population[0]
    best_fitness = float(fitness.evaluate_one(best_genome))

    for _ in range(search_config.generations):
        key, ask_key, tell_key = jax.random.split(key, 3)
        population, state = algo.ask(ask_key, state, params)
        population_np = np.asarray(population, dtype=np.int32)
        fitness_values = fitness.evaluate_batch(population_np)
        state, _metrics = algo.tell(
            tell_key,
            population,
            jnp.asarray(fitness_values, dtype=jnp.float32),
            state,
            params,
        )
        step_best_idx = int(np.argmin(fitness_values))
        step_best_fitness = float(fitness_values[step_best_idx])
        if step_best_fitness < best_fitness:
            best_fitness = step_best_fitness
            best_genome = population_np[step_best_idx]

    best_sequence = codec.decode(best_genome)
    best_mic = float(fitness.oracle.score_sequence(best_sequence))
    resolution = fitness.oracle.resolution
    return RandomSearchResult(
        best_sequence=best_sequence,
        best_fitness=best_fitness,
        best_mic=best_mic,
        unresolved_strains=resolution.unresolved_strains,
        resolved_apex_columns=resolution.resolved_apex_columns,
        best_genome=tuple(int(v) for v in best_genome.tolist()),
    )
