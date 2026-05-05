"""Sampling functions used by evosax RandomSearch."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

SamplingFn = Callable[[jax.Array], jax.Array]


def build_random_mutation_sampling_fn(
    *,
    reference_population: np.ndarray,
    vocab_size: int,
    mutation_count: int = 1,
) -> SamplingFn:
    """Create a random-mutation style sampler.

    The sampler picks a random parent from `reference_population`, mutates
    `mutation_count` positions with uniformly sampled symbols, and returns one
    candidate genome.
    """
    if mutation_count <= 0:
        raise ValueError("mutation_count must be positive.")
    ref = jnp.asarray(reference_population, dtype=jnp.int32)
    if ref.ndim != 2:
        raise ValueError("reference_population must be rank-2 [num_refs, genome_len].")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be > 1.")
    genome_len = ref.shape[1]

    def sampling_fn(key: jax.Array) -> jax.Array:
        parent_key, pos_key, aa_key = jax.random.split(key, 3)
        parent_idx = jax.random.randint(parent_key, shape=(), minval=0, maxval=ref.shape[0])
        proposal = ref[parent_idx]
        positions = jax.random.randint(
            pos_key,
            shape=(mutation_count,),
            minval=0,
            maxval=genome_len,
        )
        values = jax.random.randint(
            aa_key,
            shape=(mutation_count,),
            minval=0,
            maxval=vocab_size,
        )
        return proposal.at[positions].set(values)

    return sampling_fn
