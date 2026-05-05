"""JAX sampling function for random-mutation proposals."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

SamplingFn = Callable[[jax.Array], jax.Array]


def build_random_mutation_sampling_fn(
    *,
    parent_genomes: np.ndarray,
    vocab_size: int,
    mutation_count: int = 1,
) -> SamplingFn:
    """Build a JAX-compatible sampler that mutates random parents.

    Picks a random parent from `parent_genomes`, mutates `mutation_count`
    random positions to random symbols, returns one candidate genome.
    """
    if mutation_count <= 0:
        raise ValueError("mutation_count must be positive.")
    ref = jnp.asarray(parent_genomes, dtype=jnp.int32)
    if ref.ndim != 2:
        raise ValueError("parent_genomes must be rank-2 [num_parents, genome_len].")
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
