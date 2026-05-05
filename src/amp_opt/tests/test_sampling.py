from __future__ import annotations

import jax
import numpy as np

from amp_opt.sampling import build_random_mutation_sampling_fn


def test_random_mutation_sampling_shape_and_bounds() -> None:
    reference = np.asarray(
        [
            [0, 1, 2, 20, 20],
            [3, 4, 5, 6, 20],
        ],
        dtype=np.int32,
    )
    sampling_fn = build_random_mutation_sampling_fn(
        reference_population=reference,
        vocab_size=21,
        mutation_count=2,
    )
    candidate = np.asarray(sampling_fn(jax.random.PRNGKey(0)))
    assert candidate.shape == (5,)
    assert candidate.min() >= 0
    assert candidate.max() < 21
