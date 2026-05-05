# lamp-amp-opt

Standalone AMP optimization package using APEX full-pathogen predictions as the
oracle and evosax RandomSearch as the optimizer.

## Runtime notes

- APEX scoring uses PyTorch (`APEX_DEVICE`, default `cpu`).
- evosax uses JAX. For mixed Torch/JAX workloads, `JAX_PLATFORM_NAME=cpu` avoids
  GPU contention unless you intentionally want JAX on GPU.

## CLI

```bash
uv run lamp-amp-opt run --panel gram_negative --sequence "KKLLKLLKLLK,KRWWKWWRR"
```

## Extensibility

`run_random_search(...)` accepts a custom `sampling_fn(key)` so mutation
proposal logic can be upgraded independently from the search loop.
