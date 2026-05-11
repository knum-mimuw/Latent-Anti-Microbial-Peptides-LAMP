# lamp-amp-opt

Discrete AMP optimization driven by **`PredictorAPEX`** (`lamp-apex`) and **`poli-core`** (`AbstractBlackBox`, `BlackBoxInformation`). Peptide search uses a **POLi-style** scalar objective (better activity ⇒ **higher** fitness via **negated MIC**), with mutants proposed from **elites / history** by `StepByStepSolver` subclasses.

The **`poli_baselines`** PyPI/git package is **not** depended on here: it pulls large extras and targets a moving `poli` git API (e.g. seeding helpers not in current `poli-core` wheels). Instead, **`amp_opt.step_by_step_solver`** implements the same **`StepByStepSolver`** contract so `ProteinRandomMutation` stays a drop-in baseline.

**NumPy:** workspace members used by this stack (`lamp-apex`, `pep-eval`, `lamp-amp-opt`) constrain **`numpy>=1.26,<2`** so `poli-core` can coexist with APEX.

## Configuration

Run settings are **YAML + Pydantic** only (no env-based config for this CLI). Peptide alphabet and max length are **fixed in code** (`amp_opt.sequence_constants`).

- **`seeds`**: CSV path and columns (`id`, `sequence` by default); one run per row; fixed `L` per row (substitution-only mutations).
- **`black_box`**: **`apex`**, **`columns`** (`all` / `panel` / `explicit`; panel names must match APEX `pathogen_list` exactly), **`aggregation`** (`mean` / `max` / `min` over MIC columns).
- **`optimization`**: **`random_seed`**, at least one of **`max_total_evaluations`** or **`max_iterations`**, **`solver.import_path`** (`module:Class`), **`solver.kwargs`**.

Example: `amp_opt/configs/default.yaml`.

## CLI

```bash
uv run --package lamp-amp-opt lamp-amp-opt --config path/to/config.yaml
```

## Default solver

`amp_opt.random_mutation_solver:ProteinRandomMutation` — kwargs: `n_mutations`, `top_k`, `greedy`, `batch_size`, optional **`mutation_token_probs`** (length 20, **`STANDARD_AA_ORDER`**).

## Output CSV

`seed_id`, `seed_sequence`, `best_sequence`, `best_fitness`, `black_box_signature`, `status`, `error_message`, plus provenance columns.

## Workspace

Depends on **`lamp-apex`**, **`pep-eval`**, **`poli-core`**.
