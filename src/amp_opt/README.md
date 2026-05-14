# lamp-amp-opt

Discrete AMP optimization driven by **`PredictorAPEX`** (`lamp-apex`) and **`poli-core`** (`AbstractBlackBox`, `BlackBoxInformation`). Peptide search uses a **POLi-style** scalar objective (better activity ⇒ **higher** fitness via **negated MIC**), with mutants proposed from **elites / history** by `StepByStepSolver` subclasses.

The **`poli_baselines`** PyPI/git package is **not** depended on here: it pulls large extras and targets a moving `poli` git API (e.g. seeding helpers not in current `poli-core` wheels). Instead, **`amp_opt.step_by_step_solver`** implements the same **`StepByStepSolver`** contract so `ProteinRandomMutation` stays a drop-in baseline.

**NumPy:** workspace members used by this stack (`lamp-apex`, `pep-eval`, `lamp-amp-opt`) constrain **`numpy>=1.26,<2`** so `poli-core` can coexist with APEX.

## Configuration

Run settings are **YAML + Pydantic** only (no env-based config for this CLI). Peptide alphabet and max length are **fixed in code** (`amp_opt.sequence_constants`).

- **`seed_sequences`**: CSV path and columns (`id`, `sequence` by default); one run per row; fixed `L` per row (substitution-only mutations). Packaged configs use **`data/pep_compass_seeds.csv`** (path is relative to the working directory when you invoke the CLI).
- **`black_box`**: **`apex`**, **`columns`** (`all` / `panel` / `explicit`; panel names must match APEX `pathogen_list` exactly), **`aggregation`** (`mean` / `max` / `min` over MIC columns).
- **`optimization`**: non-empty **`random_seeds`**, at least one of **`max_total_evaluations`** or **`max_iterations`**, **`solver.factory_import_path`** (`module:callable` — the callable must have signature `(black_box, x0, y0, kwargs_dict) -> StepByStepSolver`), **`solver.kwargs`** passed through to that factory.

Example packaged configs live under `amp_opt/configs/` by method, e.g. `amp_opt/configs/random_mutation/overall_log2.yaml`, `amp_opt/configs/mutang_hydramp/overall_log2.yaml`.

## CLI

```bash
uv run --package lamp-amp-opt lamp-amp-opt --config path/to/config.yaml
```

**Hugging Face HydrAMP + mutang:** install optional Hub dependencies (this package declares them as extra `hub`):

```bash
uv sync --package lamp-amp-opt --extra hub
uv run --package lamp-amp-opt lamp-amp-opt --config path/to/mutang_hub_config.yaml
```

Private checkpoints: set **`HF_TOKEN`** (see workspace `.env-default`). You may also pass **`hf_token`** in **`solver.kwargs`** for the mutang HydrAMP factory.

## Default solver (random mutation)

YAML uses a factory, not a class path directly, for example:

`solver.factory_import_path: amp_opt.solver_factories:protein_random_mutation`

Factory **`kwargs`** match **`ProteinRandomMutation`**: `n_mutations`, `top_k`, `greedy`, `batch_size`, optional **`mutation_token_probs`** (length 20, **`STANDARD_AA_ORDER`**).

## Mutang solver (Jacobian-guided uniform mutations)

`ProteinMutangUniformMutation` picks a parent like **`ProteinRandomMutation`** (`top_k`, `greedy`, `batch_size`), runs **`encode(input_ids)`** and **`jacobian_batch_fn(z)`**, builds proposals with **`pep_compass_jr.substitutions_batch_from_jacobian`**, and applies **`n_mutations`** sequential mutation steps. Each step draws **one** pair **uniformly** from all Jacobian-derived `(position, new_vocab_id)` options that are **not** identical to the current model ids, then recomputes the Jacobian on the updated ids. If a step has no such pairs, it raises **`ValueError`** with the **mutation step** index (no fallback).

**Constructor arguments** (after `black_box`, `x0`, `y0`): required **`encode`**, **`jacobian_batch_fn`**, **`sequence_length`**, **`vocab_size`**. Here **`sequence_length`** and **`vocab_size`** refer to the **model** layout (Jacobian ambient is **`sequence_length * vocab_size`**), not necessarily the peptide length passed to the black box.

Optional: **`n_mutations`** (default **1**), **`alphabet`**, **`tokenizer`** (for **`x0.ndim == 1`**, same idea as **`ProteinRandomMutation`**), **`tokenize_row`** / **`decode_row`** (both **required together or both omitted**): map a black-box row **`(L_bb,)`** to **`input_ids`** **`[1, S_model]`** and model id vector **`(S_model,)`** back to **`(L_bb,)`** for scoring. When both are **`None`**, each parent row must have length **`sequence_length`** and use the default char ↔ id mapping via **`alphabet[vocab_id]`** for **`0 .. vocab_size - 1`**.

Optional Jacobian tuning (defaults match **`substitutions_batch_from_jacobian`**): **`direction_significance_threshold`**, **`min_number_of_directions`**, **`token_threshold`**.

**YAML / CLI:** bind models and Jacobians with **`solver.factory_import_path`**, e.g. **`amp_opt.mutang_hydramp_init:build_mutang_hydramp_solver`** (see **`configs/mutang_hydramp/overall_log2.yaml`**). That factory loads HydrAMP from the Hub with **`transformers`** and wires **`pep_compass_jr.utils.softmax_probs_jacobian_fn`**. Seed peptide length must equal **`model.config.sequence_length`** (HydrAMP is fixed-width).

## Output CSV

`seed_id`, `seed_sequence`, `best_sequence`, `best_fitness`, `black_box_signature`, `status`, `error_message`, plus provenance columns including **`solver_factory_path`**.

## Workspace

Depends on **`lamp-apex`**, **`lamp-pep-compass-jr`** (and **PyTorch**), **`pep-eval`**, **`poli-core`**. Optional **`hub`** extra adds **`transformers`** and **`huggingface_hub`** for Hub-based mutang runs.
