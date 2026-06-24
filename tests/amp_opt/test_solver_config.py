"""Pydantic config: ``SolverConfig`` and ``factory_import_path`` validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from amp_opt.config import SolverConfig, load_run_config


def test_solver_config_factory_path_requires_single_colon() -> None:
    with pytest.raises(ValueError, match="exactly one ':'"):
        SolverConfig(factory_import_path="nocolon", kwargs={})
    with pytest.raises(ValueError, match="exactly one ':'"):
        SolverConfig(factory_import_path="too:many:colons", kwargs={})
    with pytest.raises(ValueError, match="module:callable"):
        SolverConfig(factory_import_path=":callableonly", kwargs={})


def test_load_run_config_accepts_factory_solver(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            seed_sequences:
              path: data/pep_compass_seeds.csv
              id_column: id
              sequence_column: sequence
            output:
              directory: out
              mode: write
            optimization:
              random_seeds: [0]
              max_total_evaluations: 10
              solver:
                factory_import_path: amp_opt.solver_factories:protein_random_mutation
                kwargs:
                  n_mutations: 1
                  top_k: 1
                  greedy: true
                  batch_size: 1
            """
        ).strip(),
        encoding="utf-8",
    )
    rc = load_run_config(cfg_path)
    assert rc.optimization.solver.factory_import_path == (
        "amp_opt.solver_factories:protein_random_mutation"
    )
