"""CLI and batch driver: YAML config → APEX black box → configurable StepByStepSolver."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from poli.core.exceptions import BudgetExhaustedException
from tqdm import tqdm

from amp_opt.black_box import (
    ApexBlackBox,
    black_box_signature,
    build_predictor,
    resolve_mic_columns,
)
from amp_opt.config import load_run_config, validate_seed_sequence
from amp_opt.step_by_step_solver import StepByStepSolver, seed_python_numpy_torch


def _write_result_row(writer: csv.DictWriter, fh, meta: dict, **kwargs) -> None:
    writer.writerow({**meta, **kwargs})
    fh.flush()


def import_from_path(qualified: str) -> Any:
    mod_name, sep, attr = qualified.partition(":")
    if not sep or not mod_name or not attr:
        raise ValueError(f"Invalid import {qualified!r}; expected 'module:callable'.")
    module = importlib.import_module(mod_name)
    return getattr(module, attr)


def run_with_config(config_path: Path) -> None:
    cfg = load_run_config(config_path)

    predictor = build_predictor(cfg.black_box)
    mic_indices, _resolved = resolve_mic_columns(
        list(predictor.pathogen_list), cfg.black_box
    )
    signature = black_box_signature(cfg.black_box, len(mic_indices))

    opt = cfg.optimization
    factory_path = opt.solver.factory_import_path
    factory = import_from_path(factory_path)
    if not callable(factory):
        raise TypeError(
            f"solver.factory_import_path {factory_path!r} must name a callable factory."
        )

    eval_cap = opt.max_total_evaluations
    eval_budget = float("inf") if eval_cap is None else float(eval_cap)

    fieldnames = (
        "seed_id",
        "seed_sequence",
        "best_sequence",
        "best_fitness",
        "black_box_signature",
        "status",
        "error_message",
        "config_path",
        "solver_factory_path",
        "apex_path",
        "aggregation",
        "mic_transform",
        "random_seed_used",
        "max_total_evaluations",
        "max_iterations",
        "solver_iterations",
        "black_box_evaluations",
    )

    output_root = cfg.output.directory.resolve()
    if cfg.output.mode == "write":
        file_mode = "w"
    elif cfg.output.mode == "append":
        file_mode = "a"
    else:
        raise ValueError(f"Unknown output mode {cfg.output.mode!r}")

    seed_path = cfg.seed_sequences.path
    with seed_path.open(encoding="utf-8", newline="") as sfh:
        reader = csv.DictReader(sfh)
        id_col = cfg.seed_sequences.id_column
        seq_col = cfg.seed_sequences.sequence_column
        if reader.fieldnames is None or id_col not in reader.fieldnames:
            raise ValueError(
                f"Seed sequences CSV {seed_path} must contain column {id_col!r}; "
                f"have {reader.fieldnames!r}."
            )
        if seq_col not in reader.fieldnames:
            raise ValueError(
                f"Seed sequences CSV {seed_path} must contain column {seq_col!r}; "
                f"have {reader.fieldnames!r}."
            )
        rows = list(reader)

    output_root.mkdir(parents=True, exist_ok=True)
    for random_seed in opt.random_seeds:
        rng_seed = int(random_seed)
        out_path = output_root / f"seed_{rng_seed}.csv"
        if cfg.output.mode == "append":
            need_header = not out_path.exists() or out_path.stat().st_size == 0
        else:
            need_header = True

        with out_path.open(file_mode, encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if need_header:
                writer.writeheader()

            for row_index, row in tqdm(enumerate(rows), total=len(rows), desc=f"seed_{rng_seed}"):
                seed_id_raw = row.get(id_col, "")
                seq_raw = row.get(seq_col, "")
                seed_id = str(seed_id_raw).strip() if seed_id_raw is not None else ""
                normalized, err_msg = validate_seed_sequence(
                    str(seq_raw) if seq_raw is not None else ""
                )
                common_meta = {
                    "config_path": str(config_path.resolve()),
                    "solver_factory_path": factory_path,
                    "apex_path": cfg.black_box.apex.path,
                    "aggregation": cfg.black_box.aggregation,
                    "mic_transform": cfg.black_box.mic_transform,
                    "random_seed_used": str(rng_seed),
                    "max_total_evaluations": "" if eval_cap is None else str(eval_cap),
                    "max_iterations": (
                        "" if opt.max_iterations is None else str(opt.max_iterations)
                    ),
                }

                if err_msg:
                    _write_result_row(
                        writer,
                        fh,
                        common_meta,
                        seed_id=seed_id,
                        seed_sequence=normalized,
                        best_sequence="",
                        best_fitness="",
                        black_box_signature=signature,
                        status="error",
                        error_message=err_msg,
                        solver_iterations="",
                        black_box_evaluations="",
                    )
                    continue

                L = len(normalized)
                seed_python_numpy_torch(rng_seed)
                black_box_inst = ApexBlackBox(
                    predictor=predictor,
                    mic_column_indices=mic_indices,
                    aggregation=cfg.black_box.aggregation,
                    mic_transform=cfg.black_box.mic_transform,
                    sequence_length=L,
                    batch_size=cfg.black_box.apex.batch_size,
                    evaluation_budget=eval_budget,
                )
                x0 = np.array([list(normalized)])
                y0 = black_box_inst(x0)

                solver = factory(
                    black_box_inst,
                    x0,
                    y0,
                    dict(opt.solver.kwargs),
                )
                if not isinstance(solver, StepByStepSolver):
                    raise TypeError(
                        f"Solver factory {factory_path!r} must return a StepByStepSolver; "
                        f"got {type(solver).__name__}."
                    )

                iterations = 0
                while True:
                    if opt.max_iterations is not None and iterations >= opt.max_iterations:
                        break
                    if eval_cap is not None and black_box_inst.num_evaluations >= eval_cap:
                        break
                    try:
                        solver.step()
                    except BudgetExhaustedException:
                        break
                    iterations += 1

                best_x_arr = solver.get_best_solution(top_k=1)
                best_y_arr = solver.get_best_performance()
                best_seq = "".join(str(x) for x in best_x_arr.reshape(-1).tolist())
                best_fit = float(np.asarray(best_y_arr).reshape(-1)[0])

                _write_result_row(
                    writer,
                    fh,
                    common_meta,
                    seed_id=seed_id or str(row_index),
                    seed_sequence=normalized,
                    best_sequence=best_seq,
                    best_fitness=f"{best_fit:.17g}",
                    black_box_signature=signature,
                    status="ok",
                    error_message="",
                    solver_iterations=str(iterations),
                    black_box_evaluations=str(int(black_box_inst.num_evaluations)),
                )


def cli(
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", exists=True, dir_okay=False, readable=True),
    ],
) -> None:
    """Optimize each seed sequence from the config-driven CSV."""
    run_with_config(config_path)


def main() -> None:
    typer.run(cli)


if __name__ == "__main__":
    main()
