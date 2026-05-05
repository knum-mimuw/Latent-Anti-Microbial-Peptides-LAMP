"""CLI entrypoint for AMP optimization."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .core.objective import ApexPanelObjective, ScoreTransform
from .core.runner import OptimizationRunner
from .encoding import AlphabetCodec
from .methods.random_search import RandomSearchConfig, RandomSearchMethod
from .methods.random_search.config import ParentPool
from .oracle import ApexPanelOracle, ObjectivePanel

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _read_initial_sequences(
    *,
    sequence: str | None,
    sequence_file: Path | None,
) -> list[str]:
    values: list[str] = []
    if sequence:
        values.extend([s.strip() for s in sequence.split(",") if s.strip()])
    if sequence_file:
        values.extend(
            [line.strip() for line in sequence_file.read_text("utf-8").splitlines() if line.strip()]
        )
    if not values:
        raise typer.BadParameter(
            "Provide --sequence and/or --sequence-file with at least one sequence."
        )
    return values


@app.command("run")
def run_command(
    panel: Annotated[
        ObjectivePanel,
        typer.Option(help="Objective panel: gram_negative, gram_positive, or mdr_eskape."),
    ] = "gram_negative",
    sequence: Annotated[
        str | None,
        typer.Option(help="Comma-separated initial peptide sequences."),
    ] = None,
    sequence_file: Annotated[
        Path | None,
        typer.Option(help="Path to newline-separated initial sequences."),
    ] = None,
    method: Annotated[
        str,
        typer.Option(help="Optimization method. Currently: random_search."),
    ] = "random_search",
    generations: Annotated[int, typer.Option(help="Number of generations.")] = 50,
    population_size: Annotated[int, typer.Option(help="Population size per generation.")] = 64,
    mutation_count: Annotated[int, typer.Option(help="Mutated positions per proposal.")] = 1,
    parent_pool: Annotated[
        ParentPool,
        typer.Option(help="Parent pool strategy: starting, archive_top_k, archive_all, best."),
    ] = "archive_top_k",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    max_length: Annotated[int, typer.Option(help="Maximum peptide length.")] = 50,
    score_transform: Annotated[
        ScoreTransform,
        typer.Option(help="Score transform: raw_mic or log10_mic."),
    ] = "raw_mic",
    apex_device: Annotated[
        str,
        typer.Option(help="Torch device for APEX predictor."),
    ] = os.getenv("APEX_DEVICE", "cpu"),
    apex_batch_size: Annotated[int, typer.Option(help="APEX predictor batch size.")] = 2048,
    strain_map_path: Annotated[
        Path | None,
        typer.Option(help="Custom strain mapping YAML."),
    ] = None,
    top_k: Annotated[int, typer.Option(help="Number of top candidates to report.")] = 20,
    output_json: Annotated[
        Path | None,
        typer.Option(help="Output path for result JSON."),
    ] = None,
) -> None:
    """Run AMP panel optimization."""
    initial_sequences = _read_initial_sequences(sequence=sequence, sequence_file=sequence_file)

    oracle = ApexPanelOracle(
        panel=panel,
        strain_map_path=strain_map_path,
        device=apex_device,
        batch_size=apex_batch_size,
    )
    objective = ApexPanelObjective(oracle=oracle, score_transform=score_transform)

    if method != "random_search":
        raise typer.BadParameter(f"Unknown method '{method}'. Available: random_search")

    codec = AlphabetCodec(max_length=max_length)
    opt_method = RandomSearchMethod(
        config=RandomSearchConfig(
            population_size=population_size,
            mutation_count=mutation_count,
            seed=seed,
            parent_pool=parent_pool,
        ),
        codec=codec,
    )

    runner = OptimizationRunner(objective=objective, method=opt_method)
    result = runner.run(
        starting_sequences=initial_sequences,
        generations=generations,
        top_k=top_k,
    )

    payload = {
        "panel": panel,
        "method": method,
        "generations": generations,
        "best_sequence": result.best.sequence,
        "best_score": result.best.score,
        "best_generation": result.best.generation,
        "top_k": [
            {"sequence": c.sequence, "score": c.score, "generation": c.generation}
            for c in result.top_k
        ],
        "config": result.config,
    }
    typer.echo(json.dumps(payload, indent=2))
    if output_json:
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
