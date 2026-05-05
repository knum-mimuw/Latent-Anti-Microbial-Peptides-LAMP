"""CLI entrypoint for AMP optimization with evosax RandomSearch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from .encoding import AlphabetCodec
from .evosax_runner import RandomSearchConfig, run_random_search
from .fitness import GenomeFitness
from .oracle import ApexPanelOracle, ObjectivePanel

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _read_initial_sequences(
    *,
    sequence: str | None,
    sequence_file: Path | None,
) -> list[str]:
    values: list[str] = []
    if sequence:
        values.extend([entry.strip() for entry in sequence.split(",") if entry.strip()])
    if sequence_file:
        values.extend(
            [
                line.strip()
                for line in sequence_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )
    if not values:
        raise ValueError("Provide --sequence and/or --sequence-file with at least one sequence.")
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
    generations: Annotated[int, typer.Option(help="Number of evolutionary generations.")] = 50,
    population_size: Annotated[int, typer.Option(help="Population size per generation.")] = 64,
    mutation_count: Annotated[int, typer.Option(help="Mutated positions per proposal.")] = 1,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    max_length: Annotated[int, typer.Option(help="Maximum peptide length for genome codec.")] = 50,
    apex_device: Annotated[
        str,
        typer.Option(
            help="Torch device for APEX predictor. Defaults to APEX_DEVICE env or cpu.",
        ),
    ] = os.getenv("APEX_DEVICE", "cpu"),
    apex_batch_size: Annotated[int, typer.Option(help="APEX predictor batch size.")] = 2048,
    strain_map_path: Annotated[
        Path | None,
        typer.Option(help="Optional path to custom strain mapping YAML."),
    ] = None,
    use_log_mic: Annotated[
        bool,
        typer.Option(help="Optimize log10(MIC) instead of raw MIC."),
    ] = False,
    output_json: Annotated[
        Path | None,
        typer.Option(help="Optional output path for result JSON."),
    ] = None,
) -> None:
    """Run APEX-full panel optimization with evosax RandomSearch."""
    initial_sequences = _read_initial_sequences(sequence=sequence, sequence_file=sequence_file)
    codec = AlphabetCodec(max_length=max_length)
    oracle = ApexPanelOracle(
        panel=panel,
        strain_map_path=strain_map_path,
        device=apex_device,
        batch_size=apex_batch_size,
    )
    fitness = GenomeFitness(codec=codec, oracle=oracle, use_log_mic=use_log_mic)
    result = run_random_search(
        fitness=fitness,
        initial_sequences=initial_sequences,
        search_config=RandomSearchConfig(
            seed=seed,
            population_size=population_size,
            generations=generations,
            mutation_count=mutation_count,
        ),
    )

    payload = {
        "panel": panel,
        "best_sequence": result.best_sequence,
        "best_fitness": result.best_fitness,
        "best_mic": result.best_mic,
        "best_genome": list(result.best_genome),
        "resolved_apex_columns": list(result.resolved_apex_columns),
        "unresolved_strains": list(result.unresolved_strains),
    }
    typer.echo(json.dumps(payload, indent=2))
    if output_json:
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
