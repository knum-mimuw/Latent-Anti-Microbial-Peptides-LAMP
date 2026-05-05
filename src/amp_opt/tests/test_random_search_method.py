"""Integration test for RandomSearchMethod with fake predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from amp_opt.core.objective import ApexPanelObjective
from amp_opt.core.runner import OptimizationRunner
from amp_opt.encoding import AlphabetCodec
from amp_opt.methods.random_search import RandomSearchConfig, RandomSearchMethod
from amp_opt.oracle import ApexPanelOracle


class _FakePredictor:
    def __init__(self) -> None:
        self.pathogen_list = [
            "A. baumannii ATCC 19606",
            "E. coli ATCC 11775",
            "E. coli AIG221",
            "E. coli AIG222",
            "K. pneumoniae ATCC 13883",
            "P. aeruginosa PA01",
            "P. aeruginosa PA14",
            "S. aureus ATCC 12600",
            "S. aureus (ATCC BAA-1556) - MRSA",
            "vancomycin-resistant E. faecalis ATCC 700802",
            "vancomycin-resistant E. faecium ATCC 700221",
        ]

    def predict(self, sequences: list[str], use_tqdm: bool = False) -> np.ndarray:
        _ = use_tqdm
        rng = np.random.default_rng(42)
        return rng.uniform(1, 100, size=(len(sequences), len(self.pathogen_list)))


def test_random_search_method_runs_with_runner() -> None:
    oracle = ApexPanelOracle(
        panel="gram_negative",
        predictor=_FakePredictor(),
        strain_map_path=Path("src/amp_opt/src/amp_opt/configs/apex_strain_map.yaml"),
    )
    objective = ApexPanelObjective(oracle=oracle, score_transform="raw_mic")
    codec = AlphabetCodec(max_length=20)
    method = RandomSearchMethod(
        config=RandomSearchConfig(population_size=8, mutation_count=1, seed=0),
        codec=codec,
    )
    runner = OptimizationRunner(objective=objective, method=method)
    result = runner.run(starting_sequences=["KKLLKLLK", "KRWWKWWRR"], generations=3, top_k=5)
    assert result.best is not None
    assert len(result.best.sequence) > 0
    assert len(result.history) == 3
    assert result.best.score > 0
