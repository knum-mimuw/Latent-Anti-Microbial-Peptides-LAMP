from __future__ import annotations

from pathlib import Path

import numpy as np

from amp_opt.oracle import ApexPanelOracle
from amp_opt.strain_map import load_strain_map


def test_load_strain_map_from_yaml() -> None:
    path = Path("src/amp_opt/src/amp_opt/configs/apex_strain_map.yaml")
    strain_map = load_strain_map(path)
    assert strain_map.mapped_name("E. coli AIC222") == "E. coli AIG222"
    assert strain_map.mapped_name("A. baumannii ATCC BAA-1605") is None


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
        # Deterministic test matrix: row i contains values [1..11] + i
        base = np.arange(1, len(self.pathogen_list) + 1, dtype=float)
        return np.vstack([base + i for i in range(len(sequences))])


def test_oracle_scores_over_resolved_strains_only() -> None:
    oracle = ApexPanelOracle(
        panel="mdr_eskape",
        predictor=_FakePredictor(),
        strain_map_path=Path("src/amp_opt/src/amp_opt/configs/apex_strain_map.yaml"),
    )
    scores = oracle.score_sequences(["AAAA", "CCCC"])
    # Resolved MDR strains in this fake setup map to AIG222, MRSA, VRE strains.
    assert np.allclose(scores, np.asarray([8.5, 9.5]))
    assert "A. baumannii ATCC BAA-1605" in oracle.resolution.unresolved_strains
