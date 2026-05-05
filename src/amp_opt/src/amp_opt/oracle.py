"""APEX-backed scalar objective oracle for AMP panels."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np

from pep_eval.panels import panel_dict

from .strain_map import load_strain_map

ObjectivePanel = Literal["gram_negative", "gram_positive", "mdr_eskape"]


@dataclass(frozen=True)
class PanelResolution:
    """Resolved panel indices against APEX output columns."""

    panel_name: ObjectivePanel
    canonical_strains: tuple[str, ...]
    resolved_apex_columns: tuple[str, ...]
    resolved_indices: tuple[int, ...]
    unresolved_strains: tuple[str, ...]


class ApexPanelOracle:
    """Evaluate sequences via APEX full-pathogen outputs for one challenge panel."""

    def __init__(
        self,
        *,
        panel: ObjectivePanel,
        predictor: object | None = None,
        strain_map_path: Path | None = None,
        device: str = "cpu",
        batch_size: int = 2048,
        use_tqdm: bool = False,
    ) -> None:
        self.panel = panel
        self.use_tqdm = use_tqdm
        self._predictor = predictor or self._make_predictor(device=device, batch_size=batch_size)
        self._strain_map = load_strain_map(strain_map_path or _default_strain_map_path())
        self._resolution = self._resolve_panel()

    @staticmethod
    def _make_predictor(*, device: str, batch_size: int):
        from apex import PredictorAPEX

        return PredictorAPEX(device=device, batch_size=batch_size, path="all")

    @property
    def resolution(self) -> PanelResolution:
        return self._resolution

    def _resolve_panel(self) -> PanelResolution:
        panels = panel_dict()
        if self.panel not in panels:
            raise ValueError(f"Unknown panel '{self.panel}'.")
        canonical = tuple(panels[self.panel])
        column_to_idx = {name: idx for idx, name in enumerate(self._predictor.pathogen_list)}

        resolved_columns: list[str] = []
        resolved_indices: list[int] = []
        unresolved: list[str] = []
        for strain in canonical:
            mapped = self._strain_map.mapped_name(strain)
            if mapped is None:
                unresolved.append(strain)
                continue
            idx = column_to_idx.get(mapped)
            if idx is None:
                unresolved.append(strain)
                continue
            resolved_columns.append(mapped)
            resolved_indices.append(idx)

        if not resolved_indices:
            raise ValueError(
                f"No strains resolved for panel '{self.panel}'. "
                f"Missing strains: {', '.join(canonical)}."
            )
        return PanelResolution(
            panel_name=self.panel,
            canonical_strains=canonical,
            resolved_apex_columns=tuple(resolved_columns),
            resolved_indices=tuple(resolved_indices),
            unresolved_strains=tuple(unresolved),
        )

    def score_sequences(self, sequences: list[str]) -> np.ndarray:
        """Return mean MIC per sequence over resolved strains for the selected panel."""
        if not sequences:
            raise ValueError("sequences must not be empty.")
        preds = np.asarray(self._predictor.predict(sequences, use_tqdm=self.use_tqdm), dtype=float)
        if preds.ndim != 2:
            raise ValueError(f"Expected 2D predictor output, got shape {preds.shape}.")
        indices = np.asarray(self._resolution.resolved_indices, dtype=np.int32)
        return np.mean(preds[:, indices], axis=1, dtype=float)

    def score_sequence(self, sequence: str) -> float:
        """Return mean MIC for one sequence."""
        return float(self.score_sequences([sequence])[0])


def _default_strain_map_path() -> Path:
    return Path(files("amp_opt.configs").joinpath("apex_strain_map.yaml"))
