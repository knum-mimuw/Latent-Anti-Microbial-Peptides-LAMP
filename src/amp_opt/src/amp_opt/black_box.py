"""APEX-backed black box: MIC columns + aggregation, scalar fitness (maximize = better potency)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation

from amp_opt.config import BlackBoxConfig
from amp_opt.sequence_constants import STANDARD_AA_ORDER, standard_aa_set
from apex import PredictorAPEX
from pep_eval.panels import panel_dict


def resolve_mic_columns(
    pathogen_list: list[str],
    cfg: BlackBoxConfig,
) -> tuple[list[int], list[str]]:
    """Map config to MIC column indices in APEX prediction order."""
    cols = cfg.columns
    plist = list(pathogen_list)
    if cols.mode == "all":
        selected = plist
    elif cols.mode == "explicit":
        assert cols.pathogen_names is not None
        missing = [n for n in cols.pathogen_names if n not in plist]
        if missing:
            raise ValueError(f"pathogen_names not on predictor: {missing}")
        selected = list(cols.pathogen_names)
    else:
        assert cols.panel is not None
        panel = set(panel_dict()[cols.panel])
        selected = [name for name in plist if name in panel]
        if not selected:
            raise ValueError(
                f"Panel {cols.panel!r} has empty intersection with this APEX pathogen list."
            )

    idx = [plist.index(name) for name in selected]
    return idx, selected


def black_box_signature(cfg: BlackBoxConfig, n_columns: int) -> str:
    return (
        f"apex_path={cfg.apex.path}|columns_mode={cfg.columns.mode}"
        f"|mic_columns={n_columns}|aggregation={cfg.aggregation}"
        f"|mic_transform={cfg.mic_transform}"
    )


def _aggregate(mic_block: np.ndarray, how: Literal["mean", "max", "min"]) -> np.ndarray:
    if how == "mean":
        return np.nanmean(mic_block, axis=1)
    if how == "max":
        return np.nanmax(mic_block, axis=1)
    if how == "min":
        return np.nanmin(mic_block, axis=1)
    raise ValueError(how)


class ApexBlackBox(AbstractBlackBox):
    """POLi discrete oracle: lower MIC is better; black box returns −MIC_agg for maximization."""

    def __init__(
        self,
        *,
        predictor: PredictorAPEX,
        mic_column_indices: list[int],
        aggregation: Literal["mean", "max", "min"],
        mic_transform: Literal["none", "log2"],
        sequence_length: int,
        batch_size: int | None,
        evaluation_budget: float | int | None,
    ):
        if evaluation_budget is None or evaluation_budget == float("inf"):
            ebudget: int | None = None
        else:
            ebudget = int(evaluation_budget)

        super().__init__(batch_size=batch_size, evaluation_budget=ebudget)
        self._predictor = predictor
        self._mic_column_indices = list(mic_column_indices)
        self._aggregation = aggregation
        self._mic_transform = mic_transform
        self._sequence_length = int(sequence_length)
        self._allowed = standard_aa_set()

        alphabet = list(STANDARD_AA_ORDER)
        self._info = BlackBoxInformation(
            "apex_mic_scalar",
            self._sequence_length,
            True,
            True,
            True,
            alphabet,
        )

    def get_black_box_info(self) -> BlackBoxInformation:
        return self._info

    def _black_box(self, x: NDArray[np.str_], context: object | None = None) -> np.ndarray:
        if x.shape[1] != self._sequence_length:
            raise ValueError(
                f"Expected L={self._sequence_length}, got token array with shape {x.shape}."
            )
        seqs: list[str] = []
        for row in x:
            token_row = [str(t) for t in row.tolist()]
            for t in token_row:
                if t not in self._allowed:
                    raise ValueError(f"Disallowed residue token {t!r}.")
            seqs.append("".join(token_row))

        mic = self._predictor.predict(seqs, use_tqdm=False)
        if mic.ndim != 2 or mic.shape[0] != len(seqs):
            raise RuntimeError(f"Unexpected MIC shape {mic.shape} for batch {len(seqs)}.")

        idx = self._mic_column_indices
        mic_slice = mic[:, idx]
        if self._mic_transform == "log2":
            if np.any(mic_slice <= 0):
                raise RuntimeError(
                    "mic_transform='log2' requires strictly positive MIC predictions; "
                    "got non-positive values from APEX."
                )
            mic_slice = np.log2(mic_slice)

        agg_mic = _aggregate(mic_slice, self._aggregation)
        if np.any(np.isnan(agg_mic)):
            raise RuntimeError("Aggregated MIC contains NaN for one or more sequences.")

        fitness = (-agg_mic.astype(np.float64)).reshape(-1, 1)
        return fitness


def build_predictor(cfg: BlackBoxConfig) -> PredictorAPEX:
    return PredictorAPEX(
        device=cfg.apex.device,
        batch_size=cfg.apex.batch_size,
        path=cfg.apex.path,
    )
