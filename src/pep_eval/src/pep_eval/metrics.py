"""Metrics computation for AMP evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .panels import (
    GRAM_NEGATIVE_STRAINS,
    GRAM_POSITIVE_STRAINS,
    MDR_ESKAPE_STRAINS,
    OVERALL_STRAINS,
    POTENCY_THRESHOLD_UM,
)
from .parsing import parse_measurement


def panel_metrics(values: list[float]) -> tuple[float, float, float]:
    """Return success_rate, mic50, mic90 for a panel."""
    arr = np.asarray(values, dtype=float)
    success_rate = float(np.mean(arr <= POTENCY_THRESHOLD_UM))
    mic50 = float(np.percentile(arr, 50))
    mic90 = float(np.percentile(arr, 90))
    return success_rate, mic50, mic90


def compute_per_peptide_metrics(
    records: list[dict[str, Any]],
    sequence_column: str,
    hc50_column: str,
    strain_columns: dict[str, str],
) -> list[dict[str, Any]]:
    """Compute per-peptide metrics over all category panels."""
    missing_strains = sorted(set(OVERALL_STRAINS) - set(strain_columns.keys()))
    if missing_strains:
        raise ValueError(
            "strain_columns is missing required canonical strains: " + ", ".join(missing_strains)
        )

    per_peptide: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        sequence = record.get(sequence_column)
        if sequence is None:
            raise ValueError(f"Missing sequence at row {idx} (column '{sequence_column}').")

        hc50 = parse_measurement(record.get(hc50_column), default_censor_limit=128.0).value

        strain_values: dict[str, float] = {}
        for strain, source_column in strain_columns.items():
            strain_values[strain] = parse_measurement(
                record.get(source_column), default_censor_limit=64.0
            ).value

        overall_values = [strain_values[name] for name in OVERALL_STRAINS]
        gram_pos_values = [strain_values[name] for name in GRAM_POSITIVE_STRAINS]
        gram_neg_values = [strain_values[name] for name in GRAM_NEGATIVE_STRAINS]
        mdr_values = [strain_values[name] for name in MDR_ESKAPE_STRAINS]

        overall_success, overall_mic50, overall_mic90 = panel_metrics(overall_values)
        gp_success, gp_mic50, _ = panel_metrics(gram_pos_values)
        gn_success, gn_mic50, _ = panel_metrics(gram_neg_values)
        mdr_success, mdr_mic50, _ = panel_metrics(mdr_values)

        active_any = bool(np.any(np.asarray(overall_values) <= POTENCY_THRESHOLD_UM))
        safety_window = float(hc50 / overall_mic50) if overall_mic50 > 0 else float("inf")

        per_peptide.append(
            {
                "row_index": idx,
                "sequence": str(sequence),
                "overall_success_rate": overall_success,
                "overall_mic50": overall_mic50,
                "overall_mic90": overall_mic90,
                "gram_positive_success_rate": gp_success,
                "gram_positive_mic50": gp_mic50,
                "gram_negative_success_rate": gn_success,
                "gram_negative_mic50": gn_mic50,
                "mdr_eskape_success_rate": mdr_success,
                "mdr_eskape_mic50": mdr_mic50,
                "hc50": hc50,
                "safety_window": safety_window,
                "active_any": active_any,
            }
        )

    return per_peptide


def compute_scorecard(per_peptide: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-peptide metrics into challenge category scorecard."""
    if not per_peptide:
        raise ValueError("No peptide rows available for scoring.")

    def _mean(key: str) -> float:
        return float(np.mean([row[key] for row in per_peptide]))

    selectivity_rows = [row for row in per_peptide if row["active_any"]]
    if not selectivity_rows:
        selectivity_safety_window = 0.0
        selectivity_mic50_overall = float(_mean("overall_mic50"))
        eligible_fraction = 0.0
    else:
        selectivity_safety_window = float(np.mean([row["safety_window"] for row in selectivity_rows]))
        selectivity_mic50_overall = float(np.min([row["overall_mic50"] for row in selectivity_rows]))
        eligible_fraction = float(len(selectivity_rows) / len(per_peptide))

    return {
        "broad_spectrum/success_rate": _mean("overall_success_rate"),
        "broad_spectrum/mic90_overall": _mean("overall_mic90"),
        "gram_positive/success_rate": _mean("gram_positive_success_rate"),
        "gram_positive/mic50": _mean("gram_positive_mic50"),
        "gram_negative/success_rate": _mean("gram_negative_success_rate"),
        "gram_negative/mic50": _mean("gram_negative_mic50"),
        "mdr_eskape/success_rate": _mean("mdr_eskape_success_rate"),
        "mdr_eskape/mic50": _mean("mdr_eskape_mic50"),
        "selectivity/safety_window_mean": selectivity_safety_window,
        "selectivity/mic50_overall": selectivity_mic50_overall,
        "selectivity/eligible_fraction": eligible_fraction,
    }
