"""Unit tests for pep_eval metrics and parsing."""

from __future__ import annotations

import pytest

from pep_eval.metrics import compute_per_peptide_metrics, compute_scorecard
from pep_eval.parsing import parse_measurement


def test_parse_measurement_supports_censored_values() -> None:
    parsed = parse_measurement(">64", default_censor_limit=64.0)
    assert parsed.value == pytest.approx(64.0)
    assert parsed.censored_high is True

    parsed_plain = parse_measurement("12.5", default_censor_limit=64.0)
    assert parsed_plain.value == pytest.approx(12.5)
    assert parsed_plain.censored_high is False


def test_compute_scorecard_includes_all_categories() -> None:
    records = [
        {
            "sequence": "AAAAAAKK",
            "hc50": "128",
            **{strain: "8" for strain in _all_strain_names()},
        },
        {
            "sequence": "RRRRRRRR",
            "hc50": "64",
            **{strain: ">64" for strain in _all_strain_names()},
        },
    ]
    strain_columns = {strain: strain for strain in _all_strain_names()}

    per_peptide = compute_per_peptide_metrics(
        records=records,
        sequence_column="sequence",
        hc50_column="hc50",
        strain_columns=strain_columns,
    )
    scorecard = compute_scorecard(per_peptide)

    expected_keys = {
        "broad_spectrum/success_rate",
        "broad_spectrum/mic90_overall",
        "gram_positive/success_rate",
        "gram_positive/mic50",
        "gram_negative/success_rate",
        "gram_negative/mic50",
        "mdr_eskape/success_rate",
        "mdr_eskape/mic50",
        "selectivity/safety_window_mean",
        "selectivity/mic50_overall",
        "selectivity/eligible_fraction",
    }
    assert expected_keys.issubset(scorecard.keys())
    assert scorecard["selectivity/eligible_fraction"] == pytest.approx(0.5)


def _all_strain_names() -> list[str]:
    return [
        "A. baumannii ATCC 19606",
        "A. baumannii ATCC BAA-1605",
        "E. cloacae ATCC 13047",
        "E. coli ATCC 11775",
        "E. coli AIC221",
        "E. coli AIC222",
        "E. coli ATCC BAA-3170",
        "E. coli K-12 BW25113",
        "K. pneumoniae ATCC 13883",
        "K. pneumoniae ATCC BAA-2342",
        "P. aeruginosa PAO1",
        "P. aeruginosa PA14",
        "P. aeruginosa ATCC BAA-3197",
        "S. enterica ATCC 9150",
        "S. enterica Typhimurium ATCC 700720",
        "B. subtilis ATCC 23857",
        "S. aureus ATCC 12600",
        "S. aureus ATCC BAA-1556",
        "E. faecalis ATCC 700802",
        "E. faecium ATCC 700221",
    ]
