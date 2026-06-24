"""Canonical AMP challenge bacterial panels."""

from __future__ import annotations

POTENCY_THRESHOLD_UM = 16.0

OVERALL_STRAINS: tuple[str, ...] = (
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
)

GRAM_NEGATIVE_STRAINS: tuple[str, ...] = (
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
)

GRAM_POSITIVE_STRAINS: tuple[str, ...] = (
    "B. subtilis ATCC 23857",
    "S. aureus ATCC 12600",
    "S. aureus ATCC BAA-1556",
    "E. faecalis ATCC 700802",
    "E. faecium ATCC 700221",
)

MDR_ESKAPE_STRAINS: tuple[str, ...] = (
    "A. baumannii ATCC BAA-1605",
    "E. coli AIC222",
    "E. coli ATCC BAA-3170",
    "K. pneumoniae ATCC BAA-2342",
    "P. aeruginosa ATCC BAA-3197",
    "S. aureus ATCC BAA-1556",
    "E. faecalis ATCC 700802",
    "E. faecium ATCC 700221",
)


def panel_dict() -> dict[str, list[str]]:
    """Return panel definitions as serializable lists."""
    return {
        "overall": list(OVERALL_STRAINS),
        "gram_positive": list(GRAM_POSITIVE_STRAINS),
        "gram_negative": list(GRAM_NEGATIVE_STRAINS),
        "mdr_eskape": list(MDR_ESKAPE_STRAINS),
    }
