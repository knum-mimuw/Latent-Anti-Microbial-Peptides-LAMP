"""Parsing helpers for assay values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedValue:
    """Parsed assay value with censor direction."""

    value: float
    censored_high: bool


def parse_measurement(raw: Any, default_censor_limit: float) -> ParsedValue:
    """Parse numeric and censored values like ``>64``."""
    if raw is None:
        raise ValueError("Encountered null assay value.")

    if isinstance(raw, (int, float)):
        return ParsedValue(float(raw), censored_high=False)

    text = str(raw).strip()
    if not text:
        raise ValueError("Encountered empty assay value.")

    if text.startswith(">"):
        limit = text[1:].strip()
        value = float(limit) if limit else default_censor_limit
        return ParsedValue(value=value, censored_high=True)

    return ParsedValue(value=float(text), censored_high=False)
