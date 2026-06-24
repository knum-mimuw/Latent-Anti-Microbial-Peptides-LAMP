"""Fixed peptide semantics (aligned with APEX vocab in lamp-apex)."""

from __future__ import annotations

MAX_PEPTIDE_LENGTH = 50

_STANDARD_AAS: list[str] = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
STANDARD_AA_ORDER: tuple[str, ...] = tuple(sorted(_STANDARD_AAS))


def standard_aa_set() -> frozenset[str]:
    return frozenset(STANDARD_AA_ORDER)
