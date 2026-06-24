from collections.abc import Iterable

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from peptides import Peptide

SUPPORTED_PROPERTIES = {
    "charge",
    "hydrophobicity",
    "isoelectric_point",
    "molecular_weight",
    "instability_index",
    "gravy",
    "aliphatic_index",
    "boman_index",
}


def compute_properties(
    sequence: str,
    *,
    ph: float,
    hydrophobicity_scale: str,
    properties: Iterable[str],
) -> dict[str, float]:
    """Compute selected physicochemical properties for one peptide sequence."""
    selected = list(properties)
    unknown = [name for name in selected if name not in SUPPORTED_PROPERTIES]
    if unknown:
        raise ValueError(f"Unsupported physicochemical properties requested: {unknown}")

    bio = ProteinAnalysis(sequence)
    pep = Peptide(sequence)

    computed: dict[str, float] = {}
    for name in selected:
        if name == "charge":
            computed[name] = float(bio.charge_at_pH(ph))
        elif name == "hydrophobicity":
            computed[name] = float(pep.hydrophobicity(scale=hydrophobicity_scale))
        elif name == "isoelectric_point":
            computed[name] = float(bio.isoelectric_point())
        elif name == "molecular_weight":
            computed[name] = float(bio.molecular_weight())
        elif name == "instability_index":
            computed[name] = float(bio.instability_index())
        elif name == "gravy":
            computed[name] = float(bio.gravy())
        elif name == "aliphatic_index":
            computed[name] = float(pep.aliphatic_index())
        elif name == "boman_index":
            computed[name] = float(pep.boman())
    return computed
