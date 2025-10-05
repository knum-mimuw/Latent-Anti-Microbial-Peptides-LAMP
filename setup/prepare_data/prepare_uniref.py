import requests
from typing import Generator, Dict, Any, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field, Extra
from utils import SequenceItem


class PrepareUniRefConfig(BaseModel):
    """Configuration for preparing UniRef sequences."""

    url: str = Field(
        "https://rest.uniprot.org/uniref/search", description="UniRef REST URL"
    )
    query: str = Field("length:[* TO 50]", description="UniRef query string")

    class Config:
        extra = Extra.allow  # Allow future extensions without breaking


def create_sequence_item(item: Dict[str, Any]) -> SequenceItem:
    """Create a standardized sequence item from UniRef data."""
    return {
        "cluster_id": item["cluster_id"],
        "sequence": item["sequence"],
        "length": item["length"],
    }


def prepare_uniref_short_sequences(
    *,
    max_sequences: Optional[int] = None,
    max_length: int = 50,
    cfg: Optional[PrepareUniRefConfig] = None,
) -> Generator[SequenceItem, None, None]:
    """Prepare short sequences from UniRef using streaming."""
    if max_sequences is None:
        raise ValueError("max_sequences must be specified for UniRef API")

    if cfg is None:
        cfg = PrepareUniRefConfig()

    query = cfg.query
    url = cfg.url
    params = {
        "query": query,
        "format": "fasta",
        "size": max_sequences,
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    fasta_text = resp.text.strip()

    seq = []
    header = None
    count = 0

    for line in tqdm(fasta_text.splitlines(), desc="Processing sequences"):
        if line.startswith(">"):
            if header and seq:
                sequence = "".join(seq)
                if len(sequence) <= max_length:
                    yield create_sequence_item(parse_fasta_entry(header, sequence))
                    count += 1
                    if max_sequences and count >= max_sequences:
                        break
            header = line[1:].strip()
            seq = []
        else:
            seq.append(line.strip())

    # Process last sequence
    if header and seq:
        sequence = "".join(seq)
        if len(sequence) <= max_length:
            yield create_sequence_item(parse_fasta_entry(header, sequence))


def parse_fasta_entry(header: str, sequence: str) -> Dict[str, Any]:
    """Parse a UniRef FASTA header into a dict."""
    # UniRef format: >UniRef50_XXXXX n=XXX Tax=XXX RepID=XXXXX
    cluster_id = header.split()[0]  # Get the cluster ID (e.g., UniRef50_XXXXX)
    return {
        "cluster_id": cluster_id,
        "sequence": sequence,
        "length": len(sequence),
    }


def _demo():
    print("🧬 Demo of UniRef sequence preparation...")
    print("=" * 50)

    sequences = list(prepare_uniref_short_sequences(max_sequences=5, max_length=50))

    print(f"✅ Successfully prepared {len(sequences)} sequences")
    print("\n📊 Sample sequences:")
    for i, seq in enumerate(sequences, 1):
        print(f"  {i}. {seq['cluster_id']} ({seq['length']} AA)")
        print(f"     {seq['sequence']}")
        print()


if __name__ == "__main__":
    _demo()
