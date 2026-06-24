import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import typer
from Bio import SeqIO
from datasets import Features, Value
from pydantic import BaseModel, Extra, Field
from tqdm import tqdm

from .upload_data_to_huggingface import (
    UploadConfig,
    upload_streams_to_huggingface_subset,
)
from .utils import load_config_file

CONFIG_FILE_OPTION = typer.Option(
    ...,
    "--config",
    "-c",
    help=(
        "YAML/JSON config with separate 'prepare' and 'upload' sections. "
        "Prepare section: antigram_n_path, antigram_p_path, max_length, "
        "allowed_amino_acids, max_sequences, source_version. "
        "Upload section: repo_id, subset_name, commit_message, columns, token."
    ),
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)

DBAMP_UPLOAD_FEATURES = Features(
    {
        "dbamp_id": Value("string"),
        "sequence": Value("string"),
        "anti_gram_negative": Value("bool"),
        "anti_gram_positive": Value("bool"),
    }
)

DBAMPRecord = dict[str, Any]


class PrepareDBAMPConfig(BaseModel):
    """Configuration for dbAMP peptide ingestion from local FASTA files."""

    antigram_n_path: Path = Field(
        Path("data/dbamp/dbAMP_AntiGram_n_2024.fasta"),
        description="Path to the anti-Gram-negative dbAMP FASTA file.",
    )
    antigram_p_path: Path = Field(
        Path("data/dbamp/dbAMP_AntiGram_p_2024.fasta"),
        description="Path to the anti-Gram-positive dbAMP FASTA file.",
    )
    max_length: int | None = Field(
        50,
        description="Keep sequences with length <= this value. Null disables the cap.",
    )
    allowed_amino_acids: str | None = Field(
        "ACDEFGHIKLMNPQRSTVWY",
        description="Optional allowed amino-acid alphabet used for sequence filtering.",
    )
    max_sequences: int | None = Field(
        None,
        description="Optional cap on number of yielded sequences (for smoke tests).",
    )
    source_version: str = Field(
        "v3.0 (RELEASE 06/2024)",
        description="Database version string echoed in CLI output for traceability.",
    )

    class Config:
        extra = Extra.allow


def _has_only_allowed_amino_acids(sequence: str, allowed_amino_acids: frozenset[str]) -> bool:
    return set(sequence.upper()).issubset(allowed_amino_acids)


def _ensure_fasta_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} FASTA file not found: {path}")


def _load_dbamp_records(cfg: PrepareDBAMPConfig) -> dict[str, DBAMPRecord]:
    """Merge the two dbAMP FASTA files into one map keyed by dbAMP id.

    A sequence mismatch on the same id is a hard error (no silent reconciliation).
    """
    _ensure_fasta_exists(cfg.antigram_n_path, "anti-Gram-negative")
    _ensure_fasta_exists(cfg.antigram_p_path, "anti-Gram-positive")

    records: dict[str, DBAMPRecord] = {}
    for path, flag in (
        (cfg.antigram_n_path, "anti_gram_negative"),
        (cfg.antigram_p_path, "anti_gram_positive"),
    ):
        for rec in SeqIO.parse(str(path), "fasta"):
            dbamp_id = rec.id
            sequence = str(rec.seq)
            existing = records.get(dbamp_id)
            if existing is None:
                existing = {
                    "dbamp_id": dbamp_id,
                    "sequence": sequence,
                    "anti_gram_negative": False,
                    "anti_gram_positive": False,
                }
                records[dbamp_id] = existing
            elif existing["sequence"] != sequence:
                raise ValueError(
                    f"Sequence mismatch for {dbamp_id} between "
                    f"{cfg.antigram_n_path.name} and {cfg.antigram_p_path.name}."
                )
            existing[flag] = True
    return records


def _stream_dbamp(
    cfg: PrepareDBAMPConfig,
) -> Generator[DBAMPRecord, None, None]:
    """Yield filtered, deterministically-ordered dbAMP rows."""
    allowed_amino_acids: frozenset[str] | None = None
    if cfg.allowed_amino_acids is not None:
        normalized = cfg.allowed_amino_acids.strip().upper()
        if not normalized:
            raise ValueError("allowed_amino_acids must not be empty when provided.")
        allowed_amino_acids = frozenset(normalized)

    records = _load_dbamp_records(cfg)
    yielded = 0
    for dbamp_id in tqdm(
        sorted(records.keys()), desc="Filtering dbAMP records", unit="rec"
    ):
        row = records[dbamp_id]
        sequence = row["sequence"]
        if not sequence:
            continue
        if cfg.max_length is not None and len(sequence) > cfg.max_length:
            continue
        if allowed_amino_acids is not None and not _has_only_allowed_amino_acids(
            sequence, allowed_amino_acids
        ):
            continue

        yield row
        yielded += 1
        if cfg.max_sequences and yielded >= cfg.max_sequences:
            return


def prepare_dbamp_sequences(
    cfg: PrepareDBAMPConfig,
) -> Generator[DBAMPRecord, None, None]:
    """Public entry returning a generator of standardized dbAMP items."""
    return _stream_dbamp(cfg)


def prepare_and_upload_dbamp_command(
    *,
    config_file: Path = CONFIG_FILE_OPTION,
) -> None:
    """Ingest dbAMP peptides from local FASTA files and upload as a HF subset."""
    raw = load_config_file(config_file)

    prepare_raw = raw.get("prepare", {})
    if not prepare_raw:
        typer.echo("❌ Config must include 'prepare' section.", err=True)
        raise typer.Exit(1)
    prepare_cfg = PrepareDBAMPConfig(**prepare_raw)

    upload_raw = raw.get("upload", {})
    if not upload_raw:
        typer.echo("❌ Config must include 'upload' section.", err=True)
        raise typer.Exit(1)
    if not upload_raw.get("token"):
        upload_raw["token"] = os.getenv("HF_TOKEN")
    upload_cfg = UploadConfig(**upload_raw)

    typer.echo(
        f"🚀 Preparing dbAMP peptides {prepare_cfg.source_version} "
        f"(max_length={prepare_cfg.max_length}, "
        f"AA filter={'on' if prepare_cfg.allowed_amino_acids else 'off'})"
    )

    streams = {"train": prepare_dbamp_sequences(prepare_cfg)}

    typer.echo(
        f"📤 Uploading as subset '{upload_cfg.subset_name}' under {upload_cfg.repo_id}..."
    )
    upload_streams_to_huggingface_subset(
        streams,
        repo_id=upload_cfg.repo_id,
        subset_name=upload_cfg.subset_name,
        columns=upload_cfg.columns,
        commit_message=upload_cfg.commit_message,
        token=upload_cfg.token,
        features=DBAMP_UPLOAD_FEATURES,
    )
    typer.echo(
        f"📦 Uploaded subset to "
        f"https://huggingface.co/datasets/{upload_cfg.repo_id}/{upload_cfg.subset_name}"
    )


def _demo() -> None:
    cfg = PrepareDBAMPConfig(max_sequences=5)
    stream = prepare_dbamp_sequences(cfg)
    print(f"📊 Sample from dbAMP {cfg.source_version} (first 5 items):")
    for idx, row in enumerate(stream, start=1):
        flags = []
        if row["anti_gram_negative"]:
            flags.append("Gram-")
        if row["anti_gram_positive"]:
            flags.append("Gram+")
        print(
            f"  {idx}. {row['dbamp_id']} ({len(row['sequence'])} AA) "
            f"[{','.join(flags) or 'no-label'}] {row['sequence']}"
        )


if __name__ == "__main__":
    _demo()
