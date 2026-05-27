import os
from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal

import requests
import typer
from datasets import Features, Value
from pydantic import BaseModel, Extra, Field
from tqdm import tqdm

from .upload_data_to_huggingface import (
    UploadConfig,
    upload_streams_to_huggingface_subset,
)
from .utils import SequenceItem, load_config_file

CONFIG_FILE_OPTION = typer.Option(
    ...,
    "--config",
    "-c",
    help=(
        "YAML/JSON config with separate 'prepare' and 'upload' sections. "
        "Prepare section: base_url, page_size, complexity, max_length, "
        "allowed_amino_acids, max_sequences, request_timeout_s, user_agent. "
        "Upload section: repo_id, subset_name, commit_message, columns, token."
    ),
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)

DBAASP_UPLOAD_FEATURES = Features(
    {
        "dbaasp_id": Value("string"),
        "sequence": Value("string"),
    }
)

DBAASP_PEPTIDES_PATH = "/peptides"


class PrepareDBAASPConfig(BaseModel):
    """Configuration for DBAASP peptide ingestion."""

    base_url: str = Field(
        "https://dbaasp.org",
        description="Base URL for the DBAASP REST API.",
    )
    page_size: int = Field(
        500,
        description="Number of records fetched per /peptides request.",
        gt=0,
    )
    complexity: Literal["monomer", "multimer", "multi_peptide"] | None = Field(
        "monomer",
        description=(
            "Server-side complexity filter for /peptides. Use 'monomer' to skip "
            "multimer parents with empty sequence. Set null to disable filtering."
        ),
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
    request_timeout_s: float = Field(
        30.0,
        description="Per-request timeout for DBAASP HTTP calls, in seconds.",
        gt=0,
    )
    user_agent: str = Field(
        "LAMP/dbaasp-ingest",
        description="User-Agent header sent with DBAASP requests for attribution.",
    )
    source_version: str = Field(
        "v3 (API 4.0.1)",
        description="DBAASP version string echoed in CLI output for traceability.",
    )
    snapshot_date: str | None = Field(
        None,
        description="Optional ISO date (YYYY-MM-DD) marking when the snapshot was taken.",
    )

    class Config:
        extra = Extra.allow


def create_sequence_item(item: dict[str, Any]) -> SequenceItem:
    """Create a standardized DBAASP sequence item."""
    return {
        "dbaasp_id": item["dbaaspId"],
        "sequence": item["sequence"],
    }


def _has_only_allowed_amino_acids(sequence: str, allowed_amino_acids: frozenset[str]) -> bool:
    return set(sequence.upper()).issubset(allowed_amino_acids)


def _build_peptides_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{DBAASP_PEPTIDES_PATH}"


def _fetch_page(
    *,
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _stream_dbaasp(
    cfg: PrepareDBAASPConfig,
) -> Generator[SequenceItem, None, None]:
    """Paginate DBAASP /peptides and yield filtered {dbaasp_id, sequence} items."""
    allowed_amino_acids: frozenset[str] | None = None
    if cfg.allowed_amino_acids is not None:
        normalized = cfg.allowed_amino_acids.strip().upper()
        if not normalized:
            raise ValueError("allowed_amino_acids must not be empty when provided.")
        allowed_amino_acids = frozenset(normalized)

    url = _build_peptides_url(cfg.base_url)
    base_params: dict[str, Any] = {}
    if cfg.complexity is not None:
        base_params["complexity.value"] = cfg.complexity

    yielded = 0
    offset = 0
    total_count: int | None = None
    progress: tqdm | None = None
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": cfg.user_agent, "Accept": "application/json"})
            while True:
                params = {**base_params, "limit": cfg.page_size, "offset": offset}
                payload = _fetch_page(
                    session=session,
                    url=url,
                    params=params,
                    timeout=cfg.request_timeout_s,
                )
                if total_count is None:
                    total_count = int(payload.get("totalCount") or 0)
                    progress = tqdm(
                        total=total_count,
                        desc="Fetching DBAASP peptides",
                        unit="rec",
                    )

                data: list[dict[str, Any]] = payload.get("data") or []
                if not data:
                    break

                for item in data:
                    if progress is not None:
                        progress.update(1)

                    dbaasp_id = item.get("dbaaspId")
                    sequence = item.get("sequence")
                    if not dbaasp_id or not sequence:
                        continue
                    if cfg.max_length is not None and len(sequence) > cfg.max_length:
                        continue
                    if allowed_amino_acids is not None and not _has_only_allowed_amino_acids(
                        sequence, allowed_amino_acids
                    ):
                        continue

                    yield create_sequence_item(item)
                    yielded += 1
                    if cfg.max_sequences and yielded >= cfg.max_sequences:
                        return

                if len(data) < cfg.page_size:
                    break
                offset += cfg.page_size
    finally:
        if progress is not None:
            progress.close()


def prepare_dbaasp_sequences(
    cfg: PrepareDBAASPConfig,
) -> Generator[SequenceItem, None, None]:
    """Public entry returning a generator of standardized DBAASP items."""
    return _stream_dbaasp(cfg)


def prepare_and_upload_dbaasp_command(
    *,
    config_file: Path = CONFIG_FILE_OPTION,
) -> None:
    """Ingest DBAASP peptides via /peptides and upload as a HF subset."""
    raw = load_config_file(config_file)

    prepare_raw = raw.get("prepare", {})
    if not prepare_raw:
        typer.echo("❌ Config must include 'prepare' section.", err=True)
        raise typer.Exit(1)
    prepare_cfg = PrepareDBAASPConfig(**prepare_raw)

    upload_raw = raw.get("upload", {})
    if not upload_raw:
        typer.echo("❌ Config must include 'upload' section.", err=True)
        raise typer.Exit(1)
    if not upload_raw.get("token"):
        upload_raw["token"] = os.getenv("HF_TOKEN")
    upload_cfg = UploadConfig(**upload_raw)

    snapshot_suffix = (
        f", snapshot {prepare_cfg.snapshot_date}" if prepare_cfg.snapshot_date else ""
    )
    typer.echo(
        f"🚀 Preparing DBAASP peptides {prepare_cfg.source_version}{snapshot_suffix} "
        f"(complexity={prepare_cfg.complexity or 'any'}, "
        f"max_length={prepare_cfg.max_length}, "
        f"AA filter={'on' if prepare_cfg.allowed_amino_acids else 'off'})"
    )

    streams = {"train": prepare_dbaasp_sequences(prepare_cfg)}

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
        features=DBAASP_UPLOAD_FEATURES,
    )
    typer.echo(
        f"📦 Uploaded subset to "
        f"https://huggingface.co/datasets/{upload_cfg.repo_id}/{upload_cfg.subset_name}"
    )


def _demo() -> None:
    cfg = PrepareDBAASPConfig(max_sequences=5)
    stream = prepare_dbaasp_sequences(cfg)
    print("📊 Sample from DBAASP (first 5 monomer items):")
    for idx, seq in enumerate(stream, start=1):
        print(f"  {idx}. {seq['dbaasp_id']} ({len(seq['sequence'])} AA) {seq['sequence']}")


if __name__ == "__main__":
    _demo()
