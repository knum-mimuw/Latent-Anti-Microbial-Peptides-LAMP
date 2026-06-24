"""YAML-loaded run configuration (Pydantic models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from amp_opt.sequence_constants import MAX_PEPTIDE_LENGTH


def _one_colon_import_path(path: str, *, field_name: str) -> None:
    if path.count(":") != 1:
        raise ValueError(f"{field_name} must contain exactly one ':', got {path!r}.")
    mod, attr = path.split(":", 1)
    if not mod or not attr:
        raise ValueError(f"{field_name} must be 'module:callable', got {path!r}.")


class SeedSequencesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    id_column: str = "id"
    sequence_column: str = "sequence"


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: Path
    mode: Literal["write", "append"] = "write"


class ApexSubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Literal["default", "all"] = "default"
    device: str = "cpu"
    batch_size: int = 512


class ColumnsSubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["all", "panel", "explicit"] = "all"
    panel: str | None = None
    pathogen_names: list[str] | None = None

    @model_validator(mode="after")
    def _panel_explicit_rules(self) -> ColumnsSubConfig:
        if self.mode == "panel" and not self.panel:
            raise ValueError("columns.mode='panel' requires columns.panel.")
        if self.mode == "explicit":
            names = self.pathogen_names
            if not names:
                raise ValueError(
                    "columns.mode='explicit' requires non-empty columns.pathogen_names."
                )
        return self


class BlackBoxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apex: ApexSubConfig = Field(default_factory=ApexSubConfig)
    columns: ColumnsSubConfig = Field(default_factory=ColumnsSubConfig)
    aggregation: Literal["mean", "max", "min"] = "mean"
    mic_transform: Literal["none", "log2"] = "none"


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factory_import_path: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_factory_path(self) -> Self:
        _one_colon_import_path(self.factory_import_path, field_name="solver.factory_import_path")
        return self


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    random_seeds: list[int]
    max_total_evaluations: int | None = None
    max_iterations: int | None = None
    solver: SolverConfig

    @model_validator(mode="after")
    def _validate_random_seeds(self) -> OptimizationConfig:
        if not self.random_seeds:
            raise ValueError("optimization.random_seeds must be a non-empty list of integers.")
        if len(set(self.random_seeds)) != len(self.random_seeds):
            raise ValueError("optimization.random_seeds must not contain duplicates.")
        return self

    @model_validator(mode="after")
    def _require_iteration_or_eval_cap(self) -> OptimizationConfig:
        if self.max_iterations is None and self.max_total_evaluations is None:
            raise ValueError(
                "Set at least one of optimization.max_iterations or "
                "optimization.max_total_evaluations (both None would run forever)."
            )
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_sequences: SeedSequencesConfig
    output: OutputConfig
    black_box: BlackBoxConfig = Field(default_factory=BlackBoxConfig)
    optimization: OptimizationConfig


def load_run_config(path: Path | str) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping.")
    return RunConfig.model_validate(raw)


def validate_seed_sequence(seq: str) -> tuple[str, str | None]:
    """Return (normalized_sequence, error_message)."""
    s = "".join(str(seq).split()).upper()
    if not s:
        return s, "empty sequence after normalization"
    if len(s) > MAX_PEPTIDE_LENGTH:
        return s, f"length {len(s)} exceeds MAX_PEPTIDE_LENGTH={MAX_PEPTIDE_LENGTH}"
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    bad = sorted({c for c in s if c not in allowed})
    if bad:
        return s, f"non-standard residues: {''.join(bad)}"
    return s, None
