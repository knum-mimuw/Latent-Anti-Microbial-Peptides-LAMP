"""Backward-compatible publish module entrypoint."""

from __future__ import annotations

from ._zenml_sqlalchemy_uuid_compat import apply as _apply_zenml_uuid_compat

_apply_zenml_uuid_compat()

from .training.publish import main, publish_hf_pipeline

__all__ = ["publish_hf_pipeline", "main"]


if __name__ == "__main__":
    main()
