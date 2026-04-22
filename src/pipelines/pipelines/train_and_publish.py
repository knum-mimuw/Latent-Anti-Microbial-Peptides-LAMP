"""Backward-compatible train-and-publish module entrypoint."""

from __future__ import annotations

from ._zenml_sqlalchemy_uuid_compat import apply as _apply_zenml_uuid_compat

_apply_zenml_uuid_compat()

from .training.train_and_publish import main, train_and_optional_publish_pipeline

__all__ = ["train_and_optional_publish_pipeline", "main"]


if __name__ == "__main__":
    main()
