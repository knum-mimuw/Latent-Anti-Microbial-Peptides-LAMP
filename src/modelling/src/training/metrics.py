"""Optional :func:`compute_metrics` hooks for :class:`transformers.Trainer`."""

from __future__ import annotations


def empty_compute_metrics(eval_pred):  # noqa: ANN001
    """Returns ``{}``. For :class:`~transformers.Trainer`, use ``compute_metrics=None`` instead so eval does not gather logits (variable seq lengths)."""
    del eval_pred
    return {}
