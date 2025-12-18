"""Loss utilities for MetaModule configs."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sequence_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: Optional[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy for sequence logits shaped like [batch, seq_len, vocab].

    PyTorch's ``torch.nn.functional.cross_entropy`` expects logits as [N, C, ...].
    Many sequence models emit logits as [batch, seq_len, vocab], so this helper
    permutes to [batch, vocab, seq_len] before calling into PyTorch.
    """
    if input.ndim == 3:
        input = input.permute(0, 2, 1).contiguous()

    if ignore_index is None:
        return F.cross_entropy(input, target, reduction=reduction)

    return F.cross_entropy(input, target, reduction=reduction, ignore_index=ignore_index)

