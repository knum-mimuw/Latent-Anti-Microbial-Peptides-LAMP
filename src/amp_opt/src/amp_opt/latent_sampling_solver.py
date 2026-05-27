"""Elite/history-guided solver: reparameterized VAE latent sampling around prototypes."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable

import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox

from amp_opt.step_by_step_solver import StepByStepSolver


class ProteinLatentSampling(StepByStepSolver):
    """Sample analogues via reparameterized posterior: z = mean + tau * exp(log_std) * eps.

    Implements the HydrAMP creativity parameter tau (Szymczak et al., Nature Comms 2023).
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        encode_dist: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        decode_ids: Callable[[torch.Tensor], torch.Tensor],
        sequence_length: int,
        vocab_size: int,
        *,
        tau: float = 1.0,
        top_k: int = 1,
        greedy: bool = True,
        batch_size: int = 1,
        alphabet: list[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        tokenize_row: Callable[[np.ndarray], torch.Tensor] | None = None,
        decode_row: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        if (tokenize_row is None) ^ (decode_row is None):
            raise ValueError("tokenize_row and decode_row must both be set or both be None.")

        if x0.ndim == 1:
            if tokenizer is None:
                warnings.warn(
                    "ProteinLatentSampling: 1D x0 without tokenizer; "
                    "tokenizing with list(x_i) per row.",
                    stacklevel=2,
                )

                def tokenizer(x: str) -> list[str]:  # type: ignore[no-redef]
                    return list(x)

            tok = tokenizer
            x0_ = [tok(str(x_i)) for x_i in x0]
            x0 = np.array(x0_)

        bb_row_len = int(x0.shape[1])
        super().__init__(black_box, x0, y0)

        self._bb_row_len = bb_row_len
        self._encode_dist = encode_dist
        self._decode_ids = decode_ids
        self._seq_len = int(sequence_length)
        self._vocab_size = int(vocab_size)
        self.tau = float(tau)
        self.top_k = int(top_k)
        self.greedy = bool(greedy)
        self.batch_size = int(batch_size)
        self._tokenize_row = tokenize_row
        self._decode_row = decode_row

        info_alpha = black_box.info.get_alphabet()
        self.alphabet = list(info_alpha) if alphabet is None else list(alphabet)
        if self._vocab_size < 2:
            raise ValueError("vocab_size must be >= 2.")
        if len(self.alphabet) < self._vocab_size:
            raise ValueError(
                f"alphabet length {len(self.alphabet)} < vocab_size {self._vocab_size}; "
                "cannot map vocab ids 0..vocab_size-1 to tokens."
            )

        self._char_to_id: dict[str, int] = {}
        for vid in range(self._vocab_size):
            sym = self.alphabet[vid]
            if sym != "":
                self._char_to_id[str(sym)] = vid

        self._token_for_id: list[str] = [str(self.alphabet[i]) for i in range(self._vocab_size)]

    def _pick_parent_row(self) -> np.ndarray:
        if self.greedy:
            best_xs = self.get_best_solution(top_k=self.top_k)
            pick = random.randrange(best_xs.shape[0])
            return best_xs[pick].reshape(-1)
        xs, _ = self.get_history_as_arrays()
        random_indices = np.random.permutation(len(xs))
        pool = xs[random_indices[: self.top_k]]
        pick = random.randrange(pool.shape[0])
        return pool[pick].reshape(-1)

    def _tokens_to_input_ids(self, row: np.ndarray) -> torch.Tensor:
        flat = row.reshape(-1)
        if flat.size != self._seq_len:
            raise ValueError(
                f"Parent row length {flat.size} != sequence_length {self._seq_len}; "
                "supply tokenize_row/decode_row when black-box length differs from model length."
            )
        ids = np.empty(self._seq_len, dtype=np.int64)
        for i in range(self._seq_len):
            tok = flat[i]
            if row.dtype.kind in ("U", "S", "O"):
                key = str(tok)
            elif row.dtype.kind in ("i", "f"):
                vi = int(tok)
                if vi < 0 or vi >= self._vocab_size:
                    raise ValueError(f"Token id {vi} at column {i} out of vocab range.")
                ids[i] = vi
                continue
            else:
                raise ValueError(
                    f"Unsupported sequence dtype {row.dtype}; use string or int token rows."
                )
            if key not in self._char_to_id:
                raise ValueError(
                    f"Token {key!r} at position {i} has no vocab id in 0..{self._vocab_size - 1} "
                    f"under the solver alphabet / vocab_size convention."
                )
            ids[i] = self._char_to_id[key]
        return torch.from_numpy(ids).unsqueeze(0).long()

    def _bb_row_to_model_ids(self, row: np.ndarray) -> torch.Tensor:
        if self._tokenize_row is not None:
            t = self._tokenize_row(row)
            if not isinstance(t, torch.Tensor):
                raise TypeError("tokenize_row must return a torch.Tensor.")
            if tuple(t.shape) != (1, self._seq_len):
                raise ValueError(
                    f"tokenize_row must return shape (1, sequence_length); got {tuple(t.shape)}, "
                    f"expected (1, {self._seq_len})."
                )
            return t.long()
        return self._tokens_to_input_ids(row)

    def _model_ids_to_bb_row(self, ids_1d: np.ndarray) -> np.ndarray:
        vec = np.asarray(ids_1d, dtype=np.int64).reshape(-1)
        if self._decode_row is not None:
            out = self._decode_row(vec)
            out = np.asarray(out, dtype=object).reshape(-1)
            if out.size != self._bb_row_len:
                raise ValueError(
                    f"decode_row returned length {out.size}, expected black-box row length "
                    f"{self._bb_row_len}."
                )
            return out
        if vec.size != self._seq_len:
            raise ValueError(
                f"Model id length {vec.size} != sequence_length {self._seq_len}; "
                "use decode_row when lengths differ."
            )
        tokens = [self._token_for_id[int(vec[i])] for i in range(vec.size)]
        return np.array(tokens, dtype=object)

    def _next_candidate(self) -> np.ndarray:
        parent = self._pick_parent_row()
        input_ids = self._bb_row_to_model_ids(parent)

        mean, log_std = self._encode_dist(input_ids)
        z = mean + self.tau * torch.exp(log_std) * torch.randn_like(mean)

        token_ids = self._decode_ids(z)
        ids_np = token_ids.squeeze(0).detach().cpu().numpy().astype(np.int64)
        out_row = self._model_ids_to_bb_row(ids_np)
        return out_row.reshape(1, -1)

    def next_candidate(self) -> np.ndarray:
        batch = [self._next_candidate().reshape(1, -1) for _ in range(self.batch_size)]
        return np.concatenate(batch, axis=0)
