"""Riemannian Brownian motion random walk in VAE latent space.

Each solver step advances a latent position by one Euler-Maruyama increment,
sampling z_{t+1} ~ N(z_t, sigma^2 * G(z_t)^{-1}) where G = J^T J is the
pullback metric from the decoder Jacobian (Kalatzis et al., 2020).
"""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox

from amp_opt.step_by_step_solver import StepByStepSolver


class ProteinBrownianMotion(StepByStepSolver):
    """Riemannian Brownian motion walk in the decoder latent space."""

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        encode: Callable[[torch.Tensor], torch.Tensor],
        decode_ids: Callable[[torch.Tensor], torch.Tensor],
        jacobian_batch_fn: Callable[[torch.Tensor], torch.Tensor | np.ndarray],
        sequence_length: int,
        vocab_size: int,
        *,
        sigma: float = 0.1,
        direction_significance_threshold: float = 0.1,
        min_number_of_directions: int = 5,
        max_step_norm: float | None = None,
        noise_distribution: Literal["gaussian", "sphere"] = "gaussian",
        scaling_mode: Literal["riemannian", "projected", "metric"] = "projected",
        second_order: bool = False,
        field_derivative_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
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
                    "ProteinBrownianMotion: 1D x0 without tokenizer; "
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
        self._encode = encode
        self._decode_ids = decode_ids
        self._jacobian_batch_fn = jacobian_batch_fn
        self._seq_len = int(sequence_length)
        self._vocab_size = int(vocab_size)
        self.sigma = float(sigma)
        self._dir_thresh = float(direction_significance_threshold)
        self._min_dirs = int(min_number_of_directions)
        self._max_step_norm = float(max_step_norm) if max_step_norm is not None else None
        self._noise_dist = noise_distribution
        self._scaling_mode = scaling_mode
        self._second_order = bool(second_order)
        self._field_derivative_fn = field_derivative_fn
        if self._second_order and self._field_derivative_fn is None:
            raise ValueError("second_order=True requires field_derivative_fn to be set.")
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

        self._z: torch.Tensor | None = None
        self._best_y: float = float("-inf")

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
        if self._z is None:
            parent = self._pick_parent_row()
            input_ids = self._bb_row_to_model_ids(parent)
            self._z = self._encode(input_ids)

        if self.sigma == 0.0:
            z_next = self._z
        else:
            jac = self._jacobian_batch_fn(self._z.detach())
            if isinstance(jac, np.ndarray):
                jac = torch.from_numpy(jac).to(self._z.device)
            J = jac.squeeze(0).to(dtype=self._z.dtype)
            U, s, Vh = torch.linalg.svd(J, full_matrices=False)
            n_above = int((s > self._dir_thresh).sum())
            n_dirs = min(max(n_above, self._min_dirs), s.shape[0])
            s_kept = s[:n_dirs]
            Vh_kept = Vh[:n_dirs, :]

            eps = torch.randn(n_dirs, device=self._z.device, dtype=self._z.dtype)
            if self._noise_dist == "sphere":
                norm = torch.linalg.norm(eps)
                if norm > 1e-8:
                    eps = eps / norm * (n_dirs**0.5)

            if self._scaling_mode == "riemannian":
                scaled = eps / s_kept
            elif self._scaling_mode == "projected":
                scaled = eps
            elif self._scaling_mode == "metric":
                scaled = eps * s_kept
            else:
                raise ValueError(f"Unknown scaling_mode: {self._scaling_mode!r}")

            delta = self.sigma * (Vh_kept.T @ scaled)

            if self._second_order:
                accel_ambient = self._field_derivative_fn(self._z, delta.unsqueeze(0))
                U_kept = U[:, :n_dirs]
                accel_latent = Vh_kept.T @ (
                    (1.0 / s_kept) * (U_kept.T @ accel_ambient.squeeze(0))
                )
                delta = delta - 0.5 * (self.sigma**2) * accel_latent

            if self._max_step_norm is not None:
                norm = torch.linalg.norm(delta)
                if norm > self._max_step_norm:
                    delta = delta * (self._max_step_norm / norm)

            z_next = (self._z.squeeze(0) + delta).unsqueeze(0)
            self._z = z_next

        token_ids = self._decode_ids(z_next)
        ids_np = token_ids.squeeze(0).detach().cpu().numpy().astype(np.int64)
        out_row = self._model_ids_to_bb_row(ids_np)
        return out_row.reshape(1, -1)

    def next_candidate(self) -> np.ndarray:
        batch = [self._next_candidate().reshape(1, -1) for _ in range(self.batch_size)]
        return np.concatenate(batch, axis=0)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        current_best = float(np.nanmax(y))
        if current_best > self._best_y:
            self._best_y = current_best
            self._z = None
