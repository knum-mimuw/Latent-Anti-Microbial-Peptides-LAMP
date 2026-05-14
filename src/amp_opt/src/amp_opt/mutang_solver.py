"""Elite/history-guided solver: uniform random mutations from Jacobian-derived proposals."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable

import numpy as np
import torch
from poli.core.abstract_black_box import AbstractBlackBox

from amp_opt.step_by_step_solver import StepByStepSolver
from pep_compass_jr.tangent_space_mutation_proposer import substitutions_batch_from_jacobian


def non_identity_mutation_pairs(
    substitutions: dict[int, tuple[int, ...]],
    current_ids: np.ndarray,
) -> list[tuple[int, int]]:
    """All ``(position, new_vocab_id)`` from ``substitutions`` excluding no-ops."""
    flat = np.asarray(current_ids, dtype=np.int64).reshape(-1)
    out: list[tuple[int, int]] = []
    for pos, toks in substitutions.items():
        p = int(pos)
        if p < 0 or p >= flat.size:
            raise ValueError(f"Substitution position {p} out of range for length {flat.size}.")
        cur = int(flat[p])
        for v in toks:
            vi = int(v)
            if vi != cur:
                out.append((p, vi))
    return out


class ProteinMutangUniformMutation(StepByStepSolver):
    """Parent selection as in :class:`ProteinRandomMutation`; Jacobian-guided mutations."""

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        encode: Callable[[torch.Tensor], torch.Tensor],
        jacobian_batch_fn: Callable[[torch.Tensor], torch.Tensor | np.ndarray],
        sequence_length: int,
        vocab_size: int,
        *,
        n_mutations: int = 1,
        top_k: int = 1,
        greedy: bool = True,
        batch_size: int = 1,
        alphabet: list[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        tokenize_row: Callable[[np.ndarray], torch.Tensor] | None = None,
        decode_row: Callable[[np.ndarray], np.ndarray] | None = None,
        direction_significance_threshold: float = 1e-3,
        min_number_of_directions: int = 5,
        token_threshold: float = 0.1,
    ):
        if int(n_mutations) < 1:
            raise ValueError("n_mutations must be >= 1.")
        if (tokenize_row is None) ^ (decode_row is None):
            raise ValueError("tokenize_row and decode_row must both be set or both be None.")

        if x0.ndim == 1:
            if tokenizer is None:
                warnings.warn(
                    "ProteinMutangUniformMutation: 1D x0 without tokenizer; "
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
        self.n_mutations = int(n_mutations)
        self._encode = encode
        self._jacobian_batch_fn = jacobian_batch_fn
        self._seq_len = int(sequence_length)
        self._vocab_size = int(vocab_size)
        self.top_k = int(top_k)
        self.greedy = bool(greedy)
        self.batch_size = int(batch_size)
        self._dir_thresh = float(direction_significance_threshold)
        self._min_dirs = int(min_number_of_directions)
        self._token_thresh = float(token_threshold)
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
            key: str
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

    def _ids_to_bb_row_default(self, ids_1d: np.ndarray) -> np.ndarray:
        tokens = [self._token_for_id[int(ids_1d[i])] for i in range(ids_1d.size)]
        return np.array(tokens, dtype=object)

    def _bb_row_to_model_ids(self, row: np.ndarray) -> np.ndarray:
        if self._tokenize_row is not None:
            t = self._tokenize_row(row)
            if not isinstance(t, torch.Tensor):
                raise TypeError("tokenize_row must return a torch.Tensor.")
            if tuple(t.shape) != (1, self._seq_len):
                raise ValueError(
                    f"tokenize_row must return shape (1, sequence_length); got {tuple(t.shape)}, "
                    f"expected (1, {self._seq_len})."
                )
            return t.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)
        input_ids = self._tokens_to_input_ids(row)
        return input_ids.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)

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
        return self._ids_to_bb_row_default(vec)

    def _next_candidate(self) -> np.ndarray:
        parent = self._pick_parent_row()
        cur_ids = self._bb_row_to_model_ids(parent).copy()

        for step in range(self.n_mutations):
            input_ids = torch.from_numpy(cur_ids).unsqueeze(0).long()
            z = self._encode(input_ids)
            if z.ndim != 2 or z.shape[0] != 1:
                raise ValueError(
                    "encode must return shape [batch, latent] with batch=1 here; "
                    f"got {tuple(z.shape)}."
                )
            jac = self._jacobian_batch_fn(z.detach())
            subst_list = substitutions_batch_from_jacobian(
                jac,
                sequence_length=self._seq_len,
                vocab_size=self._vocab_size,
                direction_significance_threshold=self._dir_thresh,
                min_number_of_directions=self._min_dirs,
                token_threshold=self._token_thresh,
            )
            if len(subst_list) != 1:
                raise ValueError(
                    f"Expected one substitution dict for batch-1 parent; got {len(subst_list)}."
                )

            pairs = non_identity_mutation_pairs(subst_list[0], cur_ids)
            if not pairs:
                raise ValueError(
                    "Jacobian substitution proposal yielded no non-identity single mutations "
                    f"at mutation step {step}; loosen thresholds or check Jacobian / alignment."
                )

            pick = pairs[int(np.random.randint(0, len(pairs)))]
            pos, new_id = pick
            cur_ids[int(pos)] = int(new_id)

        out_row = self._model_ids_to_bb_row(cur_ids)
        return out_row.reshape(1, -1)

    def next_candidate(self) -> np.ndarray:
        batch = [self._next_candidate().reshape(1, -1) for _ in range(self.batch_size)]
        return np.concatenate(batch, axis=0)
