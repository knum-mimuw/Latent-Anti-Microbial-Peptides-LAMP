"""Elite/history-guided random substitution baseline (POLi RandomMutation analogue)."""

from __future__ import annotations

import random
import warnings
from collections.abc import Callable

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

from amp_opt.sequence_constants import STANDARD_AA_ORDER
from amp_opt.step_by_step_solver import StepByStepSolver


class ProteinRandomMutation(StepByStepSolver):
    """Random point mutations sampled from elites (greedy) or shuffled history."""

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        n_mutations: int = 1,
        top_k: int = 1,
        batch_size: int = 1,
        greedy: bool = True,
        alphabet: list[str] | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        mutation_token_probs: list[float] | None = None,
    ):
        if x0.ndim == 1:
            if tokenizer is None:
                warnings.warn(
                    "ProteinRandomMutation: 1D x0 without tokenizer; "
                    "tokenizing with list(x_i) per row.",
                    stacklevel=2,
                )

                def tokenizer(x: str) -> list[str]:  # type: ignore[no-redef]
                    return list(x)

            tok = tokenizer
            x0_ = [tok(str(x_i)) for x_i in x0]
            x0 = np.array(x0_)

        super().__init__(black_box, x0, y0)
        info_alphabet = black_box.info.get_alphabet()
        self.alphabet = list(info_alphabet) if alphabet is None else list(alphabet)
        self.alphabet_without_empty = [s for s in self.alphabet if s != ""]
        self.alphabet_size = len(self.alphabet)
        self.n_mutations = int(n_mutations)
        self.top_k = int(top_k)
        self.greedy = bool(greedy)
        self.batch_size = int(batch_size)

        if mutation_token_probs is None:
            self._replacement_choices = np.array(self.alphabet_without_empty)
            self._replacement_p: np.ndarray | None = None
        else:
            if len(mutation_token_probs) != len(STANDARD_AA_ORDER):
                raise ValueError(
                    f"mutation_token_probs must have length {len(STANDARD_AA_ORDER)} "
                    f"(STANDARD_AA_ORDER), got {len(mutation_token_probs)}."
                )
            probs = np.asarray(mutation_token_probs, dtype=np.float64)
            if np.any(probs < 0) or probs.sum() <= 0:
                raise ValueError("mutation_token_probs must be non-negative and sum to > 0.")
            probs = probs / probs.sum()
            order = list(STANDARD_AA_ORDER)
            self._replacement_choices = np.array(order)
            self._replacement_p = probs

    def _pick_residue(self) -> str:
        if self._replacement_p is None:
            return str(np.random.choice(self.alphabet_without_empty))
        idx = int(np.random.choice(len(self._replacement_choices), p=self._replacement_p))
        return str(self._replacement_choices[idx])

    def _next_candidate(self) -> np.ndarray:
        if self.greedy:
            best_xs = self.get_best_solution(top_k=self.top_k)
            pick = random.randrange(best_xs.shape[0])
            best_x = best_xs[pick]
        else:
            xs, _ = self.get_history_as_arrays()
            random_indices = np.random.permutation(len(xs))
            pool = xs[random_indices[: self.top_k]]
            pick = random.randrange(pool.shape[0])
            best_x = pool[pick]

        next_x = best_x.copy().reshape(1, -1)

        for _ in range(self.n_mutations):
            flat = next_x.reshape(-1)
            pos = int(np.random.randint(0, flat.size))
            while flat[pos] == "":
                pos = int(np.random.randint(0, flat.size))

            if next_x.dtype.kind in ("i", "f"):
                mutant = int(np.random.randint(0, self.alphabet_size))
            elif next_x.dtype.kind in ("U", "S", "O"):
                mutant = self._pick_residue()
            else:
                raise ValueError(
                    f"Unsupported dtype for sequence tokens: {next_x.dtype}. "
                    "Use unicode string tokens for protein sequences."
                )

            next_x.reshape(-1)[pos] = mutant

        return next_x

    def next_candidate(self) -> np.ndarray:
        mutations = [
            self._next_candidate().reshape(1, -1) for _ in range(self.batch_size)
        ]
        return np.concatenate(mutations, axis=0)
