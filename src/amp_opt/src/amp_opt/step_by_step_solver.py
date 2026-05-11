"""Minimal `StepByStepSolver` matching poli-baselines (without that package's extra deps)."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from typing import Self

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox


def seed_python_numpy_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class AbstractSolver:
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
    ):
        self.black_box = black_box
        self.x0 = x0
        self.y0 = y0


class StepByStepSolver(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
    ):
        super().__init__(black_box, x0, y0)
        if x0 is not None:
            assert y0 is not None, "If x0 is given, y0 must be given as well."
            self.history = {
                "x": [x0_i.reshape(1, -1) for x0_i in x0],
                "y": [y0_i.reshape(1, -1) for y0_i in y0],
            }
        else:
            self.history = {"x": [], "y": []}

        self.iteration = 0

    def next_candidate(self) -> np.ndarray:
        raise NotImplementedError

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        self.history["x"] += [x_i.reshape(1, -1) for x_i in x]
        self.history["y"] += [y_i.reshape(1, -1) for y_i in y]

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.next_candidate()
        y = self.black_box(x)
        self.update(x, y)
        self.post_update(x, y)
        self.iteration += 1
        return x, y

    def solve(
        self,
        max_iter: int = 100,
        n_initial_points: int = 0,
        seed: int | None = None,
        break_at_performance: float | None = None,
        verbose: bool = False,
        pre_step_callbacks: Iterable[Callable[[Self], None]] | None = None,
        post_step_callbacks: Iterable[Callable[[Self], None]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            seed_python_numpy_torch(seed)

        for i in range(max_iter):
            if pre_step_callbacks is not None:
                for callback in pre_step_callbacks:
                    callback(self)

            _, y = self.step()

            if post_step_callbacks is not None:
                for callback in post_step_callbacks:
                    callback(self)

            if verbose:
                print(f"Iteration {i}: {y}, best so far: {self.get_best_performance()}")

            if break_at_performance is not None and y >= break_at_performance:
                break

        return self.get_best_solution(), self.get_best_performance()

    def get_best_solution(self, top_k: int = 1) -> np.ndarray:
        inputs = [x for x in self.history["x"]]
        outputs = [y for y in self.history["y"]]
        stacked_inputs = np.vstack(inputs)
        stacked_outputs = np.vstack(outputs)

        if stacked_outputs.shape[1] != 1:
            raise NotImplementedError("Only single-objective histories are supported.")

        _top_k = min(top_k, stacked_outputs.shape[0])
        order = np.argsort(stacked_outputs.flatten())
        best_solutions = stacked_inputs[order[-_top_k:]]
        return best_solutions.reshape(_top_k, -1)

    def get_best_performance(self, until: int | None = None) -> np.ndarray:
        outputs = [y for y in self.history["y"]]
        if until is not None:
            outputs = outputs[:until]
        stacked_outputs = np.vstack(outputs)
        return np.nanmax(stacked_outputs, axis=0)

    def get_history_as_arrays(
        self, penalize_nans_with: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(self.history["x"], axis=0)
        y = np.concatenate(self.history["y"], axis=0)
        if penalize_nans_with is not None:
            y = y.copy()
            y[np.isnan(y)] = penalize_nans_with
        return x, y
