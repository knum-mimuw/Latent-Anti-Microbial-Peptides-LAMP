"""Small callables for ``solver.factory_import_path`` (see ``run.run_with_config``).

Each factory has signature ``(black_box, x0, y0, kwargs: dict) -> StepByStepSolver``.
Add a new ``def my_factory(...)`` here (or in another module) and point YAML at
``module:callable``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from poli.core.abstract_black_box import AbstractBlackBox

from amp_opt.random_mutation_solver import ProteinRandomMutation
from amp_opt.step_by_step_solver import StepByStepSolver


def protein_random_mutation(
    black_box: AbstractBlackBox,
    x0: np.ndarray,
    y0: np.ndarray,
    kwargs: dict[str, Any],
) -> StepByStepSolver:
    return ProteinRandomMutation(black_box, x0, y0, **kwargs)
