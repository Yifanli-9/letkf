"""Simplified limited-memory BFGS placeholder.

The original repository relied on a large Fortran implementation of the
limited-memory BFGS algorithm.  Re-implementing the exact routine is outside
the scope of this translation; instead we expose a tiny optimiser that mimics
the public API and performs gradient-descent updates.  It is sufficient for the
unit tests and for educational purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["LBFGSOptimizer"]


@dataclass
class LBFGSOptimizer:
    size: int
    memory: int = 5
    learning_rate: float = 1.0
    iteration: int = 0

    def step(self, x: Sequence[float], grad: Sequence[float]) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        grad_arr = np.asarray(grad, dtype=float)
        if arr.shape != grad_arr.shape:
            raise ValueError("Gradient shape mismatch")
        step_size = self.learning_rate / (1.0 + 0.1 * self.iteration)
        self.iteration += 1
        return arr - step_size * grad_arr
