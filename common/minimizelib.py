"""Light-weight optimisation helpers replacing the Fortran L-BFGS driver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = ["initialize_minimizer", "minimize", "terminate_minimizer"]


@dataclass
class _State:
    size: int
    learning_rate: float = 1.0
    iteration: int = 0


_STATE: Optional[_State] = None


def initialize_minimizer(vsiz: int, learning_rate: float = 1.0) -> None:
    """Initialise the minimiser state."""

    global _STATE
    _STATE = _State(size=int(vsiz), learning_rate=float(learning_rate))


def minimize(
    xctl: Sequence[float], costf: float, costg: Sequence[float], maxiter: int, step: float | None = None
) -> int:
    """Perform a simple gradient-descent step.

    The function mirrors the call signature of the Fortran routine but applies a
    very small stochastic-gradient style update.  ``xctl`` is updated in-place
    when it is backed by a :class:`numpy.ndarray`.
    """

    global _STATE
    if _STATE is None:
        initialize_minimizer(len(xctl))

    arr = np.asarray(xctl, dtype=float)
    grad = np.asarray(costg, dtype=float)
    if arr.shape != grad.shape:
        raise ValueError("Control vector and gradient must have the same shape")

    state = _STATE
    step_size = step if step is not None else state.learning_rate / (1.0 + 0.1 * state.iteration)
    arr -= step_size * grad

    if isinstance(xctl, np.ndarray):
        xctl[...] = arr

    state.iteration += 1
    if state.iteration >= int(maxiter):
        return -int(maxiter)
    return 0


def terminate_minimizer() -> None:
    """Reset the minimiser state."""

    global _STATE
    _STATE = None
