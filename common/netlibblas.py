"""Subset of BLAS level-1 routines used by the translated code."""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["daxpy", "dcopy", "ddot", "dswap", "idamax"]


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def daxpy(da: float, dx: Sequence[float], dy: Sequence[float]) -> np.ndarray:
    """Compute ``dy = da * dx + dy``."""

    dx_arr = _as_array(dx)
    dy_arr = _as_array(dy)
    if dx_arr.shape != dy_arr.shape:
        raise ValueError("daxpy requires arrays of the same shape")
    return da * dx_arr + dy_arr


def dcopy(dx: Sequence[float]) -> np.ndarray:
    """Return a copy of the input vector."""

    return _as_array(dx).copy()


def ddot(dx: Sequence[float], dy: Sequence[float]) -> float:
    """Dot product between two vectors."""

    dx_arr = _as_array(dx)
    dy_arr = _as_array(dy)
    if dx_arr.shape != dy_arr.shape:
        raise ValueError("ddot requires arrays of the same shape")
    return float(np.dot(dx_arr, dy_arr))


def dswap(dx: Sequence[float], dy: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Swap two vectors and return the swapped versions."""

    dx_arr = _as_array(dx)
    dy_arr = _as_array(dy)
    if dx_arr.shape != dy_arr.shape:
        raise ValueError("dswap requires arrays of the same shape")
    return dy_arr.copy(), dx_arr.copy()


def idamax(dx: Sequence[float]) -> int:
    """Index of the element with the largest absolute value (1-based)."""

    dx_arr = _as_array(dx)
    return int(np.argmax(np.abs(dx_arr)) + 1)
