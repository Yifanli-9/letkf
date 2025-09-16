"""Matrix utilities translated from the original Fortran code."""

from __future__ import annotations

import numpy as np

__all__ = ["mtx_eigen", "mtx_inv", "mtx_sqrt", "mtx_inv_rg"]


def _ensure_matrix(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Expected a square matrix")
    return arr


def mtx_eigen(a: np.ndarray, imode: int = 1):
    """Eigen-decomposition of a symmetric matrix.

    Parameters mirror the Fortran routine, with *imode* retained for
    compatibility although it is not used directly—the eigenvectors are always
    returned.
    """

    arr = _ensure_matrix(a)
    w, v = np.linalg.eigh(arr)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]

    if w[-1] <= 0:
        raise ValueError("All eigenvalues are non-positive")

    threshold = abs(w[-1]) * np.sqrt(np.finfo(float).eps)
    positive = w > threshold
    nrank_eff = int(np.sum(positive))
    w = np.where(positive, w, 0.0)
    v = v.copy()
    v[:, ~positive] = 0.0
    return w, v, nrank_eff


def mtx_inv(a: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric matrix."""

    arr = _ensure_matrix(a)
    return np.linalg.inv(arr)


def mtx_sqrt(a: np.ndarray) -> np.ndarray:
    """Matrix square root using the eigen-decomposition."""

    arr = _ensure_matrix(a)
    w, v, _ = mtx_eigen(arr)
    sqrt_w = np.sqrt(np.clip(w, 0.0, None))
    return (v * sqrt_w) @ v.T


def mtx_inv_rg(a: np.ndarray) -> np.ndarray:
    """General matrix inversion."""

    return mtx_inv(a)
