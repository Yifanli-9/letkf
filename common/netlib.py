"""Thin wrappers that replace a subset of the original Netlib routines."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["rs", "dspfa", "dspdi"]


def _packed_to_full(ap: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            matrix[i, j] = matrix[j, i] = float(ap[idx])
            idx += 1
    return matrix


def _full_to_packed(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    packed = np.zeros(n * (n + 1) // 2, dtype=float)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            packed[idx] = a[i, j]
            idx += 1
    return packed


def rs(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of a symmetric matrix."""

    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square")
    w, v = np.linalg.eigh(arr)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


def dspfa(ap: np.ndarray, n: int):
    """Factor a symmetric matrix stored in packed form."""

    matrix = _packed_to_full(np.asarray(ap, dtype=float), int(n))
    try:
        np.linalg.cholesky(matrix)
        info = 0
    except np.linalg.LinAlgError:
        info = 1
    kpvt = np.arange(1, int(n) + 1)
    return matrix, kpvt, info


def dspdi(ap: np.ndarray, n: int, kpvt, det, inert, work, job) -> np.ndarray:
    """Compute the inverse of a symmetric matrix in packed form."""

    matrix = _packed_to_full(np.asarray(ap, dtype=float), int(n))
    inv = np.linalg.inv(matrix)
    return _full_to_packed(inv)
