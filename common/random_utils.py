"""Random number helpers used by the translated ``common`` utilities.

The original Fortran implementation shipped multiple Mersenne Twister
variations.  Re-implementing those bit-level algorithms in Python would make
the code harder to maintain without providing tangible benefits, therefore this
module exposes a light-weight wrapper around :mod:`numpy.random`.  The API
resembles the routines that were historically provided by ``SFMT`` and
``mt19937ar`` so that higher level code can remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class _RandomState:
    """Container holding the shared random number generator."""

    rng: np.random.Generator = np.random.default_rng()

    def reseed(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(None if seed is None else int(seed))


_STATE = _RandomState()


def init_gen_rand(seed: Optional[int] = None) -> None:
    """Initialise the global generator.

    Parameters
    ----------
    seed:
        Optional seed used for reproducibility.  When *None* the generator is
        seeded from system entropy, mimicking the original behaviour where the
        seed was derived from the current time.
    """

    _STATE.reseed(seed)


def init_by_array(init_key: Sequence[int]) -> None:
    """Initialise the generator using an array of integers.

    The Fortran implementation folded the key into the internal state using a
    deterministic recurrence.  The numpy generator accepts a sequence directly,
    therefore we simply convert the key to a :class:`numpy.random.SeedSequence`.
    """

    seq = np.random.SeedSequence([int(v) for v in init_key])
    _STATE.rng = np.random.default_rng(seq)


def genrand_int32(size: Optional[int] = None) -> np.ndarray:
    """Return random 32-bit unsigned integers in ``[0, 2**32)``."""

    return _STATE.rng.integers(0, 2**32, size=size, dtype=np.uint32)


def genrand_int31(size: Optional[int] = None) -> np.ndarray:
    """Return random 31-bit signed integers in ``[0, 2**31)``."""

    return _STATE.rng.integers(0, 2**31, size=size, dtype=np.int64)


def genrand_real1(size: Optional[int] = None) -> np.ndarray:
    """Random real numbers in the closed interval ``[0, 1]``."""

    return np.asarray(_STATE.rng.random(size=size), dtype=float)


def genrand_real2(size: Optional[int] = None) -> np.ndarray:
    """Random real numbers in the half-open interval ``[0, 1)``."""

    return np.asarray(_STATE.rng.random(size=size), dtype=float)


def genrand_real3(size: Optional[int] = None) -> np.ndarray:
    """Random real numbers in the open interval ``(0, 1)``."""

    data = _STATE.rng.random(size=size)
    eps = np.finfo(float).eps
    return np.asarray(np.clip(data, eps, 1.0 - eps), dtype=float)


def genrand_res53(size: Optional[int] = None) -> np.ndarray:
    """Random real numbers with 53-bit resolution in ``[0, 1)``."""

    return np.asarray(_STATE.rng.random(size=size), dtype=float)


def rand(size: int) -> np.ndarray:
    """Return a vector of uniformly distributed random numbers."""

    return genrand_real2(size)


def randn(size: int) -> np.ndarray:
    """Return a vector of normally distributed random numbers."""

    return np.asarray(_STATE.rng.standard_normal(size=size), dtype=float)


__all__ = [
    "init_gen_rand",
    "init_by_array",
    "genrand_int31",
    "genrand_int32",
    "genrand_real1",
    "genrand_real2",
    "genrand_real3",
    "genrand_res53",
    "rand",
    "randn",
]
