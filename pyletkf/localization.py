"""Localization utilities."""

from __future__ import annotations

import numpy as np

__all__ = ["gaspari_cohn"]


def gaspari_cohn(distances: np.ndarray, radius: float) -> np.ndarray:
    """Return Gaspari--Cohn taper weights for *distances* and localisation *radius*.

    The implementation follows Gaspari & Cohn (1999).  Distances greater than
    twice the localisation radius receive zero weight.
    """

    if radius <= 0.0:
        raise ValueError("The localisation radius must be positive")

    dist = np.asarray(distances, dtype=float)
    x = dist / float(radius)

    weights = np.zeros_like(x)

    mask1 = x <= 1.0
    if np.any(mask1):
        x1 = x[mask1]
        weights[mask1] = (((-0.25 * x1 + 0.5) * x1 + 0.625) * x1 - 5.0 / 3.0) * x1**2 + 1.0

    mask2 = (x > 1.0) & (x < 2.0)
    if np.any(mask2):
        x2 = x[mask2]
        weights[mask2] = ((((0.125 * x2 - 0.5) * x2 + 0.625) * x2 + 5.0 / 3.0) * x2 - 5.0) * x2 + 4.0
        weights[mask2] -= 2.0 / (3.0 * x2)

    weights[x >= 2.0] = 0.0
    np.maximum(weights, 0.0, out=weights)
    return weights
