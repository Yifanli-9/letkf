"""Python implementation of the LETKF core routine."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["letkf_core"]


def letkf_core(
    hdxb: np.ndarray,
    rdiag: Sequence[float],
    rloc: Sequence[float],
    dep: Sequence[float],
    parm_infl: float,
    *,
    rdiag_wloc: bool = False,
    minfl: float | None = None,
    return_transm: bool = False,
    return_pao: bool = False,
) -> Dict[str, np.ndarray | float]:
    """Core LETKF update.

    Parameters
    ----------
    hdxb:
        Observation operator applied to forecast perturbations with shape
        ``(nobs, ne)``.
    rdiag:
        Observation error variance.
    rloc:
        Localisation weights.
    dep:
        Observation departures ``yo - H xb``.
    parm_infl:
        Background covariance inflation parameter; the updated value is returned
        in the result dictionary under the ``"parm_infl"`` key.
    rdiag_wloc:
        Indicates whether ``rdiag`` already includes the localisation weights.
    minfl:
        Optional lower bound for ``parm_infl``.

    Returns
    -------
    dict
        Contains the transformation matrix ``trans`` and the updated
        ``parm_infl``.  When the optional flags are set the dictionary also
        includes ``transm`` and/or ``pao``.
    """

    hdxb = np.asarray(hdxb, dtype=float)
    dep = np.asarray(dep, dtype=float)
    rdiag = np.asarray(rdiag, dtype=float)
    rloc = np.asarray(rloc, dtype=float)

    nobsl, ne = hdxb.shape
    result: Dict[str, np.ndarray | float] = {}

    if nobsl == 0:
        trans = np.zeros((ne, ne), dtype=float)
        np.fill_diagonal(trans, np.sqrt(parm_infl))
        result["trans"] = trans
        result["parm_infl"] = parm_infl
        if return_transm:
            result["transm"] = np.zeros(ne, dtype=float)
        if return_pao:
            pao = np.zeros((ne, ne), dtype=float)
            if ne > 1:
                np.fill_diagonal(pao, parm_infl / float(ne - 1))
            result["pao"] = pao
        return result

    dep = dep[:nobsl]
    rdiag = rdiag[:nobsl]
    rloc = rloc[:nobsl]

    if minfl is not None and parm_infl < minfl:
        parm_infl = float(minfl)

    if rdiag_wloc:
        hdxb_rinv = hdxb / rdiag[:, None]
    else:
        hdxb_rinv = hdxb / rdiag[:, None] * rloc[:, None]

    work1 = hdxb_rinv.T @ hdxb
    rho = 1.0 / parm_infl
    work1 += np.eye(ne) * ((ne - 1) * rho)

    eigvals, eigvecs = np.linalg.eigh(work1)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if eigvals[-1] <= 0:
        raise ValueError("All eigenvalues are non-positive")

    threshold = abs(eigvals[-1]) * np.sqrt(np.finfo(float).eps)
    positive = eigvals > threshold
    inv_eigvals = np.zeros_like(eigvals)
    inv_eigvals[positive] = 1.0 / eigvals[positive]

    pa = (eigvecs * inv_eigvals) @ eigvecs.T
    work2 = pa @ hdxb_rinv.T
    work3 = work2 @ dep

    sqrt_factor = np.zeros_like(eigvals)
    sqrt_factor[positive] = np.sqrt((ne - 1) / eigvals[positive])
    trans = (eigvecs * sqrt_factor) @ eigvecs.T

    if return_transm:
        result["transm"] = work3.copy()
    else:
        trans = trans + work3[:, None]

    result["trans"] = trans
    if return_pao:
        result["pao"] = pa

    if rdiag_wloc:
        parm1 = float(np.sum(dep * dep / rdiag))
    else:
        parm1 = float(np.sum(dep * dep / rdiag * rloc))
    parm2 = float(np.sum(hdxb_rinv * hdxb) / float(ne - 1))
    parm3 = float(np.sum(rloc))
    if parm2 != 0.0 and parm3 != 0.0:
        parm4 = (parm1 - parm3) / parm2 - parm_infl
        sigma_o = 2.0 / parm3 * ((parm_infl * parm2 + parm3) / parm2) ** 2
        sigma_b = 0.04
        gain = sigma_b**2 / (sigma_o + sigma_b**2)
        parm_infl = parm_infl + gain * parm4

    result["parm_infl"] = parm_infl
    return result
