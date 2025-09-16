"""Observation handling utilities."""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Tuple

import numpy as np

ID_U_OBS = 2819
ID_V_OBS = 2820
ID_T_OBS = 3073
ID_Q_OBS = 3330
ID_RH_OBS = 3331
ID_PS_OBS = 14593
ID_Z_OBS = 2567
ID_S_OBS = 3332
ID_RAIN_OBS = 9999

nobs = 0

__all__ = [
    "ID_U_OBS",
    "ID_V_OBS",
    "ID_T_OBS",
    "ID_Q_OBS",
    "ID_RH_OBS",
    "ID_PS_OBS",
    "ID_Z_OBS",
    "ID_S_OBS",
    "ID_RAIN_OBS",
    "get_nobs",
    "read_obs",
]


def _prepare(records: Iterable[Iterable[float]]) -> np.ndarray:
    data = np.asarray(list(records), dtype=float)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError("Each record must contain six fields")
    return data


def get_nobs(records: Iterable[Iterable[float]]) -> Tuple[int, Mapping[int, int]]:
    """Count the number of observations and classify them by type."""

    global nobs
    data = _prepare(records)
    nobs = data.shape[0]

    counts: MutableMapping[int, int] = {
        ID_U_OBS: 0,
        ID_V_OBS: 0,
        ID_T_OBS: 0,
        ID_Q_OBS: 0,
        ID_RH_OBS: 0,
        ID_PS_OBS: 0,
        ID_Z_OBS: 0,
        ID_S_OBS: 0,
        ID_RAIN_OBS: 0,
    }
    for elem in data[:, 0].astype(int):
        if elem in counts:
            counts[elem] += 1

    return nobs, counts


def read_obs(records: Iterable[Iterable[float]]):
    """Return observation arrays with unit conversions applied."""

    data = _prepare(records)
    elem = data[:, 0].astype(int)
    rlon = data[:, 1].astype(float)
    rlat = data[:, 2].astype(float)
    rlev = data[:, 3].astype(float)
    odat = data[:, 4].astype(float)
    oerr = data[:, 5].astype(float)

    for idx, kind in enumerate(elem):
        if kind in {ID_U_OBS, ID_V_OBS, ID_T_OBS, ID_Q_OBS, ID_RH_OBS, ID_Z_OBS}:
            rlev[idx] *= 100.0
        if kind == ID_PS_OBS:
            odat[idx] *= 100.0
            oerr[idx] *= 100.0
        if kind == ID_RH_OBS:
            odat[idx] *= 0.01
            oerr[idx] *= 0.01

    return (
        elem.astype(float),
        rlon,
        rlat,
        rlev,
        odat,
        oerr,
    )
