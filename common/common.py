"""Translation of the historical Fortran ``common`` module to Python."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Iterable, Sequence, Tuple

import numpy as np

from .random_utils import init_gen_rand, rand, randn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI = math.pi
GG = 9.81
RD = 287.05
RV = 461.50
CP = 1005.7
HVAP = 2.5e6
FVIRT = RV / RD - 1.0
RE = 6_371_300.0
R_OMEGA = 7.292e-5
T0C = 273.15
UNDEF = -9.99e33
DEG2RAD = PI / 180.0
RAD2DEG = 180.0 / PI

__all__ = [
    "CP",
    "DEG2RAD",
    "FVIRT",
    "GG",
    "HVAP",
    "PI",
    "RAD2DEG",
    "RD",
    "RE",
    "RV",
    "T0C",
    "UNDEF",
    "com_anomcorrel",
    "com_datetime_reg",
    "com_distll",
    "com_distll_1",
    "com_filter_lanczos",
    "com_gamma",
    "com_interp_spline",
    "com_l2norm",
    "com_ll_arc_distance",
    "com_mean",
    "com_mdays",
    "com_pos2ij",
    "com_rand",
    "com_randn",
    "com_rms",
    "com_stdev",
    "com_time2ymdh",
    "com_timeinc_hr",
    "com_tai2utc",
    "com_utc2tai",
    "com_ymdh2time",
    "init_gen_rand",
]


def _ensure_array(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("Expected a one dimensional array of values")
    if data.size == 0:
        raise ValueError("Input array must not be empty")
    return data


def com_mean(var: Sequence[float]) -> float:
    """Return the arithmetic mean of *var*."""

    return float(np.mean(_ensure_array(var)))


def com_stdev(var: Sequence[float]) -> float:
    """Sample standard deviation of *var*."""

    data = _ensure_array(var)
    if data.size < 2:
        return 0.0
    return float(np.std(data, ddof=1))


def com_covar(var1: Sequence[float], var2: Sequence[float]) -> float:
    """Sample covariance between *var1* and *var2*."""

    a = _ensure_array(var1)
    b = _ensure_array(var2)
    if a.size != b.size:
        raise ValueError("Input arrays must share the same length")
    if a.size < 2:
        return 0.0
    return float(np.cov(a, b, ddof=1)[0, 1])


def com_correl(var1: Sequence[float], var2: Sequence[float]) -> float:
    """Pearson correlation coefficient between *var1* and *var2*."""

    std1 = com_stdev(var1)
    std2 = com_stdev(var2)
    if std1 == 0.0 or std2 == 0.0:
        return 0.0
    return com_covar(var1, var2) / std1 / std2


def com_anomcorrel(
    var1: Sequence[float], var2: Sequence[float], varmean: Sequence[float]
) -> float:
    """Anomaly correlation between two ensembles."""

    a = _ensure_array(var1)
    b = _ensure_array(var2)
    mean = _ensure_array(varmean)
    if a.size != b.size or a.size != mean.size:
        raise ValueError("Input arrays must share the same length")
    dev1 = a - mean
    dev2 = b - mean
    num = float(np.dot(dev1, dev2))
    den = float(np.sqrt(np.dot(dev1, dev1) * np.dot(dev2, dev2)))
    return num / den if den else 0.0


def com_l2norm(var: Sequence[float]) -> float:
    """Return the Euclidean norm of *var*."""

    return float(np.linalg.norm(_ensure_array(var)))


def com_rms(var: Sequence[float]) -> float:
    """Root mean square of *var*."""

    data = _ensure_array(var)
    return float(np.sqrt(np.mean(np.square(data))))


def com_filter_lanczos(var: Sequence[float], fc: float, lresol: int = 10) -> np.ndarray:
    """Lanczos low-pass filter with cyclic boundary conditions."""

    arr = np.asarray(var, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Lanczos filter expects a one dimensional array")
    if arr.size == 0:
        return arr.copy()

    idx = np.arange(-lresol, lresol + 1, dtype=float)
    weight = np.empty(idx.size, dtype=float)
    weight[idx == 0.0] = fc / PI
    nonzero = idx != 0.0
    rl = idx[nonzero]
    weight[nonzero] = (
        np.sin(fc * rl)
        * np.sin(PI * rl / float(lresol))
        * float(lresol)
        / (PI * PI * rl * rl)
    )

    pad_width = lresol
    padded = np.concatenate((arr[-pad_width:], arr, arr[:pad_width]))
    filtered = np.empty_like(arr)
    for i in range(arr.size):
        window = padded[i : i + weight.size]
        filtered[i] = float(np.dot(weight, window))

    if isinstance(var, np.ndarray) and var.shape == filtered.shape:
        var[...] = filtered
        return var
    return filtered


def com_rand(ndim: int) -> np.ndarray:
    """Uniform random numbers in ``[0, 1)``."""

    return rand(int(ndim))


def com_randn(ndim: int) -> np.ndarray:
    """Normal distributed random numbers with mean 0 and variance 1."""

    return randn(int(ndim))


def com_timeinc_hr(iy: int, im: int, iday: int, ih: int, incr: int) -> Tuple[int, int, int, int]:
    """Increase a timestamp by ``incr`` hours."""

    dt = datetime(iy, im, iday, ih) + timedelta(hours=int(incr))
    return dt.year, dt.month, dt.day, dt.hour


def com_time2ymdh(itime: int) -> Tuple[int, int, int, int]:
    """Convert a packed integer time representation to Y/M/D/H."""

    s = f"{int(itime):010d}"
    return int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10])


def com_ymdh2time(iy: int, im: int, iday: int, ih: int) -> int:
    """Pack Y/M/D/H into the integer representation used by the legacy code."""

    return int(iy) * 1_000_000 + int(im) * 10_000 + int(iday) * 100 + int(ih)


def com_distll(
    alon: Sequence[float],
    alat: Sequence[float],
    blon: Sequence[float],
    blat: Sequence[float],
) -> np.ndarray:
    """Great-circle distance between corresponding points in metres."""

    lon1 = np.deg2rad(np.asarray(alon, dtype=float))
    lon2 = np.deg2rad(np.asarray(blon, dtype=float))
    lat1 = np.deg2rad(np.asarray(alat, dtype=float))
    lat2 = np.deg2rad(np.asarray(blat, dtype=float))

    cosd = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    cosd = np.clip(cosd, -1.0, 1.0)
    return np.asarray(np.arccos(cosd) * RE, dtype=float)


def com_distll_1(alon: float, alat: float, blon: float, blat: float) -> float:
    """Scalar version of :func:`com_distll`."""

    return float(com_distll([alon], [alat], [blon], [blat])[0])


def com_interp_spline(
    x: Sequence[float], y: Sequence[float], n: int, x5: Sequence[float]
) -> np.ndarray:
    """Natural cubic spline interpolation."""

    grid_x = _ensure_array(x)
    grid_y = _ensure_array(y)
    targets = _ensure_array(x5)[:n]

    order = np.argsort(grid_x)
    grid_x = grid_x[order]
    grid_y = grid_y[order]

    if grid_x.size < 2:
        raise ValueError("At least two grid points are required")

    h = np.diff(grid_x)
    if np.any(h == 0):
        raise ValueError("Grid points must be strictly increasing")

    alpha = np.zeros(grid_x.size, dtype=float)
    for i in range(1, grid_x.size - 1):
        alpha[i] = (
            3.0 / h[i] * (grid_y[i + 1] - grid_y[i])
            - 3.0 / h[i - 1] * (grid_y[i] - grid_y[i - 1])
        )

    l = np.ones_like(grid_x)
    mu = np.zeros_like(grid_x)
    z = np.zeros_like(grid_x)
    for i in range(1, grid_x.size - 1):
        l[i] = 2.0 * (grid_x[i + 1] - grid_x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    c = np.zeros_like(grid_x)
    b = np.zeros(grid_x.size - 1, dtype=float)
    d = np.zeros(grid_x.size - 1, dtype=float)
    for j in range(grid_x.size - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (grid_y[j + 1] - grid_y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    result = np.empty_like(targets, dtype=float)
    for i, xv in enumerate(targets):
        idx = int(np.searchsorted(grid_x, xv) - 1)
        if idx < 0:
            idx = 0
        elif idx >= grid_x.size - 1:
            idx = grid_x.size - 2
        dx = xv - grid_x[idx]
        result[i] = grid_y[idx] + b[idx] * dx + c[idx] * dx * dx + d[idx] * dx * dx * dx

    return result


def com_pos2ij(
    msw: int,
    nx: int,
    ny: int,
    flon: Sequence[Sequence[float]],
    flat: Sequence[Sequence[float]],
    num_obs: int,
    olon: Sequence[float],
    olat: Sequence[float],
    detailout: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map observation positions to grid indices.

    Parameters mirror the Fortran routine.  The returned arrays contain the
    fractional grid coordinates for each observation.  Unresolved locations are
    encoded as ``NaN``.
    """

    grid_lon = np.asarray(flon, dtype=float)
    grid_lat = np.asarray(flat, dtype=float)
    if grid_lon.shape != (nx, ny) or grid_lat.shape != (nx, ny):
        raise ValueError("Grid dimensions do not match nx/ny")

    obs_lon = np.asarray(olon, dtype=float)[:num_obs]
    obs_lat = np.asarray(olat, dtype=float)[:num_obs]

    oi = np.full(num_obs, np.nan, dtype=float)
    oj = np.full(num_obs, np.nan, dtype=float)

    if msw == 1:
        for idx in range(num_obs):
            target_lon = obs_lon[idx]
            target_lat = obs_lat[idx]
            cell = None
            for jy in range(ny - 1):
                for ix in range(nx - 1):
                    lon_patch = grid_lon[ix : ix + 2, jy : jy + 2]
                    lat_patch = grid_lat[ix : ix + 2, jy : jy + 2]
                    if (
                        lon_patch.min() <= target_lon <= lon_patch.max()
                        and lat_patch.min() <= target_lat <= lat_patch.max()
                    ):
                        cell = (ix, jy)
                        break
                if cell:
                    break
            if cell is None:
                continue
            ix, jy = cell
            corners = [
                (ix, jy),
                (ix + 1, jy),
                (ix, jy + 1),
                (ix + 1, jy + 1),
            ]
            distances = np.array(
                [
                    com_distll_1(
                        grid_lon[i0, j0],
                        grid_lat[i0, j0],
                        target_lon,
                        target_lat,
                    )
                    for i0, j0 in corners
                ]
            )
            distances *= 1.0e-3
            dist_sq = np.clip(distances * distances, 1.0e-12, None)
            sum_dist = (
                dist_sq[0] * dist_sq[1] * dist_sq[2]
                + dist_sq[1] * dist_sq[2] * dist_sq[3]
                + dist_sq[2] * dist_sq[3] * dist_sq[0]
                + dist_sq[3] * dist_sq[0] * dist_sq[1]
            )
            if sum_dist == 0.0:
                oi[idx] = float(ix + 1)
                oj[idx] = float(jy + 1)
                continue
            ratio = np.empty(4, dtype=float)
            ratio[0] = dist_sq[1] * dist_sq[2] * dist_sq[3] / sum_dist
            ratio[1] = dist_sq[2] * dist_sq[3] * dist_sq[0] / sum_dist
            ratio[2] = dist_sq[3] * dist_sq[0] * dist_sq[1] / sum_dist
            ratio[3] = dist_sq[0] * dist_sq[1] * dist_sq[2] / sum_dist
            oi[idx] = (
                ratio[0] * (ix + 1)
                + ratio[1] * (ix + 2)
                + ratio[2] * (ix + 1)
                + ratio[3] * (ix + 2)
            )
            oj[idx] = (
                ratio[0] * (jy + 1)
                + ratio[1] * (jy + 1)
                + ratio[2] * (jy + 2)
                + ratio[3] * (jy + 2)
            )
    elif msw == 2:
        max_dist = 2.0e6
        for idx in range(num_obs):
            target_lon = obs_lon[idx]
            target_lat = obs_lat[idx]
            distances = []
            for jy in range(ny):
                for ix in range(nx):
                    d = com_distll_1(
                        grid_lon[ix, jy], grid_lat[ix, jy], target_lon, target_lat
                    )
                    if d <= max_dist:
                        distances.append((d, ix, jy))
            if len(distances) < 4:
                continue
            distances.sort(key=lambda item: item[0])
            nearest = distances[:4]
            dist = np.array([item[0] for item in nearest], dtype=float) * 1.0e-3
            dist_sq = np.clip(dist * dist, 1.0e-12, None)
            sum_dist = (
                dist_sq[0] * dist_sq[1] * dist_sq[2]
                + dist_sq[1] * dist_sq[2] * dist_sq[3]
                + dist_sq[2] * dist_sq[3] * dist_sq[0]
                + dist_sq[3] * dist_sq[0] * dist_sq[1]
            )
            if sum_dist == 0.0:
                oi[idx] = float(nearest[0][1] + 1)
                oj[idx] = float(nearest[0][2] + 1)
                continue
            ratio = np.empty(4, dtype=float)
            ratio[0] = dist_sq[1] * dist_sq[2] * dist_sq[3] / sum_dist
            ratio[1] = dist_sq[2] * dist_sq[3] * dist_sq[0] / sum_dist
            ratio[2] = dist_sq[3] * dist_sq[0] * dist_sq[1] / sum_dist
            ratio[3] = dist_sq[0] * dist_sq[1] * dist_sq[2] / sum_dist
            oi[idx] = sum(
                ratio[i] * (nearest[i][1] + 1) for i in range(4)
            )
            oj[idx] = sum(
                ratio[i] * (nearest[i][2] + 1) for i in range(4)
            )
    else:
        raise ValueError("Unsupported mode: msw must be 1 or 2")

    return oi, oj


def com_utc2tai(iy: int, im: int, iday: int, ih: int, imin: int, sec: float) -> float:
    """Convert UTC to seconds since 1 Jan 1993 (TAI93)."""

    mins = 60.0
    hour = 60.0 * mins
    day = 24.0 * hour
    year = 365.0 * day
    mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    tai93 = float(iy - 1993) * year + math.floor((iy - 1993) / 4.0) * day
    days = iday - 1
    for i in range(12):
        if im > i + 1:
            days += mdays[i]
    if iy % 4 == 0 and im > 2:
        days += 1
    tai93 += days * day + ih * hour + imin * mins + sec

    if iy > 1993 or (iy == 1993 and im > 6):
        tai93 += 1.0
    if iy > 1994 or (iy == 1994 and im > 6):
        tai93 += 1.0
    if iy > 1995:
        tai93 += 1.0
    if iy > 1997 or (iy == 1997 and im > 6):
        tai93 += 1.0
    if iy > 1998:
        tai93 += 1.0
    if iy > 2005:
        tai93 += 1.0
    if iy > 2008:
        tai93 += 1.0
    if iy > 2012 or (iy == 2012 and im > 6):
        tai93 += 1.0

    return tai93


def com_tai2utc(tai93: float) -> Tuple[int, int, int, int, int, float]:
    """Convert seconds since 1 Jan 1993 to UTC."""

    leapsec = np.array(
        [15638399, 47174400, 94608001, 141868802, 189302403, 410227204, 504921605, 615254406],
        dtype=float,
    )
    mins = 60.0
    hour = 60.0 * mins
    day = 24.0 * hour
    year = 365.0 * day
    mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    tai = float(tai93)
    sec = 0.0
    for leap in leapsec:
        if math.floor(tai93) == leap + 1:
            sec = 60.0 + tai93 - math.floor(tai93)
        if math.floor(tai93) > leap:
            tai -= 1.0

    iy = 1993 + int(math.floor(tai / year))
    wk = tai - (iy - 1993) * year - math.floor((iy - 1993) / 4.0) * day
    if wk < 0.0:
        iy -= 1
        wk = tai - (iy - 1993) * year - math.floor((iy - 1993) / 4.0) * day

    days = int(math.floor(wk / day))
    wk -= days * day

    im = 1
    for m, md in enumerate(mdays, start=1):
        leap = 1 if m == 2 and iy % 4 == 0 else 0
        if days >= md + leap:
            days -= md + leap
            im += 1
        else:
            break
    iday = days + 1

    ih = int(math.floor(wk / hour))
    wk -= ih * hour
    imin = int(math.floor(wk / mins))
    if sec < 60.0:
        sec = wk - imin * mins

    return iy, im, iday, ih, imin, sec


def com_mdays(iy: int, im: int) -> int:
    """Number of days in the month."""

    if im in {1, 3, 5, 7, 8, 10, 12}:
        return 31
    if im in {4, 6, 9, 11}:
        return 30
    if im == 2:
        if iy % 100 == 0:
            return 29 if iy % 400 == 0 else 28
        return 29 if iy % 4 == 0 else 28
    raise ValueError("Invalid month")


def com_datetime_reg(iy: int, im: int, iday: int, ih: int, imin: int, isec: int) -> Tuple[int, int, int, int, int, int]:
    """Regularise date and time values, mirroring the Fortran implementation."""

    while im <= 0:
        im += 12
        iy -= 1
    while im > 12:
        im -= 12
        iy += 1
    while isec < 0:
        isec += 60
        imin -= 1
    while isec >= 60:
        isec -= 60
        imin += 1
    while imin < 0:
        imin += 60
        ih -= 1
    while imin >= 60:
        imin -= 60
        ih += 1
    while ih < 0:
        ih += 24
        iday -= 1
    while ih >= 24:
        ih -= 24
        iday += 1
    while iday <= 0:
        im -= 1
        if im <= 0:
            im += 12
            iy -= 1
        iday += com_mdays(iy, im)
    while True:
        mdays = com_mdays(iy, im)
        if iday <= mdays:
            break
        iday -= mdays
        im += 1
        if im > 12:
            im -= 12
            iy += 1
    return iy, im, iday, ih, imin, isec


def com_gamma(x: float) -> float:
    """Gamma function."""

    if float(x).is_integer() and x <= 0:
        return math.inf
    return float(math.gamma(x))


def com_ll_arc_distance(
    ini_lon: float, ini_lat: float, distance: float, azimuth: float
) -> Tuple[float, float]:
    """Move along a great-circle arc."""

    if distance == 0.0:
        return ini_lon, ini_lat

    cdist = math.cos(distance / RE)
    sdist = math.sin(distance / RE)
    sinll1 = math.sin(ini_lat * DEG2RAD)
    cosll1 = math.cos(ini_lat * DEG2RAD)

    final_lat = math.asin(sinll1 * cdist + cosll1 * sdist * math.cos(azimuth * DEG2RAD)) / DEG2RAD
    final_lon = ini_lon + (
        math.atan2(
            sdist * math.sin(azimuth * DEG2RAD),
            cosll1 * cdist - sinll1 * sdist * math.cos(azimuth * DEG2RAD),
        )
        / DEG2RAD
    )
    return final_lon, final_lat
