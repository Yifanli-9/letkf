"""LETKF driver utilities working on :mod:`xarray` objects."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import xarray as xr

from common.common_letkf import letkf_core

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AssimilationResult:
    """Container for prior and posterior statistics."""

    prior_mean: xr.DataArray
    prior_std: xr.DataArray
    post_mean: xr.DataArray
    post_std: xr.DataArray
    inflation: float


def build_hist_time_selector(len_hist: int, len_total: int) -> np.ndarray:
    """Return the extraction matrix that isolates historical times.

    The matrix mimics the behaviour of the historical Fortran implementation by
    placing an identity matrix in the top-left corner and zeros elsewhere.
    """

    selector = np.zeros((len_hist, len_total), dtype=float)
    selector[:, :len_hist] = np.eye(len_hist, dtype=float)
    return selector


def _stack_state(data: xr.DataArray, member_dim: str, state_dims: Iterable[str]) -> np.ndarray:
    ordered = data.transpose(member_dim, *state_dims)
    ne = ordered.sizes[member_dim]
    state_size = int(np.prod([ordered.sizes[dim] for dim in state_dims]))
    values = ordered.values.reshape(ne, state_size).T
    return values


def _reshape_state(
    values: np.ndarray,
    template: xr.DataArray,
    state_dims: Iterable[str],
) -> xr.DataArray:
    ordered = template.transpose(*state_dims)
    data = values.reshape([ordered.sizes[dim] for dim in state_dims])
    reshaped = xr.DataArray(
        data,
        coords={dim: ordered.coords[dim] for dim in state_dims},
        dims=tuple(state_dims),
        attrs=template.attrs,
    )
    reshaped.name = template.name
    return reshaped


def _collect_observations(
    obs: xr.DataArray,
    hist_template: xr.DataArray,
    *,
    time_dim: str,
    state_dims: Iterable[str],
    obs_error_var: float | xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract observation values, indices and variances."""

    target_coords = {time_dim: hist_template.coords[time_dim]}
    for dim in state_dims:
        if dim in (time_dim,):
            continue
        if dim in hist_template.dims and dim in obs.dims:
            target_coords[dim] = hist_template.coords[dim]

    obs_on_hist = obs.reindex(target_coords)
    obs_on_hist = obs_on_hist.transpose(time_dim, *[dim for dim in state_dims if dim != time_dim])
    stacked = obs_on_hist.stack(obs_state=(time_dim, *[dim for dim in state_dims if dim != time_dim]))
    values = stacked.values

    if isinstance(obs_error_var, xr.DataArray):
        obs_var = obs_error_var.reindex(target_coords)
        obs_var = obs_var.transpose(time_dim, *[dim for dim in state_dims if dim != time_dim])
        obs_var = obs_var.stack(obs_state=(time_dim, *[dim for dim in state_dims if dim != time_dim]))
        var_values = obs_var.values.astype(float)
    else:
        var_values = np.full_like(values, float(obs_error_var), dtype=float)

    valid_mask = np.isfinite(values) & np.isfinite(var_values) & (var_values > 0.0)
    if not np.any(valid_mask):
        return np.empty(0, dtype=float), np.empty(0, dtype=int), np.empty(0, dtype=float)

    indices = np.nonzero(valid_mask)[0]
    obs_values = values[valid_mask]
    obs_var_values = var_values[valid_mask]

    return obs_values.astype(float), indices.astype(int), obs_var_values.astype(float)


def run_letkf_assimilation(
    hist: xr.DataArray,
    fut: xr.DataArray,
    obs: xr.DataArray,
    *,
    member_dim: str = "member",
    time_dim: str = "year",
    obs_error_var: float | xr.DataArray = 1.0,
    inflation: float = 1.0,
) -> AssimilationResult:
    """Assimilate *obs* into the ensemble defined by *hist* and *fut*.

    The returned :class:`AssimilationResult` only contains ensemble statistics
    (mean and standard deviation) in accordance with the user requirements.
    """

    if hist.dims != fut.dims:
        raise ValueError("Historical and future ensembles must share dimensions")
    if member_dim not in hist.dims:
        raise ValueError(f"Historical data is missing the {member_dim!r} dimension")
    if hist.sizes[member_dim] != fut.sizes[member_dim]:
        raise ValueError("Historical and future ensembles must have the same size")
    if hist.sizes[member_dim] < 2:
        raise ValueError("At least two ensemble members are required")

    if time_dim not in hist.dims:
        raise ValueError(f"Historical data is missing the {time_dim!r} dimension")

    state_dims = [dim for dim in hist.dims if dim != member_dim]
    if time_dim not in state_dims:
        raise ValueError(f"Historical data does not contain the time dimension {time_dim!r}")
    state_dims.remove(time_dim)
    state_dims = [time_dim] + state_dims

    hist = hist.transpose(member_dim, *state_dims)
    fut = fut.transpose(member_dim, *state_dims)
    combined = xr.concat([hist, fut], dim=time_dim)

    prior_mean = combined.mean(dim=member_dim)
    prior_std = combined.std(dim=member_dim, ddof=1)

    hist_len = hist.sizes[time_dim]
    total_len = combined.sizes[time_dim]

    LOGGER.debug("Historical samples: %s, combined samples: %s", hist_len, total_len)
    time_selector = build_hist_time_selector(hist_len, total_len)
    if not np.array_equal(time_selector[:, :hist_len], np.eye(hist_len)):
        raise RuntimeError("Invalid historical time selector")

    xb = _stack_state(combined, member_dim, state_dims)
    xb_mean = xb.mean(axis=1)
    xb_pert = xb - xb_mean[:, None]
    ne = xb.shape[1]

    hist_template = hist.mean(dim=member_dim)
    obs_values, obs_indices, obs_var = _collect_observations(
        obs, hist_template, time_dim=time_dim, state_dims=state_dims, obs_error_var=obs_error_var
    )

    if obs_values.size == 0:
        LOGGER.warning("No valid observations found; analysis equals the background")
        post_mean = prior_mean
        post_std = prior_std
        return AssimilationResult(prior_mean, prior_std, post_mean, post_std, inflation)

    points_per_time = int(np.prod([hist.sizes[dim] for dim in state_dims if dim != time_dim]))
    hist_time_mask = time_selector.any(axis=0)
    hist_state_mask = np.repeat(hist_time_mask, points_per_time)
    if np.any(~hist_state_mask[obs_indices]):
        raise IndexError("Observation index exceeds historical state extent")

    hx = xb[obs_indices, :]
    hx_mean = hx.mean(axis=1)
    hdxb = hx - hx_mean[:, None]
    dep = obs_values - hx_mean

    rdiag = obs_var
    rloc = np.ones_like(rdiag)

    LOGGER.info("Running LETKF update for %d observations and %d ensemble members", obs_values.size, ne)
    result = letkf_core(
        hdxb,
        rdiag,
        rloc,
        dep,
        inflation,
        return_transm=True,
    )

    trans = result["trans"]
    transm = result["transm"]
    updated_inflation = float(result.get("parm_infl", inflation))

    xa_pert = xb_pert @ trans.T
    mean_increment = xb_pert @ transm
    xa_mean = xb_mean + mean_increment
    xa_std = np.sqrt(np.sum(xa_pert**2, axis=1) / float(ne - 1))

    post_mean = _reshape_state(xa_mean, prior_mean, state_dims)
    post_std = _reshape_state(xa_std, prior_std, state_dims)

    return AssimilationResult(prior_mean, prior_std, post_mean, post_std, updated_inflation)
