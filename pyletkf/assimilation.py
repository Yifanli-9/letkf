"""LETKF driver utilities working on :mod:`xarray` objects."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import xarray as xr

from common.common_letkf import letkf_core

from .localization import gaspari_cohn
from .mpi import MPI, get_world_comm

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AssimilationResult:
    """Container for prior and posterior statistics."""

    prior_mean: xr.DataArray
    prior_std: xr.DataArray
    post_mean: xr.DataArray
    post_std: xr.DataArray
    inflation: float


def _normalise_coordinate(values: np.ndarray) -> np.ndarray:
    """Convert arbitrary coordinate arrays to floating point values."""

    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        array = array.astype("datetime64[ns]").astype(np.int64)
    elif np.issubdtype(array.dtype, np.timedelta64):
        array = array.astype("timedelta64[ns]").astype(np.int64)
    elif not np.issubdtype(array.dtype, np.number):
        return np.arange(array.size, dtype=float)
    return array.astype(float)


def _space_coordinate_matrix(space_coord: xr.Coordinate, space_dims: Sequence[str]) -> np.ndarray:
    """Return a coordinate matrix used to compute spatial distances."""

    index = space_coord.to_index()
    if getattr(index, "nlevels", 1) == 1:
        values = _normalise_coordinate(index.values)
        return values.reshape(-1, 1)

    columns = []
    for dim in space_dims:
        level = index.get_level_values(dim).to_numpy()
        columns.append(_normalise_coordinate(level))
    if not columns:
        return np.zeros((index.size, 0), dtype=float)
    return np.column_stack(columns)


def _prepare_observations(
    obs: xr.DataArray,
    obs_error_var: float | xr.DataArray,
    template: xr.DataArray,
    *,
    time_dim: str,
    space_dims: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Align observations and their error variances with the model grid."""

    coords = {time_dim: template.coords[time_dim]}
    for dim in space_dims:
        coords[dim] = template.coords[dim]

    obs_on_grid = obs.reindex(coords)
    obs_flat = obs_on_grid.stack(space=space_dims).transpose(time_dim, "space")
    obs_values = obs_flat.values.astype(float)

    if isinstance(obs_error_var, xr.DataArray):
        obs_var = obs_error_var.reindex(coords)
        obs_var_flat = obs_var.stack(space=space_dims).transpose(time_dim, "space").values.astype(float)
    else:
        obs_var_flat = np.full(obs_values.shape, float(obs_error_var), dtype=float)

    valid_mask = np.isfinite(obs_values) & np.isfinite(obs_var_flat) & (obs_var_flat > 0.0)
    obs_indices = [np.nonzero(valid_mask[t])[0] for t in range(valid_mask.shape[0])]

    return obs_values, obs_var_flat, obs_indices


def _stack_ensemble(
    data: xr.DataArray,
    *,
    member_dim: str,
    time_dim: str,
    space_dims: Sequence[str],
) -> tuple[np.ndarray, xr.Coordinate]:
    """Return the ensemble as an array shaped ``(nspace, ntime, ne)``."""

    stacked = data.stack(space=space_dims)
    stacked = stacked.transpose("space", time_dim, member_dim)
    return stacked.values.astype(float), stacked.coords["space"]


def run_letkf_assimilation(
    hist: xr.DataArray,
    fut: xr.DataArray,
    obs: xr.DataArray,
    *,
    member_dim: str = "member",
    time_dim: str = "year",
    obs_error_var: float | xr.DataArray = 1.0,
    inflation: float = 1.0,
    localization_radius: float | None = None,
    comm=None,
) -> AssimilationResult:
    """Assimilate *obs* into the ensemble defined by *hist* and *fut*.

    The returned :class:`AssimilationResult` contains ensemble statistics (mean and
    spread).  Spatial localisation follows the Gaspari--Cohn taper.  When an MPI
    communicator is provided or automatically detected, spatial grid points are
    distributed evenly across the participating ranks.
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
    if localization_radius is not None and localization_radius <= 0.0:
        raise ValueError("The localisation radius must be positive")

    if comm is None:
        comm = get_world_comm()
    if comm is not None:
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        size = 1
        rank = 0

    space_dims = [dim for dim in hist.dims if dim not in (member_dim, time_dim)]
    if not space_dims:
        raise ValueError("At least one spatial dimension is required for localisation")

    hist = hist.transpose(member_dim, time_dim, *space_dims)
    fut = fut.transpose(member_dim, time_dim, *space_dims)
    combined = xr.concat([hist, fut], dim=time_dim)

    template = combined.mean(dim=member_dim)

    ensemble_values, space_coord = _stack_ensemble(
        combined, member_dim=member_dim, time_dim=time_dim, space_dims=space_dims
    )
    nspace, ntime_total, ne = ensemble_values.shape

    xb_mean = ensemble_values.mean(axis=2)
    xb_pert = ensemble_values - xb_mean[..., None]
    prior_std_values = np.sqrt(np.sum(xb_pert**2, axis=2) / float(ne - 1))

    coords = {"space": space_coord, time_dim: combined.coords[time_dim]}
    prior_mean = xr.DataArray(xb_mean, coords=coords, dims=("space", time_dim)).unstack("space")
    prior_std = xr.DataArray(prior_std_values, coords=coords, dims=("space", time_dim)).unstack("space")
    prior_mean = prior_mean.transpose(time_dim, *space_dims)
    prior_std = prior_std.transpose(time_dim, *space_dims)
    prior_mean.name = hist.name
    prior_std.name = hist.name
    prior_mean.attrs.update(hist.attrs)
    prior_std.attrs.update(hist.attrs)

    obs_values, obs_var_values, obs_indices_per_time = _prepare_observations(
        obs,
        obs_error_var,
        template,
        time_dim=time_dim,
        space_dims=space_dims,
    )
    total_obs = int(sum(len(indices) for indices in obs_indices_per_time))

    space_coordinates = _space_coordinate_matrix(space_coord, space_dims)

    if rank == 0:
        LOGGER.info(
            "Running LETKF update for %d spatial points, %d time steps, %d ensemble members and %d observations",
            nspace,
            ntime_total,
            ne,
            total_obs,
        )
        if localization_radius is not None:
            LOGGER.info("Using Gaspari--Cohn localisation radius %.3f", localization_radius)
        elif total_obs == 0:
            LOGGER.warning("No valid observations found; analysis equals the background")

    post_mean_local = np.zeros_like(xb_mean)
    post_std_local = np.zeros_like(prior_std_values)
    inflation_sum = 0.0
    successful_updates = 0

    assigned_indices: Iterable[int]
    if size > 1:
        assigned_indices = range(rank, nspace, size)
    else:
        assigned_indices = range(nspace)

    for space_index in assigned_indices:
        xb_local_mean = xb_mean[space_index, :]
        xb_local_pert = xb_pert[space_index, :, :]
        prior_std_local = prior_std_values[space_index, :]

        hdxb_chunks: list[np.ndarray] = []
        dep_chunks: list[np.ndarray] = []
        rdiag_chunks: list[np.ndarray] = []
        rloc_chunks: list[np.ndarray] = []

        for time_index, obs_indices in enumerate(obs_indices_per_time):
            if obs_indices.size == 0:
                continue

            hx_mean = xb_mean[obs_indices, time_index]
            hdxb = xb_pert[obs_indices, time_index, :]
            obs_val = obs_values[time_index, obs_indices]
            obs_var = obs_var_values[time_index, obs_indices]

            if localization_radius is not None:
                distances = np.linalg.norm(
                    space_coordinates[obs_indices, :] - space_coordinates[space_index, :], axis=1
                )
                weights = gaspari_cohn(distances, localization_radius)
            else:
                weights = np.ones_like(obs_val, dtype=float)

            mask = (
                (weights > 0.0)
                & np.isfinite(weights)
                & np.isfinite(hx_mean)
                & np.isfinite(obs_val)
                & np.isfinite(obs_var)
            )
            if not np.any(mask):
                continue

            hdxb_chunks.append(hdxb[mask, :])
            dep_chunks.append(obs_val[mask] - hx_mean[mask])
            rdiag_chunks.append(obs_var[mask])
            rloc_chunks.append(weights[mask])

        if not hdxb_chunks:
            post_mean_local[space_index, :] = xb_local_mean
            post_std_local[space_index, :] = prior_std_local
            continue

        hdxb_matrix = np.vstack(hdxb_chunks).astype(float)
        dep_vector = np.concatenate(dep_chunks).astype(float)
        rdiag_vector = np.concatenate(rdiag_chunks).astype(float)
        rloc_vector = np.concatenate(rloc_chunks).astype(float)

        result = letkf_core(
            hdxb_matrix,
            rdiag_vector,
            rloc_vector,
            dep_vector,
            inflation,
            return_transm=True,
        )

        trans = result["trans"]
        transm = result["transm"]
        updated_inflation = float(result.get("parm_infl", inflation))

        xa_pert = xb_local_pert @ trans.T
        mean_increment = xb_local_pert @ transm
        xa_mean = xb_local_mean + mean_increment
        xa_std = np.sqrt(np.sum(xa_pert**2, axis=1) / float(ne - 1))

        post_mean_local[space_index, :] = xa_mean
        post_std_local[space_index, :] = xa_std
        inflation_sum += updated_inflation
        successful_updates += 1

    if comm is not None and size > 1:
        post_mean_total = np.zeros_like(post_mean_local)
        comm.Allreduce(post_mean_local, post_mean_total, op=MPI.SUM)
        post_std_total = np.zeros_like(post_std_local)
        comm.Allreduce(post_std_local, post_std_total, op=MPI.SUM)

        buffer = np.array([inflation_sum, float(successful_updates)], dtype=float)
        global_buffer = np.zeros_like(buffer)
        comm.Allreduce(buffer, global_buffer, op=MPI.SUM)
        inflation_sum_total = global_buffer[0]
        inflation_count_total = int(round(global_buffer[1]))
    else:
        post_mean_total = post_mean_local
        post_std_total = post_std_local
        inflation_sum_total = inflation_sum
        inflation_count_total = successful_updates

    coords = {"space": space_coord, time_dim: combined.coords[time_dim]}
    post_mean = xr.DataArray(post_mean_total, coords=coords, dims=("space", time_dim)).unstack("space")
    post_std = xr.DataArray(post_std_total, coords=coords, dims=("space", time_dim)).unstack("space")
    post_mean = post_mean.transpose(time_dim, *space_dims)
    post_std = post_std.transpose(time_dim, *space_dims)
    post_mean.name = hist.name
    post_std.name = hist.name
    post_mean.attrs.update(hist.attrs)
    post_std.attrs.update(hist.attrs)

    if inflation_count_total > 0:
        final_inflation = float(inflation_sum_total / float(inflation_count_total))
    else:
        final_inflation = float(inflation)

    if rank == 0:
        LOGGER.info(
            "Completed LETKF updates for %d of %d spatial points", inflation_count_total, nspace
        )

    return AssimilationResult(prior_mean, prior_std, post_mean, post_std, final_inflation)
