"""Input/output helpers for the stand-alone Python LETKF pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import xarray as xr

LOGGER = logging.getLogger(__name__)


def _ensure_nc_files(directory: Path) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.suffix == ".nc")
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {directory!s}")
    return files


def load_ensemble(
    directory: str | Path,
    var_name: str,
    *,
    member_dim: str = "member",
    load_coords: Mapping[str, Iterable] | None = None,
) -> xr.DataArray:
    """Load ensemble members from NetCDF files in *directory*.

    Parameters
    ----------
    directory:
        Folder containing one NetCDF file per ensemble member.
    var_name:
        Name of the state variable to load.
    member_dim:
        Name of the ensemble dimension to create.
    load_coords:
        Optional mapping of dimension name to coordinates.  When provided, the
        resulting data are reindexed onto these coordinates which is useful to
        enforce a consistent grid between different ensemble batches.
    """

    directory = Path(directory)
    files = _ensure_nc_files(directory)

    members: list[xr.DataArray] = []
    for idx, path in enumerate(files):
        LOGGER.info("Loading ensemble member from %s", path)
        with xr.open_dataset(path) as ds:
            if var_name not in ds:
                raise KeyError(f"Variable {var_name!r} not found in {path!s}")
            data = ds[var_name].load()

        if member_dim in data.dims:
            raise ValueError(
                f"Variable {var_name!r} in {path!s} already has a {member_dim!r} dimension"
            )

        if load_coords is not None:
            reindexer = {dim: coords for dim, coords in load_coords.items() if dim in data.dims}
            if reindexer:
                data = data.reindex(reindexer)

        data = data.expand_dims({member_dim: [idx]})
        data = data.assign_coords({member_dim: idx})
        members.append(data)

    ensemble = xr.concat(members, dim=member_dim)
    ensemble.attrs["source_files"] = [str(path) for path in files]
    return ensemble


def load_observations(path: str | Path, var_name: str) -> xr.DataArray:
    """Load the observation field from *path*."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Observation file {path!s} not found")

    LOGGER.info("Loading observations from %s", path)
    with xr.open_dataset(path) as ds:
        if var_name not in ds:
            raise KeyError(f"Variable {var_name!r} not present in {path!s}")
        data = ds[var_name].load()

    data.attrs["source_file"] = str(path)
    return data


def save_outputs(outputs: Mapping[str, xr.DataArray], output_dir: str | Path) -> None:
    """Persist output fields to NetCDF files.

    The *outputs* mapping should associate a relative file name with the
    corresponding :class:`xarray.DataArray` that is to be saved.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, data in outputs.items():
        path = output_dir / filename
        dataset = data.to_dataset(name=data.name or "value")
        LOGGER.info("Writing %s", path)
        dataset.to_netcdf(path)
