"""Command line interface to run the Python LETKF pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import xarray as xr

from .assimilation import run_letkf_assimilation
from .io import load_ensemble, load_observations, save_outputs

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _load_obs_error(
    obs: xr.DataArray,
    *,
    obs_error_std: float | None,
    obs_error_var_name: str | None,
    obs_dataset_path: Path,
) -> float | xr.DataArray:
    if obs_error_var_name:
        with xr.open_dataset(obs_dataset_path) as ds:
            if obs_error_var_name not in ds:
                raise KeyError(
                    f"Observation error variable {obs_error_var_name!r} not found in {obs_dataset_path!s}"
                )
            obs_var = ds[obs_error_var_name].load()
        if obs_var.dims != obs.dims:
            LOGGER.warning("Observation error field dimensions differ from observations; attempting alignment")
        return obs_var

    if obs_error_std is None:
        raise ValueError("Either an observation error standard deviation or a variable name must be provided")
    if obs_error_std <= 0.0:
        raise ValueError("Observation error standard deviation must be positive")
    return float(obs_error_std) ** 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a LETKF experiment using xarray inputs")
    parser.add_argument("--hist-dir", required=True, help="Directory containing historical ensemble NetCDF files")
    parser.add_argument("--fut-dir", required=True, help="Directory containing future ensemble NetCDF files")
    parser.add_argument("--obs", required=True, help="Observation NetCDF file")
    parser.add_argument("--var", required=True, help="Variable name present in ensemble files")
    parser.add_argument(
        "--obs-var",
        default=None,
        help="Variable name holding the observations (defaults to --var)",
    )
    parser.add_argument(
        "--obs-error-std",
        type=float,
        default=None,
        help="Observation error standard deviation (constant for all observations)",
    )
    parser.add_argument(
        "--obs-error-var-name",
        default=None,
        help="Name of the variable in the observation file holding the error variance",
    )
    parser.add_argument("--time-dim", default="year", help="Name of the temporal dimension")
    parser.add_argument("--member-dim", default="member", help="Name of the ensemble member dimension")
    parser.add_argument("--inflation", type=float, default=1.0, help="Background covariance inflation parameter")
    parser.add_argument("--output-dir", required=True, help="Directory where NetCDF outputs will be written")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional JSON file storing run metadata to be embedded in the outputs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    obs_var_name = args.obs_var or args.var

    hist = load_ensemble(args.hist_dir, args.var, member_dim=args.member_dim)
    spatial_coords = {
        dim: hist.coords[dim]
        for dim in hist.dims
        if dim not in (args.member_dim, args.time_dim) and dim in hist.coords
    }
    fut = load_ensemble(args.fut_dir, args.var, member_dim=args.member_dim, load_coords=spatial_coords)
    obs = load_observations(args.obs, obs_var_name)

    obs_error_var = _load_obs_error(
        obs,
        obs_error_std=args.obs_error_std,
        obs_error_var_name=args.obs_error_var_name,
        obs_dataset_path=Path(args.obs),
    )

    result = run_letkf_assimilation(
        hist,
        fut,
        obs,
        member_dim=args.member_dim,
        time_dim=args.time_dim,
        obs_error_var=obs_error_var,
        inflation=args.inflation,
    )

    time_dim = args.time_dim
    hist_len = hist.sizes[time_dim]

    prior_mean_hist = result.prior_mean.isel({time_dim: slice(0, hist_len)})
    prior_std_hist = result.prior_std.isel({time_dim: slice(0, hist_len)})
    post_mean_hist = result.post_mean.isel({time_dim: slice(0, hist_len)})
    post_std_hist = result.post_std.isel({time_dim: slice(0, hist_len)})

    prior_mean_fut = result.prior_mean.isel({time_dim: slice(hist_len, None)})
    prior_std_fut = result.prior_std.isel({time_dim: slice(hist_len, None)})
    post_mean_fut = result.post_mean.isel({time_dim: slice(hist_len, None)})
    post_std_fut = result.post_std.isel({time_dim: slice(hist_len, None)})

    outputs: dict[str, xr.DataArray] = {
        "hist_prior_mean.nc": prior_mean_hist.rename(f"{args.var}_prior_mean"),
        "hist_prior_std.nc": prior_std_hist.rename(f"{args.var}_prior_std"),
        "hist_post_mean.nc": post_mean_hist.rename(f"{args.var}_post_mean"),
        "hist_post_std.nc": post_std_hist.rename(f"{args.var}_post_std"),
        "fut_prior_mean.nc": prior_mean_fut.rename(f"{args.var}_prior_mean"),
        "fut_prior_std.nc": prior_std_fut.rename(f"{args.var}_prior_std"),
        "fut_post_mean.nc": post_mean_fut.rename(f"{args.var}_post_mean"),
        "fut_post_std.nc": post_std_fut.rename(f"{args.var}_post_std"),
    }

    if args.metadata:
        metadata_path = Path(args.metadata)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata: dict[str, Any] = json.load(handle)
        for data in outputs.values():
            data.attrs.update(metadata)

    save_outputs(outputs, args.output_dir)
    LOGGER.info("Analysis inflation parameter: %.3f", result.inflation)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
