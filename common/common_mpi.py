"""Minimal MPI helpers used by the Python rewrite."""

from __future__ import annotations

try:  # pragma: no cover - optional dependency
    from mpi4py import MPI
except Exception:  # pragma: no cover - mpi4py is optional
    MPI = None

nprocs = 1
myrank = 0
MPI_r_size = None

__all__ = ["initialize_mpi", "finalize_mpi", "nprocs", "myrank", "MPI_r_size"]


def initialize_mpi() -> None:
    """Initialise MPI if :mod:`mpi4py` is available."""

    global nprocs, myrank, MPI_r_size
    if MPI is None:
        nprocs = 1
        myrank = 0
        MPI_r_size = None
        return

    if not MPI.Is_initialized():  # pragma: no cover - depends on MPI runtime
        MPI.Init()

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    myrank = comm.Get_rank()
    MPI_r_size = MPI.DOUBLE


def finalize_mpi() -> None:
    """Shutdown MPI if it was previously initialised."""

    if MPI is not None and MPI.Is_initialized():  # pragma: no cover
        MPI.Finalize()
