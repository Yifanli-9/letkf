"""Helpers for optional MPI support."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

MPI: Any | None
_spec = importlib.util.find_spec("mpi4py")
if _spec is None:
    MPI = None
else:
    mpi4py = importlib.import_module("mpi4py")
    MPI = mpi4py.MPI

__all__ = ["MPI", "get_world_comm"]


def get_world_comm():
    """Return :data:`mpi4py.MPI.COMM_WORLD` when MPI is available."""

    if MPI is None:
        return None
    return MPI.COMM_WORLD
