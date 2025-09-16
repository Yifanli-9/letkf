"""Python implementations of the utilities from the original Fortran ``common``
package.

The submodules provide drop-in replacements for a subset of the routines that
were historically implemented in Fortran.  They have been rewritten in Python
with a focus on readability and interoperability with the scientific Python
stack.  Only the files that are still required by the repository were ported;
``common_enkf`` and ``common_lekf`` remain in their original form as requested.
"""

from .common import (
    DEG2RAD,
    GG,
    PI,
    RAD2DEG,
    RE,
    com_anomcorrel,
    com_datetime_reg,
    com_distll,
    com_distll_1,
    com_filter_lanczos,
    com_gamma,
    com_interp_spline,
    com_l2norm,
    com_ll_arc_distance,
    com_mean,
    com_mdays,
    com_pos2ij,
    com_rand,
    com_randn,
    com_rms,
    com_stdev,
    com_time2ymdh,
    com_timeinc_hr,
    com_tai2utc,
    com_utc2tai,
    com_ymdh2time,
    init_gen_rand,
)
from .common_mpi import finalize_mpi, initialize_mpi, myrank, nprocs
from .common_mtx import mtx_eigen, mtx_inv, mtx_inv_rg, mtx_sqrt
from .common_obs import (
    ID_PS_OBS,
    ID_Q_OBS,
    ID_RAIN_OBS,
    ID_RH_OBS,
    ID_S_OBS,
    ID_T_OBS,
    ID_U_OBS,
    ID_V_OBS,
    ID_Z_OBS,
    get_nobs,
    read_obs,
)
from .common_letkf import letkf_core
from .minimizelib import initialize_minimizer, minimize, terminate_minimizer

__all__ = [
    "DEG2RAD",
    "GG",
    "PI",
    "RAD2DEG",
    "RE",
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
    "initialize_mpi",
    "finalize_mpi",
    "myrank",
    "nprocs",
    "mtx_eigen",
    "mtx_inv",
    "mtx_inv_rg",
    "mtx_sqrt",
    "letkf_core",
    "initialize_minimizer",
    "minimize",
    "terminate_minimizer",
    "ID_PS_OBS",
    "ID_Q_OBS",
    "ID_RAIN_OBS",
    "ID_RH_OBS",
    "ID_S_OBS",
    "ID_T_OBS",
    "ID_U_OBS",
    "ID_V_OBS",
    "ID_Z_OBS",
    "get_nobs",
    "read_obs",
]
