"""kinUV — visibility-plane kinematic models. 066-1 DFT oracle; 066-5 FINUFFT T2."""

from .constants import ARCSEC_TO_RAD, C_LIGHT_M_S, C_LIGHT_KM_S, F_REST_CO21_HZ
from .decisions import load_decision_index, requires
from .transforms import dft_numpy, nufft2_degrid, uv_wavelengths

__version__ = "0.1.0.dev0"

__all__ = [
    "ARCSEC_TO_RAD",
    "C_LIGHT_M_S",
    "C_LIGHT_KM_S",
    "F_REST_CO21_HZ",
    "dft_numpy",
    "nufft2_degrid",
    "uv_wavelengths",
    "load_decision_index",
    "requires",
    "__version__",
]
