"""kinUV — visibility-plane kinematic models. 066-1 DFT; 066-3 geometry; 066-5 T2."""

from .constants import ARCSEC_TO_RAD, C_LIGHT_M_S, C_LIGHT_KM_S, F_REST_CO21_HZ
from .decisions import load_decision_index, requires
from .geometry import (
    catalogue_ba,
    galaxy_to_sky,
    incline,
    inclination_deg,
    inclination_prior_half_width_deg,
    inclination_rad,
    pa_seed_deg,
    pa_seed_rad,
    rotate_by_pa,
    sky_to_galaxy,
)
from .transforms import dft_numpy, nufft2_degrid, uv_wavelengths

__version__ = "0.1.0.dev0"

__all__ = [
    "ARCSEC_TO_RAD",
    "C_LIGHT_M_S",
    "C_LIGHT_KM_S",
    "F_REST_CO21_HZ",
    "catalogue_ba",
    "dft_numpy",
    "galaxy_to_sky",
    "incline",
    "inclination_deg",
    "inclination_prior_half_width_deg",
    "inclination_rad",
    "load_decision_index",
    "nufft2_degrid",
    "pa_seed_deg",
    "pa_seed_rad",
    "requires",
    "rotate_by_pa",
    "sky_to_galaxy",
    "uv_wavelengths",
    "__version__",
]
