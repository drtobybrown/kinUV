"""kinUV — 066-1 DFT; 066-2 Wiener+PB; 066-3 geometry; 066-5 T2; 066-6/7."""

from .constants import ARCSEC_TO_RAD, C_LIGHT_M_S, C_LIGHT_KM_S, F_REST_CO21_HZ
from .decisions import load_decision_index, requires
from .forward.model import los_velocity, predict_vis
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
from .io.vis import VisData, load_kgas066
from .likelihood.chi2 import chi2, chi2_zero, delta_chi2
from .response.primary_beam import fwhm_pb_arcsec, primary_beam
from .response.spectral import hann_then_bin
from .template.wiener import ico_to_template, k_to_jy_per_beam
from .transforms import dft_numpy, nufft2_degrid, uv_wavelengths

__version__ = "0.1.0.dev0"

__all__ = [
    "ARCSEC_TO_RAD",
    "C_LIGHT_M_S",
    "C_LIGHT_KM_S",
    "F_REST_CO21_HZ",
    "VisData",
    "catalogue_ba",
    "chi2",
    "chi2_zero",
    "delta_chi2",
    "dft_numpy",
    "fwhm_pb_arcsec",
    "galaxy_to_sky",
    "hann_then_bin",
    "ico_to_template",
    "incline",
    "inclination_deg",
    "inclination_prior_half_width_deg",
    "inclination_rad",
    "k_to_jy_per_beam",
    "load_decision_index",
    "load_kgas066",
    "los_velocity",
    "nufft2_degrid",
    "pa_seed_deg",
    "pa_seed_rad",
    "predict_vis",
    "primary_beam",
    "requires",
    "rotate_by_pa",
    "sky_to_galaxy",
    "uv_wavelengths",
    "__version__",
]
