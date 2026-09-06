"""Observed-frame constants used by the visibility modeling engine.

The inference path uses angular sky coordinates, observed frequencies, and
line-of-sight velocities. Physical-radius conversion and cosmology are
downstream concerns in :mod:`kinuv.postprocess`; they do not belong in the
forward model or likelihood dependency graph.
"""

from __future__ import annotations

import numpy as np

#: Speed of light, exact SI definition [m/s].
C_LIGHT_M_S = 299_792_458.0

#: Speed of light [km/s].
C_LIGHT_KM_S = C_LIGHT_M_S / 1.0e3

#: Radians per arcsecond.
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

#: CO(2-1) rest frequency [Hz].
F_REST_CO21_HZ = 230.538e9

#: Conversion from a Gaussian FWHM to its standard deviation.
FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))

#: Boltzmann constant, SI exact [J/K].
K_BOLTZMANN_J_K = 1.380649e-23

#: Jansky in SI [W m⁻² Hz⁻¹].
JY_W_M2_HZ = 1.0e-26


def freq_to_velocity_kms(freq_hz, f_rest_hz: float = F_REST_CO21_HZ):
    """Radio-convention velocity [km/s] for observed frequencies [Hz].

    .. math:: v = c\\,(\\nu_{\\rm rest} - \\nu) / \\nu_{\\rm rest}
    """
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(freq_hz)
    freq_hz = xp.asarray(freq_hz)
    return C_LIGHT_KM_S * (f_rest_hz - freq_hz) / f_rest_hz


def velocity_to_freq_hz(v_kms, f_rest_hz: float = F_REST_CO21_HZ):
    """Inverse of :func:`freq_to_velocity_kms`."""
    v_kms = np.asarray(v_kms, dtype=np.float64)
    return f_rest_hz * (1.0 - v_kms / C_LIGHT_KM_S)
