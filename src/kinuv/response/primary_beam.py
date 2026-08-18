"""ALMA 12 m Gaussian primary beam (DEC-066-PB)."""

from __future__ import annotations

import numpy as np

from kinuv.constants import ARCSEC_TO_RAD, C_LIGHT_M_S
from kinuv.decisions import requires
from kinuv.template.fourier_shift import fourier_shift
from kinuv.template.resample import sky_axes

D_ANT_M = 12.0
FWHM_PB_FACTOR = 1.13


@requires("DEC-066-PB")
def fwhm_pb_arcsec(nu_hz, d_ant_m: float = D_ANT_M) -> float:
    """``FWHM_PB(ν) = 1.13 λ / D``. Never ``56.6″ / ν_GHz``."""
    lam = C_LIGHT_M_S / float(nu_hz)
    fwhm_rad = FWHM_PB_FACTOR * lam / float(d_ant_m)
    return float(fwhm_rad / ARCSEC_TO_RAD)


@requires("DEC-066-PB")
def primary_beam(
    x_arcsec,
    y_arcsec,
    nu_hz,
    x_phase_arcsec: float = 0.0,
    y_phase_arcsec: float = 0.0,
    d_ant_m: float = D_ANT_M,
):
    """``A = exp(−4 ln 2 · r² / FWHM²)`` with ``r`` from the phase centre.

    ``x,y`` may be 1-D axes (meshgridded) or 2-D maps. ``A`` does not follow
    ``(dx, dy)``.
    """
    x = np.asarray(x_arcsec, dtype=np.float64)
    y = np.asarray(y_arcsec, dtype=np.float64)
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)
    fwhm = fwhm_pb_arcsec(nu_hz, d_ant_m)
    r2 = (x - float(x_phase_arcsec)) ** 2 + (y - float(y_phase_arcsec)) ** 2
    return np.exp(-4.0 * np.log(2.0) * r2 / fwhm**2)


@requires("DEC-066-PB")
def attenuate(
    image,
    x_arcsec,
    y_arcsec,
    nu_hz,
    x_phase_arcsec: float = 0.0,
    y_phase_arcsec: float = 0.0,
    d_ant_m: float = D_ANT_M,
):
    """Multiply sky SB by ``A`` at the phase centre. Does not touch visibilities."""
    a = primary_beam(
        x_arcsec,
        y_arcsec,
        nu_hz,
        x_phase_arcsec=x_phase_arcsec,
        y_phase_arcsec=y_phase_arcsec,
        d_ant_m=d_ant_m,
    )
    img = np.asarray(image, dtype=np.float64)
    if a.shape != img.shape:
        raise ValueError("primary beam and image shapes differ")
    return img * a


@requires("DEC-066-PB")
def translate_then_attenuate(
    image,
    dx_arcsec,
    dy_arcsec,
    cell_arcsec,
    nu_hz,
    x_phase_arcsec: float = 0.0,
    y_phase_arcsec: float = 0.0,
    d_ant_m: float = D_ANT_M,
    pad_n=None,
    x_arcsec=None,
    y_arcsec=None,
):
    """``I_sky = I(x − dx, y − dy)`` then ``A`` at the phase centre, then stop.

    Order is mandatory. Do not apply ``A`` and then a visibility phase ramp.
    """
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    x = sky_axes(nx, cell_arcsec) if x_arcsec is None else np.asarray(x_arcsec)
    y = sky_axes(ny, cell_arcsec) if y_arcsec is None else np.asarray(y_arcsec)
    sky = fourier_shift(img, dx_arcsec, dy_arcsec, cell_arcsec, pad_n=pad_n)
    att = attenuate(
        sky,
        x,
        y,
        nu_hz,
        x_phase_arcsec=x_phase_arcsec,
        y_phase_arcsec=y_phase_arcsec,
        d_ant_m=d_ant_m,
    )
    return sky, att
