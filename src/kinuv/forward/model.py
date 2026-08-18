"""Circular thin-disk native-channel visibilities (066-7).

``I_sky(x,y,ν) = flux × I_template(x−dx, y−dy) × Gaussian(v(ν)−v_los; σ)``
with Stage A arctan ``V_c`` (DEC-066-VC). Then Fourier-shift the template,
``A`` at the **phase centre** (DEC-066-PB / SHIFT), FINUFFT T2 (DEC-066-GRID).
Do not apply a visibility phase ramp after PB.
"""

from __future__ import annotations

import numpy as np

from kinuv.constants import freq_to_velocity_kms
from kinuv.decisions import requires
from kinuv.geometry import inclination_rad, sky_to_galaxy
from kinuv.profiles.rotation import (
    CALIBRATION_RT_ARCSEC,
    CALIBRATION_V0_KM_S,
    DV_CHAN_NATIVE_KM_S,
    arctan_vc,
)
from kinuv.response.primary_beam import attenuate, primary_beam
from kinuv.template.fourier_shift import fourier_shift
from kinuv.transforms.grid import ImageGrid
from kinuv.transforms.nufft import nufft2_degrid

from .sb import image_grid_xy_arcsec

VSYS_SEED_KM_S = 8299.563
GAS_SIGMA_SEED_KM_S = 10.0
INJECT_OFFSET_ARCSEC = 0.3
LINE_V_MIN_KM_S = 8034.0
LINE_V_MAX_KM_S = 8536.0


def channel_width_kms(freqs_hz) -> float:
    """Native channel width from the frequency grid (radio vs rest)."""
    vel = freq_to_velocity_kms(freqs_hz)
    if vel.size < 2:
        return float(DV_CHAN_NATIVE_KM_S)
    return float(np.median(np.abs(np.diff(vel))))


@requires("DEC-066-PA", "DEC-066-INC", "DEC-066-VC")
def los_velocity(
    x_east_arcsec,
    y_north_arcsec,
    pa_rad,
    i_rad,
    vsys_kms,
    v0_kms: float = CALIBRATION_V0_KM_S,
    r_t_arcsec: float = CALIBRATION_RT_ARCSEC,
):
    """``v_los = vsys + V_c(R) sin(i) cos(θ)``; +x is receding (DEC-066-PA)."""
    xg, yg = sky_to_galaxy(x_east_arcsec, y_north_arcsec, pa_rad, i_rad)
    radius = np.hypot(xg, yg)
    vc = arctan_vc(radius, v0_kms, r_t_arcsec)
    cos_th = np.divide(xg, radius, out=np.zeros_like(radius), where=radius > 0.0)
    return float(vsys_kms) + vc * np.sin(i_rad) * cos_th


def _gaussian_pdf(v_kms, v_los, sigma_kms):
    sig = float(sigma_kms)
    delta = (v_kms - v_los) / sig
    return np.exp(-0.5 * delta**2) / (sig * np.sqrt(2.0 * np.pi))


@requires("DEC-066-PB", "DEC-066-SHIFT", "DEC-066-VC", "DEC-066-PA", "DEC-066-INC")
def sky_cube(
    template,
    grid: ImageGrid,
    freqs_hz,
    *,
    flux,
    pa_rad,
    vsys_kms,
    dx_arcsec,
    dy_arcsec,
    gas_sigma_kms,
    v0_kms: float = CALIBRATION_V0_KM_S,
    r_t_arcsec: float = CALIBRATION_RT_ARCSEC,
    i_rad=None,
):
    """Attenuated sky cube ``(ny, nx, n_chan)`` in Jy/pixel (native channels).

    Spatial interpolator is :func:`fourier_shift` on the template. ``A`` is
    evaluated at the phase centre, not ``(dx, dy)``. Kinematics follow the
    galaxy: ``v_los`` is evaluated at ``(x−dx, y−dy)``.
    """
    sb = np.asarray(template, dtype=np.float64)
    if sb.shape != (grid.ny, grid.nx):
        raise ValueError(f"template {sb.shape} != grid {(grid.ny, grid.nx)}")
    i_use = inclination_rad() if i_rad is None else float(i_rad)
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    vel = freq_to_velocity_kms(freqs)
    dv = channel_width_kms(freqs)
    shifted = fourier_shift(sb, dx_arcsec, dy_arcsec, grid.cell_arcsec)
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    v_los = los_velocity(
        xe - float(dx_arcsec),
        yn - float(dy_arcsec),
        pa_rad,
        i_use,
        vsys_kms,
        v0_kms,
        r_t_arcsec,
    )
    phi = _gaussian_pdf(vel[None, None, :], v_los[:, :, None], gas_sigma_kms)
    d_omega = grid.cell_arcsec**2
    cube = float(flux) * shifted[:, :, None] * d_omega * phi * dv
    nu_mid = float(np.median(freqs))
    att = attenuate(np.ones((grid.ny, grid.nx), dtype=np.float64), x, y, nu_mid)
    # ``primary_beam`` is the envelope; keep a named use so PB stays in the path.
    _ = primary_beam(x, y, nu_mid)
    return cube * att[:, :, None]


@requires(
    "DEC-066-PB",
    "DEC-066-SHIFT",
    "DEC-066-GRID",
    "DEC-066-VC",
    "DEC-066-PA",
    "DEC-066-INC",
)
def predict_vis(
    u_m,
    v_m,
    freqs_hz,
    *,
    flux,
    pa_rad,
    vsys_kms,
    dx_arcsec,
    dy_arcsec,
    gas_sigma_kms,
    template,
    grid: ImageGrid,
    v0_kms: float = CALIBRATION_V0_KM_S,
    r_t_arcsec: float = CALIBRATION_RT_ARCSEC,
    i_rad=None,
    eps: float = 1e-8,
):
    """Native-channel model visibilities ``(n_row, n_chan)`` complex128.

    No visibility phase ramp after PB (DEC-066-SHIFT / PB).
    """
    cube = sky_cube(
        template,
        grid,
        freqs_hz,
        flux=flux,
        pa_rad=pa_rad,
        vsys_kms=vsys_kms,
        dx_arcsec=dx_arcsec,
        dy_arcsec=dy_arcsec,
        gas_sigma_kms=gas_sigma_kms,
        v0_kms=v0_kms,
        r_t_arcsec=r_t_arcsec,
        i_rad=i_rad,
    )
    return nufft2_degrid(grid, cube, u_m, v_m, freqs_hz, eps=eps)
