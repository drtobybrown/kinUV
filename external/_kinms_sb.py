"""Empirical surface brightness helpers for KinMS (external venv only)."""

from __future__ import annotations

import numpy as np


def sky_axes_arcsec(header) -> tuple[np.ndarray, np.ndarray]:
    """East/north offsets [arcsec] for each pixel (2-D mesh)."""
    nx = int(header["NAXIS1"])
    ny = int(header["NAXIS2"])
    x = (np.arange(nx, dtype=np.float64) + 1.0 - float(header["CRPIX1"])) * (
        float(header["CDELT1"]) * 3600.0
    )
    y = (np.arange(ny, dtype=np.float64) + 1.0 - float(header["CRPIX2"])) * (
        float(header["CDELT2"]) * 3600.0
    )
    east = -x if float(header["CDELT1"]) < 0.0 else x
    xe, yn = np.meshgrid(east, y, indexing="xy")
    return xe, yn


def m0_from_cube(cube_k: np.ndarray, vel_kms: np.ndarray, mask3d: np.ndarray, dv_kms: float):
    """Moment-0 map [K km/s] from a masked cube."""
    del vel_kms
    t = np.where(mask3d, np.asarray(cube_k, dtype=np.float64), 0.0)
    return np.sum(t, axis=0) * float(abs(dv_kms))


def m0_extent_arcsec(m0: np.ndarray, header, *, frac: float = 0.90) -> float:
    """Radius (arcsec) enclosing ``frac`` of total M0 flux."""
    xe, yn = sky_axes_arcsec(header)
    f = np.clip(np.asarray(m0, dtype=np.float64), 0.0, None)
    tot = float(f.sum())
    if tot <= 0.0:
        return 0.0
    r = np.hypot(xe, yn)
    order = np.argsort(r.ravel())
    cum = np.cumsum(f.ravel()[order]) / tot
    k = int(np.searchsorted(cum, frac))
    k = min(k, order.size - 1)
    return float(r.ravel()[order[k]])


def sample_sky_clouds_from_m0(
    m0: np.ndarray,
    header,
    *,
    n_clouds: int = 100000,
    flux_floor_frac: float = 0.001,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``n_clouds`` sky-plane positions weighted by the M0 map.

    Returns east/north offsets [arcsec] and flux weights proportional to M0.
    Draws use replacement so ``n_clouds`` may exceed the number of bright pixels.
    """
    rng = rng or np.random.default_rng(66)
    m0 = np.clip(np.asarray(m0, dtype=np.float64), 0.0, None)
    peak = float(np.nanmax(m0))
    if peak <= 0.0:
        raise ValueError("M0 map is empty")
    floor = flux_floor_frac * peak
    xe, yn = sky_axes_arcsec(header)
    w = m0.ravel()
    pos = w > floor
    if int(pos.sum()) < 100:
        pos = w > 0.0
    x_flat = xe.ravel()[pos]
    y_flat = yn.ravel()[pos]
    f_flat = w[pos]
    prob = f_flat / float(f_flat.sum())
    idx = rng.choice(x_flat.size, size=int(n_clouds), replace=True, p=prob)
    x_sky = x_flat[idx].astype(np.float64)
    y_sky = y_flat[idx].astype(np.float64)
    flux = np.maximum(f_flat[idx], floor).astype(np.float64)
    return x_sky, y_sky, flux


def deproject_sky_to_kinms_disk(
    x_sky: np.ndarray,
    y_sky: np.ndarray,
    pa_deg: float,
    inc_deg: float,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
) -> np.ndarray:
    """Deproject sky M0 samples to face-on disk coords for KinMS ``inClouds``.

    Uses the kinUV convention: rotate sky (E, N) by PA to (major, minor), then
    deproject the minor axis by ``1/cos(i)``. This matches ``sky_to_galaxy``.
    """
    dx = np.asarray(x_sky, dtype=np.float64) - float(x0)
    dy = np.asarray(y_sky, dtype=np.float64) - float(y0)
    pa = np.radians(float(pa_deg))
    inc = np.radians(float(inc_deg))
    s, c = np.sin(pa), np.cos(pa)
    x_maj = dx * s + dy * c
    y_min = dx * c - dy * s
    ci = np.cos(inc)
    if abs(ci) < 1e-6:
        ci = 1e-6
    out = np.empty((dx.size, 3), dtype=np.float64)
    out[:, 0] = x_maj
    out[:, 1] = y_min / ci
    out[:, 2] = 0.0
    return out


def radial_sb_from_m0(
    m0: np.ndarray,
    header,
    pa_deg: float,
    inc_deg: float,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    r_max_arcsec: float = 12.0,
    n_rad: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged I(R) in elliptical annuli (galaxy plane).

    KinMS ``sbProf``/``sbRad`` expect this intrinsic radial profile; KinMS
    then samples face-on clouds and applies its own inc/PA projection.
    """
    xe, yn = sky_axes_arcsec(header)
    dx = xe - float(x0)
    dy = yn - float(y0)
    pa = np.radians(float(pa_deg))
    inc = np.radians(float(inc_deg))
    s, c = np.sin(pa), np.cos(pa)
    x_maj = dx * s + dy * c
    y_min = dx * c - dy * s
    ci = max(abs(np.cos(inc)), 1e-6)
    r = np.hypot(x_maj, y_min / ci)
    flux = np.clip(np.asarray(m0, dtype=np.float64), 0.0, None)
    r_bins = np.linspace(0.05, float(r_max_arcsec), int(n_rad))
    prof = np.empty_like(r_bins)
    dr = 0.5 * (r_bins[1] - r_bins[0])
    for k, rb in enumerate(r_bins):
        m = (r >= rb - dr) & (r < rb + dr) & (flux > 0.0)
        prof[k] = float(np.nanmean(flux[m])) if np.any(m) else 0.0
    peak = float(np.nanmax(prof)) if np.any(prof > 0) else 1.0
    prof = np.maximum(prof, 1.0e-12 * peak)
    return r_bins, prof


def inclouds_from_m0(
    m0: np.ndarray,
    header,
    *,
    n_clouds: int = 100000,
    flux_floor_frac: float = 0.001,
    pa_deg: float = 0.0,
    inc_deg: float = 45.0,
    x0: float = 0.0,
    y0: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample sky M0 and deproject to face-on ``inClouds`` for KinMS."""
    x_sky, y_sky, flux = sample_sky_clouds_from_m0(
        m0, header, n_clouds=n_clouds, flux_floor_frac=flux_floor_frac, rng=rng
    )
    ic = deproject_sky_to_kinms_disk(x_sky, y_sky, pa_deg, inc_deg, x0=x0, y0=y0)
    return ic, flux
