"""Wiener deconvolution of the restoring beam (DEC-066-SB)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.constants import (
    ARCSEC_TO_RAD,
    C_LIGHT_M_S,
    FWHM_TO_SIGMA,
    JY_W_M2_HZ,
    K_BOLTZMANN_J_K,
)
from kinuv.decisions import requires
from kinuv.template.fftpad import crop_centered, default_pad_n, embed_centered
from kinuv.template.resample import resample_flux_conserving, sky_axes

CENTROID_TOL_ARCSEC = 0.01
TAPER_FLOOR = 0.05
SIGNED_FLUX_FRACTION = 0.5


class SignedFluxError(ValueError):
    """Clip skipped and ``∫ I ≤ 0.5 ∫ |I|`` on the mask."""


@dataclass(frozen=True)
class WienerTemplate:
    """Unit-integral SB template after Wiener, gate, optional resample."""

    sb: np.ndarray
    clipped: bool
    k_wiener: float
    cell_arcsec: float
    mask: np.ndarray
    signed: np.ndarray


@requires("DEC-066-SB")
def k_to_jy_per_beam(t_k, nu_hz, bmaj_arcsec, bmin_arcsec):
    """Rayleigh–Jeans K → Jy/beam at the **observed** frequency.

    ``S = T × (2 k ν² / c²) × π θ_maj θ_min / (4 ln 2)``, with the SI→Jy
    factor ``1 / JY_W_M2_HZ``. Recompute; do not hardcode 0.063 Jy/K.
    """
    theta_maj = float(bmaj_arcsec) * ARCSEC_TO_RAD
    theta_min = float(bmin_arcsec) * ARCSEC_TO_RAD
    omega = np.pi * theta_maj * theta_min / (4.0 * np.log(2.0))
    si = (2.0 * K_BOLTZMANN_J_K * float(nu_hz) ** 2 / C_LIGHT_M_S**2) * omega
    return np.asarray(t_k, dtype=np.float64) * (si / JY_W_M2_HZ)


@requires("DEC-066-SB")
def restoring_beam_ft(u, v, bmaj_arcsec, bmin_arcsec, bpa_deg):
    """Analytic restoring-beam FT, ``|B̃(0)| = 1``. ``u,v`` in cycles/arcsec."""
    sigma_maj = float(bmaj_arcsec) * FWHM_TO_SIGMA
    sigma_min = float(bmin_arcsec) * FWHM_TO_SIGMA
    bpa = np.deg2rad(float(bpa_deg))
    c, s = np.cos(bpa), np.sin(bpa)
    up = u * c + v * s
    vp = -u * s + v * c
    return np.exp(-2.0 * np.pi**2 * ((sigma_maj * up) ** 2 + (sigma_min * vp) ** 2))


def _finite_zero(image) -> np.ndarray:
    out = np.array(image, dtype=np.float64, copy=True)
    out[~np.isfinite(out)] = 0.0
    return out


def empty_corner_rms(image, corner_frac: float = 0.12):
    """RMS of finite pixels in the four corner boxes, or ``None`` if too few."""
    a = np.asarray(image, dtype=np.float64)
    ny, nx = a.shape
    ny_c = max(3, int(np.floor(corner_frac * ny)))
    nx_c = max(3, int(np.floor(corner_frac * nx)))
    blocks = (
        a[:ny_c, :nx_c],
        a[:ny_c, -nx_c:],
        a[-ny_c:, :nx_c],
        a[-ny_c:, -nx_c:],
    )
    vals = np.concatenate([b[np.isfinite(b)].ravel() for b in blocks])
    if vals.size < 8:
        return None
    return float(np.std(vals, ddof=1))


def _beam_on_fft_grid(pad_y, pad_x, cell_arcsec, bmaj, bmin, bpa):
    uy = np.fft.fftfreq(int(pad_y), d=float(cell_arcsec))[:, None]
    ux = np.fft.fftfreq(int(pad_x), d=float(cell_arcsec))[None, :]
    return restoring_beam_ft(ux, uy, bmaj, bmin, bpa)


@requires("DEC-066-SB")
def convolve_restoring_beam(
    image,
    cell_arcsec,
    bmaj_arcsec,
    bmin_arcsec,
    bpa_deg,
    pad_n=None,
):
    """Sum-preserving convolution with the unit-DC restoring Gaussian."""
    img = _finite_zero(image)
    ny, nx = img.shape
    pad = int(pad_n) if pad_n is not None else default_pad_n(max(ny, nx))
    padded, y0, x0 = embed_centered(img, pad, pad)
    beam = _beam_on_fft_grid(pad, pad, cell_arcsec, bmaj_arcsec, bmin_arcsec, bpa_deg)
    ft = np.fft.fft2(np.fft.ifftshift(padded)) * beam
    recon = np.fft.fftshift(np.fft.ifft2(ft)).real
    return crop_centered(recon, ny, nx, y0, x0)


@requires("DEC-066-SB")
def wiener_deconvolve(
    image,
    cell_arcsec,
    bmaj_arcsec,
    bmin_arcsec,
    bpa_deg,
    k_wiener,
    pad_n=None,
    taper=TAPER_FLOOR,
):
    """Undo the restoring beam. Crop to the input stamp after iFFT."""
    img = _finite_zero(image)
    ny, nx = img.shape
    pad = int(pad_n) if pad_n is not None else default_pad_n(max(ny, nx))
    padded, y0, x0 = embed_centered(img, pad, pad)
    beam = _beam_on_fft_grid(pad, pad, cell_arcsec, bmaj_arcsec, bmin_arcsec, bpa_deg)
    k = float(k_wiener)
    ft = np.fft.fft2(np.fft.ifftshift(padded))
    denom = np.abs(beam) ** 2 + k
    deconv = ft * np.conj(beam) / denom
    deconv[np.abs(beam) < float(taper)] = 0.0
    recon = np.fft.fftshift(np.fft.ifft2(deconv)).real
    return crop_centered(recon, ny, nx, y0, x0)


def _xy_grids(ny, nx, cell_arcsec):
    y = sky_axes(ny, cell_arcsec)
    x = sky_axes(nx, cell_arcsec)
    return np.meshgrid(x, y)


def flux_weighted_centroid(image, cell_arcsec, mask=None):
    """``⟨x⟩ = ∫ x I dΩ / ∫ I dΩ`` [arcsec]. ``(nan, nan)`` if the integral vanishes."""
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    X, Y = _xy_grids(ny, nx, cell_arcsec)
    w = img if mask is None else img * np.asarray(mask, dtype=bool)
    tot = float(np.sum(w))
    if not np.isfinite(tot) or abs(tot) < 1e-30:
        return float("nan"), float("nan")
    return float(np.sum(X * w) / tot), float(np.sum(Y * w) / tot)


@requires("DEC-066-SB")
def clip_if_centroid_stable(
    image,
    cell_arcsec,
    mask=None,
    tol_arcsec=CENTROID_TOL_ARCSEC,
):
    """``max(I, 0)`` only if the centroid moves by ``< 0.01″`` absolute."""
    img = np.asarray(image, dtype=np.float64)
    clipped = np.maximum(img, 0.0)
    cx_s, cy_s = flux_weighted_centroid(img, cell_arcsec, mask)
    cx_c, cy_c = flux_weighted_centroid(clipped, cell_arcsec, mask)
    if not (np.isfinite(cx_s) and np.isfinite(cx_c)):
        return img.copy(), False
    shift = float(np.hypot(cx_c - cx_s, cy_c - cy_s))
    if shift < float(tol_arcsec):
        return clipped, True
    return img.copy(), False


@requires("DEC-066-SB")
def assert_signed_flux_gate(image, mask):
    """Abort if ``∫_mask I ≤ 0.5 ∫_mask |I|``."""
    img = np.asarray(image, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    s = float(np.sum(img[m]))
    a = float(np.sum(np.abs(img[m])))
    if a <= 0.0 or s <= SIGNED_FLUX_FRACTION * a:
        raise SignedFluxError(
            f"signed-flux gate failed: integral {s} vs 0.5 * abs-integral {a}"
        )


def _apply_mask(image, mask) -> np.ndarray:
    out = np.array(image, dtype=np.float64, copy=True)
    out[~np.asarray(mask, dtype=bool)] = 0.0
    return out


@requires("DEC-066-SB")
def normalise_unit_integral(image, cell_arcsec, mask, clipped: bool):
    """Shape only: unit ``∫ I dΩ`` on the mask. Signed gate if clip was skipped."""
    masked = _apply_mask(image, mask)
    if not clipped:
        assert_signed_flux_gate(masked, mask)
    d_omega = float(cell_arcsec) ** 2
    integral = float(np.sum(masked) * d_omega)
    if abs(integral) < 1e-30:
        raise SignedFluxError("template integral is zero; cannot normalise")
    return masked / integral


@requires("DEC-066-SB")
def ico_to_template(
    ico,
    cell_arcsec,
    nu_obs_hz,
    bmaj_arcsec,
    bmin_arcsec,
    bpa_deg,
    *,
    mask=None,
    model_cell_arcsec=None,
    pad_n=None,
    k_wiener=None,
    sigma_empty=None,
    units="K",
):
    """K km/s (or Jy/beam km/s) → unit-integral Wiener template.

    ``nu_obs_hz`` is the observed line frequency, not rest CO.
    """
    raw = np.asarray(ico, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("ico must be 2-D")
    if units == "K":
        jy = k_to_jy_per_beam(raw, nu_obs_hz, bmaj_arcsec, bmin_arcsec)
    elif units == "Jy":
        jy = raw.copy()
    else:
        raise ValueError("units must be 'K' or 'Jy'")
    jy = np.where(np.isfinite(raw), jy, np.nan)

    if mask is None:
        work_mask = np.isfinite(raw)
        if not np.any(work_mask):
            work_mask = np.ones(raw.shape, dtype=bool)
    else:
        work_mask = np.asarray(mask, dtype=bool)

    if k_wiener is None:
        sig = sigma_empty if sigma_empty is not None else empty_corner_rms(jy)
        if sig is None:
            raise ValueError("empty-corner rms unavailable; pass sigma_empty or k_wiener")
        peak = float(np.nanmax(np.abs(jy)))
        if peak <= 0.0:
            raise ValueError("I_peak is non-positive")
        k_wiener = (float(sig) / peak) ** 2

    signed = wiener_deconvolve(
        jy,
        cell_arcsec,
        bmaj_arcsec,
        bmin_arcsec,
        bpa_deg,
        k_wiener,
        pad_n=pad_n,
    )
    gated, clipped = clip_if_centroid_stable(signed, cell_arcsec, mask=work_mask)
    gated = _apply_mask(gated, work_mask)

    cell_out = float(cell_arcsec)
    mask_out = work_mask
    if model_cell_arcsec is not None and not np.isclose(
        float(model_cell_arcsec), float(cell_arcsec)
    ):
        gated = resample_flux_conserving(gated, cell_arcsec, model_cell_arcsec)
        mask_f = resample_flux_conserving(
            work_mask.astype(np.float64), cell_arcsec, model_cell_arcsec
        )
        mask_out = mask_f > 0.5
        cell_out = float(model_cell_arcsec)
        gated = _apply_mask(gated, mask_out)

    sb = normalise_unit_integral(gated, cell_out, mask_out, clipped)
    return WienerTemplate(
        sb=sb,
        clipped=clipped,
        k_wiener=float(k_wiener),
        cell_arcsec=cell_out,
        mask=mask_out,
        signed=signed,
    )
