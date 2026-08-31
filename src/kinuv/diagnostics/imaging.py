"""Match a Jy/pixel sky cube to an imaging cube and form moments / PV.

The Stage B product is the PB-attenuated sky model (native channels, no
CLEAN beam). KILOGAS v1.3 cubes are restored brightness temperature. This
module is the image-plane comparison, not a second fit.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import fftconvolve

from kinuv.constants import C_LIGHT_KM_S, FWHM_TO_SIGMA, F_REST_CO21_HZ
from kinuv.io.vis import radio_to_optical_kms
from kinuv.response.primary_beam import primary_beam
from kinuv.template.wiener import k_to_jy_per_beam

PB_FLOOR = 0.05


def spectral_axis_kms(header) -> np.ndarray:
    """Channel-centre velocity in km/s from a 3-D FITS header."""
    n = int(header["NAXIS3"])
    v = float(header["CRVAL3"]) + (
        np.arange(1, n + 1, dtype=np.float64) - float(header["CRPIX3"])
    ) * float(header["CDELT3"])
    unit = str(header.get("CUNIT3", "km/s")).lower().replace(" ", "")
    if unit in {"m/s", "ms-1"}:
        v = v / 1.0e3
    return v


def radio_header_velocity_kms(header) -> np.ndarray:
    """Optical km/s for a ``VRAD`` header (radio km/s or m/s in the card)."""
    return radio_to_optical_kms(spectral_axis_kms(header))


def flux_weighted_velocity(spec, vel_kms) -> float:
    """Centroid of ``max(spec, 0)`` on ``vel_kms``. NaN if no positive flux."""
    s = np.clip(np.asarray(spec, dtype=np.float64), 0.0, None)
    v = np.asarray(vel_kms, dtype=np.float64)
    w = float(np.sum(s))
    if w <= 0.0:
        return float("nan")
    return float(np.sum(s * v) / w)


def spectral_wcs_report(header, *, label: str = "") -> dict:
    """RESTFRQ / CRPIX3 / CRVAL3 / CTYPE3 / SPECSYS for a 3-D header."""
    n = int(header.get("NAXIS3", 0) or 0)
    crval = float(header["CRVAL3"]) if n else None
    crpix = float(header["CRPIX3"]) if n else None
    cdelt = float(header["CDELT3"]) if n else None
    v = spectral_axis_kms(header) if n else np.array([])
    ctype = str(header.get("CTYPE3", ""))
    radio = ctype.upper().startswith("VRAD")
    v_opt = radio_to_optical_kms(v) if (radio and v.size) else v
    rec = {
        "label": label,
        "ctype3": ctype,
        "cunit3": str(header.get("CUNIT3", "")),
        "crval3": crval,
        "crpix3": crpix,
        "cdelt3": cdelt,
        "restfrq_hz": float(header["RESTFRQ"]) if "RESTFRQ" in header else None,
        "specsys": str(header.get("SPECSYS", "")),
        "naxis3": n,
        "header_axis": "radio" if radio else "optical",
        "vel_chan0_kms": float(v[0]) if v.size else None,
        "vel_chanN_kms": float(v[-1]) if v.size else None,
        "vel_optical_chan0_kms": float(v_opt[0]) if v_opt.size else None,
        "vel_optical_chanN_kms": float(v_opt[-1]) if v_opt.size else None,
    }
    return rec


def rebin_spectrum(cube, v_in_kms, v_out_kms, dv_out_kms):
    """Overlap-average ``cube`` (nv, ny, nx) onto ``v_out`` channel windows."""
    cub = np.asarray(cube, dtype=np.float64)
    v_in = np.asarray(v_in_kms, dtype=np.float64)
    v_out = np.asarray(v_out_kms, dtype=np.float64)
    dv_out = float(abs(dv_out_kms))
    if cub.shape[0] != v_in.size:
        raise ValueError("spectral axis length mismatch")
    dv_in = float(np.median(np.abs(np.diff(v_in)))) if v_in.size > 1 else dv_out
    lo_in, hi_in = v_in - 0.5 * dv_in, v_in + 0.5 * dv_in
    out = np.empty((v_out.size,) + cub.shape[1:], dtype=np.float64)
    for k, v0 in enumerate(v_out):
        lo, hi = v0 - 0.5 * dv_out, v0 + 0.5 * dv_out
        w = np.clip(np.minimum(hi_in, hi) - np.maximum(lo_in, lo), 0.0, None)
        sw = float(w.sum())
        if sw <= 0.0:
            i = int(np.argmin(np.abs(v_in - v0)))
            out[k] = cub[i]
        else:
            out[k] = np.tensordot(w / sw, cub, axes=(0, 0))
    return out


def restoring_beam_kernel(cell_arcsec, bmaj_arcsec, bmin_arcsec, bpa_deg, cdelt1):
    """Unit-sum elliptical Gaussian in pixel coords. ``bpa_deg`` is E of N."""
    cell = float(cell_arcsec)
    n = int(max(15, np.ceil(8.0 * float(bmaj_arcsec) / cell)))
    if n % 2 == 0:
        n += 1
    off = (np.arange(n, dtype=np.float64) - (n - 1) * 0.5) * cell
    x, y = np.meshgrid(off, off, indexing="xy")
    east = -x if float(cdelt1) < 0.0 else x
    north = y
    pa = np.deg2rad(float(bpa_deg))
    xm = east * np.sin(pa) + north * np.cos(pa)
    ym = east * np.cos(pa) - north * np.sin(pa)
    sj = float(bmaj_arcsec) * FWHM_TO_SIGMA
    sn = float(bmin_arcsec) * FWHM_TO_SIGMA
    ker = np.exp(-0.5 * ((xm / sj) ** 2 + (ym / sn) ** 2))
    s = float(ker.sum())
    if s <= 0.0:
        raise ValueError("restoring-beam kernel is empty")
    return ker / s


def jy_per_pixel_to_k(jy_pix, cell_arcsec, nu_hz, bmaj_arcsec, bmin_arcsec):
    """Smoothed Jy/pixel → Rayleigh–Jeans K for the given restoring beam."""
    omega_beam = (
        np.pi * float(bmaj_arcsec) * float(bmin_arcsec) / (4.0 * np.log(2.0))
    )
    omega_pix = float(cell_arcsec) ** 2
    jy_beam = np.asarray(jy_pix, dtype=np.float64) * (omega_beam / omega_pix)
    scale = float(k_to_jy_per_beam(1.0, nu_hz, bmaj_arcsec, bmin_arcsec))
    return jy_beam / scale


def _convolve_channels(cube, kernel):
    out = np.empty_like(cube)
    for i in range(cube.shape[0]):
        out[i] = fftconvolve(cube[i], kernel, mode="same")
    return out


def _celestial_extent_pix(header):
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def regrid_to_header(cube, hdr_in, hdr_out):
    """Bilinear brightness resample onto ``hdr_out`` celestial grid."""
    from astropy.wcs import WCS

    cub = np.asarray(cube, dtype=np.float64)
    w_in = WCS(hdr_in).celestial
    w_out = WCS(hdr_out).celestial
    ny, nx = _celestial_extent_pix(hdr_out)
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = w_out.all_pix2world(np.stack([xx.ravel(), yy.ravel()], axis=1), 0)
    pix_in = w_in.all_world2pix(world, 0)
    x_in = pix_in[:, 0].reshape(ny, nx)
    y_in = pix_in[:, 1].reshape(ny, nx)
    coords = np.array([y_in, x_in])
    out = np.empty((cub.shape[0], ny, nx), dtype=np.float64)
    for i in range(cub.shape[0]):
        out[i] = map_coordinates(cub[i], coords, order=1, cval=np.nan)
    return out


def _model_sky_axes_arcsec(header):
    nx = int(header["NAXIS1"])
    ny = int(header["NAXIS2"])
    cell = abs(float(header["CDELT1"])) * 3600.0
    x = (np.arange(nx, dtype=np.float64) + 1.0 - float(header["CRPIX1"])) * (
        float(header["CDELT1"]) * 3600.0
    )
    y = (np.arange(ny, dtype=np.float64) + 1.0 - float(header["CRPIX2"])) * (
        float(header["CDELT2"]) * 3600.0
    )
    # CDELT1 < 0 → +x is west; primary_beam wants +x east.
    x_east = -x if float(header["CDELT1"]) < 0.0 else x
    return x_east, y, cell


def match_model_to_imaging(
    model_jy_pix,
    model_header,
    data_header,
    *,
    nu_hz=None,
    undo_pb: bool = True,
):
    """Sky cube (nv, ny, nx) Jy/pixel → K on the imaging WCS and channels."""
    cub = np.asarray(model_jy_pix, dtype=np.float64)
    v_model = radio_header_velocity_kms(model_header)
    v_data = spectral_axis_kms(data_header)
    dv_data = float(abs(data_header["CDELT3"]))
    unit = str(data_header.get("CUNIT3", "km/s")).lower().replace(" ", "")
    if unit in {"m/s", "ms-1"}:
        dv_data /= 1.0e3
    if undo_pb:
        x_east, y_north, _ = _model_sky_axes_arcsec(model_header)
        nu = float(nu_hz) if nu_hz is not None else float(
            data_header.get("RESTFRQ", F_REST_CO21_HZ)
        )
        att = primary_beam(x_east, y_north, nu)
        cub = cub / np.maximum(att[None, :, :], PB_FLOOR)
    cub = rebin_spectrum(cub, v_model, v_data, dv_data)
    cell = abs(float(model_header["CDELT1"])) * 3600.0
    bmaj = float(data_header["BMAJ"]) * 3600.0
    bmin = float(data_header["BMIN"]) * 3600.0
    bpa = float(data_header["BPA"])
    ker = restoring_beam_kernel(cell, bmaj, bmin, bpa, model_header["CDELT1"])
    cub = _convolve_channels(cub, ker)
    rest = float(data_header.get("RESTFRQ", F_REST_CO21_HZ))
    v_mid = float(np.median(v_data))
    nu_use = float(nu_hz) if nu_hz is not None else rest / (1.0 + v_mid / C_LIGHT_KM_S)
    cub = jy_per_pixel_to_k(cub, cell, nu_use, bmaj, bmin)
    return regrid_to_header(cub, model_header, data_header), v_data, dv_data


def masked_moments(cube_k, vel_kms, mask, dv_kms):
    """Moments 0 (K km/s), 1 (km/s), 2 (km/s) with a 3-D mask."""
    t = np.asarray(cube_k, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    v = np.asarray(vel_kms, dtype=np.float64)[:, None, None]
    dv = float(abs(dv_kms))
    t = np.where(m, t, 0.0)
    w = np.sum(t, axis=0)
    m0 = np.sum(t * dv, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.where(w > 0.0, np.sum(t * v, axis=0) / w, np.nan)
        var = np.where(w > 0.0, np.sum(t * (v - m1) ** 2, axis=0) / w, np.nan)
    m2 = np.sqrt(np.maximum(var, 0.0))
    empty = ~np.any(m, axis=0)
    m0[empty] = np.nan
    m1[empty] = np.nan
    m2[empty] = np.nan
    return m0, m1, m2


def offset_world(ra_deg, dec_deg, dx_east_arcsec, dy_north_arcsec):
    """Phase-centre RA/Dec plus east/north offsets [arcsec]."""
    dra = -(float(dx_east_arcsec) / 3600.0) / np.cos(np.deg2rad(float(dec_deg)))
    ddec = float(dy_north_arcsec) / 3600.0
    return float(ra_deg) + dra, float(dec_deg) + ddec


def pv_diagram(
    cube,
    header,
    ra_deg,
    dec_deg,
    pa_deg,
    length_arcsec,
    width_arcsec,
):
    """PV along receding PA (E of N). Positive offset is the receding side.

    ``cube`` is (nv, ny, nx). Returns ``(pv, offsets_arcsec)``.
    """
    from astropy.wcs import WCS

    cub = np.asarray(cube, dtype=np.float64)
    w = WCS(header).celestial
    cell = abs(float(header["CDELT1"])) * 3600.0
    n_off = max(8, int(round(float(length_arcsec) / cell)))
    n_w = max(1, int(round(float(width_arcsec) / cell)))
    offsets = (np.arange(n_off) - (n_off - 1) * 0.5) * cell
    width = (np.arange(n_w) - (n_w - 1) * 0.5) * cell
    pa = np.deg2rad(float(pa_deg))
    nv = cub.shape[0]
    z = np.arange(nv, dtype=np.float64)
    pv = np.zeros((nv, n_off), dtype=np.float64)
    for i, s in enumerate(offsets):
        acc = np.zeros(nv, dtype=np.float64)
        n_ok = 0
        for woff in width:
            east = s * np.sin(pa) + woff * np.cos(pa)
            north = s * np.cos(pa) - woff * np.sin(pa)
            ra, dec = offset_world(ra_deg, dec_deg, east, north)
            x, y = w.all_world2pix(ra, dec, 0)
            spec = map_coordinates(
                cub,
                np.vstack([z, np.full(nv, float(y)), np.full(nv, float(x))]),
                order=1,
                cval=np.nan,
            )
            if np.any(np.isfinite(spec)):
                acc += np.nan_to_num(spec, nan=0.0)
                n_ok += 1
        pv[:, i] = acc / max(n_ok, 1)
    return pv, offsets
