"""S1 sub-beam recovery metrics. Not a second likelihood.

Stage A arctan inject on the production Hann+bin path. Image-plane numbers
are a CLEAN-beam comparator, not χ².
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from kinuv.constants import F_REST_CO21_HZ
from kinuv.geometry import inclination_deg, inclination_rad
from kinuv.profiles.rotation import BMAJ_ICO_ARCSEC
from kinuv.response.spectral import hann_then_bin

PIPELINE_KERNEL = "hann_then_bin"
S1_RT_ARCSEC = 0.25
S1_SIGMA_KMS = 8.0
S1_V0_KMS = 250.0
S1_RT_BOUNDS_ARCSEC = (0.05, 15.0)
R_EVAL_OVER_BMAJ = 0.25
PASS_V0_KMS = 10.0
PASS_SIGMA_KMS = 2.0
POL_LABEL = "XX"

CANFAR_NPZ = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz"
)
CANFAR_ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CANFAR_CUBE_30 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
CANFAR_CUBE_10 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
    "KGAS66_clipped_cube.fits"
)
MAP_DIR = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map"
)


def assert_hann_bin_operator() -> str:
    """Fail unless the production SPECRESP operator is Hann+bin."""
    if hann_then_bin.__module__ != "kinuv.response.spectral":
        raise AssertionError(
            f"pipeline_kernel must be kinuv.response.spectral.hann_then_bin; "
            f"got {hann_then_bin.__module__}.{hann_then_bin.__name__}"
        )
    if hann_then_bin.__name__ != "hann_then_bin":
        raise AssertionError("pipeline_kernel != hann_then_bin")
    return PIPELINE_KERNEL


def r_eval_arcsec(bmaj_arcsec: float = BMAJ_ICO_ARCSEC) -> float:
    return float(R_EVAL_OVER_BMAJ) * float(bmaj_arcsec)


def inner_slope_arctan(v0_kms, r_t_arcsec, r_arcsec) -> float:
    """dV_c/dr of ``V_0 (2/π) arctan(r/r_t)`` [km/s / arcsec]."""
    rt = float(r_t_arcsec)
    r = float(r_arcsec)
    if rt <= 0.0:
        raise ValueError("r_t_arcsec must be positive")
    return float(v0_kms) * (2.0 / np.pi) * rt / (rt * rt + r * r)


def add_xx_noise(model_vis, weights, s, rng):
    """Complex Gaussian noise with var(Re)=var(Im)=1/(s w) (XX empirical s)."""
    m = np.asarray(model_vis)
    w = np.asarray(weights, dtype=np.float64)
    std = np.zeros_like(w)
    good = w > 0.0
    std[good] = 1.0 / np.sqrt(float(s) * w[good])
    noise = std * (
        rng.standard_normal(m.shape) + 1j * rng.standard_normal(m.shape)
    )
    return m + noise


def inject_vis(data, truth, template, grid, *, rng=None):
    """Hann+bin model visibilities; optional XX noise. Does not mutate ``data``."""
    from kinuv.infer.map import predict_binned

    assert_hann_bin_operator()
    model = predict_binned(data, truth, template, grid)
    if rng is None:
        vis = np.asarray(model)
    else:
        vis = add_xx_noise(model, data.weights, data.s, rng)
    return replace(data, vis=np.asarray(vis, dtype=np.complex128)), np.asarray(model)


def params_from_map(rec) -> dict[str, float]:
    return {
        "flux": float(rec.flux),
        "pa_deg": float(rec.pa_deg),
        "vsys_kms": float(rec.vsys_kms),
        "gas_sigma_kms": float(rec.gas_sigma_kms),
        "dx_arcsec": float(rec.dx_arcsec),
        "dy_arcsec": float(rec.dy_arcsec),
        "v0_kms": float(rec.v0_kms),
        "r_t_arcsec": float(rec.r_t_arcsec),
    }


def vis_recovery_table(truth, rec, bmaj_arcsec: float = BMAJ_ICO_ARCSEC):
    r_e = r_eval_arcsec(bmaj_arcsec)
    slope_t = inner_slope_arctan(truth["v0_kms"], truth["r_t_arcsec"], r_e)
    slope_r = inner_slope_arctan(rec.v0_kms, rec.r_t_arcsec, r_e)
    d_v0 = float(rec.v0_kms - truth["v0_kms"])
    d_sig = float(rec.gas_sigma_kms - truth["gas_sigma_kms"])
    return {
        "r_eval_arcsec": r_e,
        "bmaj_arcsec": float(bmaj_arcsec),
        "slope_truth_kms_per_arcsec": slope_t,
        "slope_vis_kms_per_arcsec": slope_r,
        "delta_v0_kms": d_v0,
        "delta_sigma_kms": d_sig,
        "delta_slope_kms_per_arcsec": float(slope_r - slope_t),
        "pass_v0": bool(abs(d_v0) < PASS_V0_KMS),
        "pass_sigma": bool(abs(d_sig) < PASS_SIGMA_KMS),
        "i_held_fixed": True,
        "i_deg": float(inclination_deg()),
        "h_z_in_model": False,
        "pol": POL_LABEL,
        "s": float(rec.s),
        "pipeline_kernel": PIPELINE_KERNEL,
    }


def leftover_chi2(data, model):
    """Per-row and per-channel residual χ² of the real MAP (not the mock)."""
    r = np.asarray(data.vis) - np.asarray(model)
    mag2 = r.real.astype(np.float64) ** 2 + r.imag.astype(np.float64) ** 2
    w = np.asarray(data.weights, dtype=np.float64)
    s = float(data.s)
    per_row = s * np.sum(w * mag2, axis=1)
    per_chan = s * np.sum(w * mag2, axis=0)
    baseline_m = np.hypot(
        np.asarray(data.u_m, dtype=np.float64),
        np.asarray(data.v_m, dtype=np.float64),
    )
    return baseline_m, per_row, np.asarray(data.vel, dtype=np.float64), per_chan


def chi2_slice_pa_sigma(data, template, grid, params, pa, sigma):
    from kinuv.infer.map import predict_binned
    from kinuv.likelihood.chi2 import chi2

    out = np.empty((len(sigma), len(pa)), dtype=np.float64)
    for i, sig in enumerate(sigma):
        for j, pdeg in enumerate(pa):
            p = dict(params)
            p["pa_deg"] = float(pdeg)
            p["gas_sigma_kms"] = float(sig)
            model = predict_binned(data, p, template, grid)
            out[i, j] = chi2(data.vis, model, data.weights, data.s)
    return out


def chi2_slice_sigma_inc(data, template, grid, params, sigma, i_deg):
    """Diagnostic: i unfrozen for the scan only. Not an official MAP."""
    from kinuv.infer.map import predict_binned
    from kinuv.likelihood.chi2 import chi2

    out = np.empty((len(i_deg), len(sigma)), dtype=np.float64)
    for i, ideg in enumerate(i_deg):
        i_rad = np.radians(float(ideg))
        for j, sig in enumerate(sigma):
            p = dict(params)
            p["gas_sigma_kms"] = float(sig)
            model = predict_binned(data, p, template, grid, i_rad=i_rad)
            out[i, j] = chi2(data.vis, model, data.weights, data.s)
    return out


def chi2_slice_pa_rt(data, template, grid, params, pa, r_t):
    from kinuv.infer.map import predict_binned
    from kinuv.likelihood.chi2 import chi2

    out = np.empty((len(r_t), len(pa)), dtype=np.float64)
    for i, rt in enumerate(r_t):
        for j, pdeg in enumerate(pa):
            p = dict(params)
            p["pa_deg"] = float(pdeg)
            p["r_t_arcsec"] = float(rt)
            model = predict_binned(data, p, template, grid)
            out[i, j] = chi2(data.vis, model, data.weights, data.s)
    return out


def quadratic_cov_2d(x, y, chi2_grid, x0, y0):
    """Laplace covariance from a χ² grid. ``ln L = −χ²/2`` so cov = 2 H_χ²^{-1}."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(chi2_grid, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    dx = (xx - float(x0)).ravel()
    dy = (yy - float(y0)).ravel()
    zz = z.ravel()
    a = np.column_stack(
        [np.ones(dx.size), dx, dy, dx * dx, dx * dy, dy * dy]
    )
    coeff, *_ = np.linalg.lstsq(a, zz, rcond=None)
    c20, c11, c02 = float(coeff[3]), float(coeff[4]), float(coeff[5])
    hess = np.array([[2.0 * c20, c11], [c11, 2.0 * c02]], dtype=np.float64)
    try:
        cov = 2.0 * np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    v0, v1 = float(cov[0, 0]), float(cov[1, 1])
    corr = float(cov[0, 1] / np.sqrt(v0 * v1)) if v0 > 0.0 and v1 > 0.0 else float("nan")
    return {
        "cov": cov.tolist(),
        "corr": corr,
        "hess_chi2": hess.tolist(),
        "chi2_min_grid": float(np.min(z)),
    }


def model_cube_header(grid, vel_kms, ico_hdr):
    """WCS for a Jy/pixel sky cube; NAXIS1 increases west (CDELT1<0)."""
    from astropy.wcs import WCS

    w = WCS(naxis=3)
    w.wcs.crpix = [grid.nx // 2 + 1, grid.ny // 2 + 1, 1.0]
    cdelt = float(grid.cell_arcsec) / 3600.0
    dv = float(np.median(np.diff(vel_kms))) if vel_kms.size > 1 else 1.0
    w.wcs.cdelt = np.array([-cdelt, cdelt, dv])
    w.wcs.crval = [
        float(ico_hdr.get("CRVAL1", 0.0)),
        float(ico_hdr.get("CRVAL2", 0.0)),
        float(vel_kms[0]),
    ]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "VRAD"]
    w.wcs.cunit = ["deg", "deg", "km/s"]
    w.wcs.specsys = "LSRK"
    hdr = w.to_header()
    hdr["BUNIT"] = "Jy/pixel"
    hdr["RESTFRQ"] = (float(F_REST_CO21_HZ), "CO(2-1) rest frequency [Hz]")
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = int(grid.nx)
    hdr["NAXIS2"] = int(grid.ny)
    hdr["NAXIS3"] = int(np.asarray(vel_kms).size)
    return hdr


def sky_cube_fits(cube_yxv, grid, vel_kms, ico_hdr):
    """``(ny,nx,nv)`` Jy/pixel +x east → FITS ``(nv,ny,nx)`` CDELT1<0."""
    data = np.moveaxis(np.asarray(cube_yxv, dtype=np.float64), 2, 0)
    data = np.flip(data, axis=2)
    return data, model_cube_header(grid, vel_kms, ico_hdr)


def cube_inner_kinematics(
    cube_k,
    header,
    vel_kms,
    dv_kms,
    pa_deg,
    vsys_kms,
    i_rad,
    r_eval_arcsec,
):
    """Beam-convolved M1 inner slope and median M2 (apparent σ)."""
    from kinuv.diagnostics.imaging import masked_moments
    from kinuv.geometry import sky_to_galaxy

    cub = np.asarray(cube_k, dtype=np.float64)
    msk = cub > 0.0
    m0, m1, m2 = masked_moments(cub, vel_kms, msk, dv_kms)
    nx = int(header["NAXIS1"])
    ny = int(header["NAXIS2"])
    x = (
        np.arange(nx, dtype=np.float64) + 1.0 - float(header["CRPIX1"])
    ) * (float(header["CDELT1"]) * 3600.0)
    y = (
        np.arange(ny, dtype=np.float64) + 1.0 - float(header["CRPIX2"])
    ) * (float(header["CDELT2"]) * 3600.0)
    east = -x if float(header["CDELT1"]) < 0.0 else x
    xe, yn = np.meshgrid(east, y, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, np.radians(float(pa_deg)), float(i_rad))
    radius = np.hypot(xg, yg)
    cos_th = np.divide(xg, radius, out=np.zeros_like(radius), where=radius > 0.0)
    peak = float(np.nanmax(m0))
    disk = np.isfinite(m0) & np.isfinite(m1) & (m0 > 0.05 * peak)
    major = disk & (np.abs(yg) < 0.6) & (radius > 0.05)
    sini = float(np.sin(i_rad))
    with np.errstate(invalid="ignore", divide="ignore"):
        vc = np.where(
            major & (np.abs(cos_th) > 0.3),
            (m1 - float(vsys_kms)) / (sini * cos_th),
            np.nan,
        )
    r_pix = radius[np.isfinite(vc)]
    v_pix = vc[np.isfinite(vc)]
    inner = (r_pix > 0.08) & (r_pix < 0.6 * BMAJ_ICO_ARCSEC)
    if int(inner.sum()) >= 4:
        coeff = np.polyfit(r_pix[inner], v_pix[inner], 1)
        slope = float(coeff[0])
    else:
        slope = float("nan")
    r0 = float(r_eval_arcsec)
    near = np.isfinite(vc) & (np.abs(radius - r0) < 0.15)
    vc_at = float(np.nanmedian(vc[near])) if np.any(near) else float("nan")
    inner_m2 = disk & (radius < BMAJ_ICO_ARCSEC)
    sig_app = float(np.nanmedian(m2[inner_m2])) if np.any(inner_m2) else float("nan")
    return {
        "slope_cube_kms_per_arcsec": slope,
        "vc_at_r_eval_kms": vc_at,
        "sigma_m2_kms": sig_app,
        "n_major_inner": int(inner.sum()) if r_pix.size else 0,
    }


def dirty_cube_from_truth(truth, template, grid, freqs_hz, ico_hdr, imaging_hdr):
    """Intrinsic sky cube → restoring-beam cube on the 10 km/s WCS."""
    from kinuv.constants import freq_to_velocity_kms
    from kinuv.diagnostics.imaging import match_model_to_imaging
    from kinuv.forward.model import sky_cube
    from kinuv.io.vis import radio_to_optical_kms

    cube = sky_cube(
        template,
        grid,
        freqs_hz,
        flux=truth["flux"],
        pa_rad=np.radians(truth["pa_deg"]),
        vsys_kms=truth["vsys_kms"],
        dx_arcsec=truth["dx_arcsec"],
        dy_arcsec=truth["dy_arcsec"],
        gas_sigma_kms=truth["gas_sigma_kms"],
        v0_kms=truth["v0_kms"],
        r_t_arcsec=truth["r_t_arcsec"],
        i_rad=inclination_rad(),
    )
    vel = freq_to_velocity_kms(freqs_hz)
    fits_c, model_hdr = sky_cube_fits(cube, grid, vel, ico_hdr)
    matched, v_data, dv = match_model_to_imaging(fits_c, model_hdr, imaging_hdr)
    kin = cube_inner_kinematics(
        matched,
        imaging_hdr,
        v_data,
        dv,
        truth["pa_deg"],
        radio_to_optical_kms(truth["vsys_kms"]),
        inclination_rad(),
        r_eval_arcsec(),
    )
    return matched, v_data, dv, kin
