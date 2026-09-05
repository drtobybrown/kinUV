#!/usr/bin/env python3
"""KinMS cube fit with sbProf/sbRad SB (external_fitters venv only).

KinMS inClouds are face-on disk coordinates; KinMS then applies inclination
and PA itself. Passing deprojected sky samples caused a double projection.
This worker lets KinMS generate cloudlets from an elliptical-annulus I(R)
measured on the CLEAN M0 map.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import differential_evolution, minimize

from _kinms_sb import m0_extent_arcsec, m0_from_cube, radial_sb_from_m0


def _vel_optical_kms(header) -> np.ndarray:
    n = int(header["NAXIS3"])
    v = float(header["CRVAL3"]) + (
        np.arange(1, n + 1, dtype=np.float64) - float(header["CRPIX3"])
    ) * float(header["CDELT3"])
    unit = str(header.get("CUNIT3", "km/s")).lower().replace(" ", "")
    if unit in {"m/s", "ms-1"}:
        v = v / 1.0e3
    return v


def _jy_beam_to_k(jy_beam, header, vel_kms):
    from astropy.constants import c

    bmaj = float(header["BMAJ"]) * 3600.0
    bmin = float(header["BMIN"]) * 3600.0
    rest = float(header.get("RESTFRQ", 230.538e9))
    v_mid = float(np.median(vel_kms))
    nu = rest / (1.0 + v_mid / (c.value / 1e3))
    theta_maj = bmaj * np.pi / (180.0 * 3600.0)
    theta_min = bmin * np.pi / (180.0 * 3600.0)
    omega = np.pi * theta_maj * theta_min / (4.0 * np.log(2.0))
    kb = 1.380649e-23
    jy_w = 1.0e-26
    scale = (2.0 * kb * nu**2 / c.value**2) * omega / jy_w
    return np.asarray(jy_beam, dtype=np.float64) / scale


def kinms_to_fits(cube) -> np.ndarray:
    """KinMS (nx, ny, nv) → FITS getdata order (nv, ny, nx)."""
    return np.transpose(np.asarray(cube, dtype=np.float64), (2, 1, 0))


def _make_kinms_cube(params, hdr, sb_rad, sb_prof, *, x0, y0, v_cube_mid, n_samps):
    from kinms import KinMS

    v0, rt, pa, inc, vsys_opt, gs = params
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    nv = int(hdr["NAXIS3"])
    cdelt = abs(float(hdr.get("CDELT1", -5.5e-5))) * 3600.0
    dv = abs(float(hdr.get("CDELT3", 10.0)))
    if dv > 200:
        dv = dv / 1000.0
    r = np.linspace(0.02, max(4.5, float(rt) * 3.0, float(np.max(sb_rad))), 60)
    vcirc = float(v0) * (2.0 / np.pi) * np.arctan(r / max(float(rt), 1e-3))
    v_offset = float(vsys_opt) - float(v_cube_mid)
    model = KinMS(
        xs=nx * cdelt,
        ys=ny * cdelt,
        vs=nv * dv,
        cellSize=cdelt,
        dv=dv,
        beamSize=[
            float(hdr.get("BMAJ", 0.0002)) * 3600.0,
            float(hdr.get("BMIN", 0.0002)) * 3600.0,
            float(hdr.get("BPA", 0.0)),
        ],
        nSamps=float(n_samps),
    )
    cube = model.model_cube(
        inc=float(inc),
        posAng=float(pa),
        vSys=float(vsys_opt),
        vOffset=v_offset,
        sbProf=np.asarray(sb_prof, dtype=np.float64),
        sbRad=np.asarray(sb_rad, dtype=np.float64),
        velProf=vcirc,
        velRad=r,
        gasSigma=float(gs),
        intFlux=1.0,
        phaseCent=[float(x0), float(y0)],
        vPhaseCent=[0.0, 0.0],
    )
    return kinms_to_fits(cube)


def _scaled_cube_chi2(model_k, data_k, mask3d):
    m = np.asarray(mask3d, dtype=bool) & np.isfinite(model_k) & np.isfinite(data_k)
    if not np.any(m):
        return 1.0e30, 0.0
    mod = model_k[m]
    dat = data_k[m]
    denom = float(np.dot(mod, mod))
    if denom <= 0.0:
        return 1.0e30, 0.0
    scale = float(np.dot(dat, mod) / denom)
    resid = scale * model_k - data_k
    return float(np.sum(resid[m] ** 2)), scale


def main() -> int:
    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())
    cube_path = Path(cfg["cube"])
    work = Path(cfg["work"])
    work.mkdir(parents=True, exist_ok=True)
    init = cfg["init"]
    bounds = cfg["bounds"]
    hdr = fits.getheader(cube_path)
    data_k = np.asarray(fits.getdata(cube_path), dtype=np.float64)
    mask3d = np.ones_like(data_k, dtype=bool)
    if cfg.get("mask"):
        mask3d = np.asarray(fits.getdata(cfg["mask"]), dtype=np.float64) > 0.5
    vel = _vel_optical_kms(hdr)
    dv = abs(float(hdr["CDELT3"]))
    if dv > 200:
        dv = dv / 1000.0
    sb_path = cfg.get("sb_cube")
    if sb_path:
        sb_k = np.asarray(fits.getdata(sb_path), dtype=np.float64)
        sb_hdr = fits.getheader(sb_path)
        sb_vel = _vel_optical_kms(sb_hdr)
        sb_dv = abs(float(sb_hdr["CDELT3"]))
        if sb_dv > 200:
            sb_dv = sb_dv / 1000.0
        sb_mask = mask3d if sb_k.shape == data_k.shape else np.ones_like(sb_k, dtype=bool)
        m0 = m0_from_cube(sb_k, sb_vel, sb_mask, sb_dv)
        m0_hdr = sb_hdr
    else:
        m0 = m0_from_cube(data_k, vel, mask3d, dv)
        m0_hdr = hdr
    phase = cfg.get("phase_center") or [init.get("dx_arcsec", 0.0), init.get("dy_arcsec", 0.0)]
    x0, y0 = float(phase[0]), float(phase[1])
    n_samps = int(cfg.get("n_clouds", 100000))
    n_samps_de = int(cfg.get("n_clouds_de", min(n_samps, 50000)))
    de_workers = int(cfg.get("de_workers", min(12, os.cpu_count() or 4)))
    seed_pa = float(cfg.get("sb_pa_deg", init["pa_deg"]))
    seed_inc = float(cfg.get("sb_inc_deg", init["i_deg"]))
    m0_data_extent = m0_extent_arcsec(m0, m0_hdr, frac=0.90)
    sb_rad, sb_prof = radial_sb_from_m0(
        m0, m0_hdr, seed_pa, seed_inc, x0=x0, y0=y0
    )
    np.savez(work / "sb_profile.npz", sb_rad=sb_rad, sb_prof=sb_prof, m0=m0)
    v_cube_mid = float(vel[len(vel) // 2])

    bnds = [
        bounds["v0_kms"],
        bounds["r_t_arcsec"],
        bounds["pa_deg"],
        bounds["i_deg"],
        bounds["vsys_optical_kms"],
        bounds["gas_sigma_kms"],
    ]
    n_eval = {"n": 0}
    last_err: list[str] = []

    def objective(x, *, n_use=n_samps_de):
        n_eval["n"] += 1
        try:
            cube_jy = _make_kinms_cube(
                x, hdr, sb_rad, sb_prof, x0=x0, y0=y0, v_cube_mid=v_cube_mid, n_samps=n_use
            )
            model_k = _jy_beam_to_k(cube_jy, hdr, vel)
            chi2, _ = _scaled_cube_chi2(model_k, data_k, mask3d)
            return chi2
        except Exception as exc:
            last_err[:] = [f"{type(exc).__name__}: {exc}"]
            return 1.0e30

    x0par = np.array(
        [
            init["v0_kms"],
            init["r_t_arcsec"],
            init["pa_deg"],
            init["i_deg"],
            init["vsys_optical_kms"],
            init["gas_sigma_kms"],
        ],
        dtype=np.float64,
    )
    de = differential_evolution(
        objective,
        bnds,
        seed=42,
        maxiter=12,
        popsize=6,
        polish=False,
        workers=1,
        x0=x0par,
        init="latinhypercube",
    )
    res = minimize(
        lambda x: objective(x, n_use=n_samps),
        de.x,
        method="L-BFGS-B",
        bounds=bnds,
        options={"maxfun": 40, "ftol": 1e-9},
    )
    if float(de.fun) < float(res.fun):
        res.x = de.x
        res.fun = de.fun

    final_jy = _make_kinms_cube(
        res.x, hdr, sb_rad, sb_prof, x0=x0, y0=y0, v_cube_mid=v_cube_mid, n_samps=n_samps
    )
    final_k = _jy_beam_to_k(final_jy, hdr, vel)
    chi2, flux_scale = _scaled_cube_chi2(final_k, data_k, mask3d)
    final_jy_scaled = final_jy * flux_scale
    model_m0 = m0_from_cube(final_k, vel, mask3d, dv)
    model_m0_extent = m0_extent_arcsec(model_m0, hdr, frac=0.90)

    fitted = {
        "v0_kms": float(res.x[0]),
        "r_t_arcsec": float(res.x[1]),
        "pa_deg": float(res.x[2]),
        "i_deg": float(res.x[3]),
        "vsys_optical_kms": float(res.x[4]),
        "gas_sigma_kms": float(res.x[5]),
        "flux_scale": float(flux_scale),
    }
    out_hdr = hdr.copy()
    out_hdr["BUNIT"] = "Jy/beam"
    out_hdr["ORIGIN"] = "KinMS sbProf/sbRad fit; KinMS owns projection"
    fits.PrimaryHDU(data=final_jy_scaled.astype(np.float32), header=out_hdr).writeto(
        work / "model_cube.fits",
        overwrite=True,
    )
    result = {
        "success": bool(res.fun < 1.0e29),
        "message": (
            f"sbProf nSamps={n_samps}; DE nfev={de.nfev}; polish {res.message}"
        ),
        "sb_mode": "sbProf_from_M0_elliptical",
        "n_clouds": n_samps,
        "n_clouds_de": n_samps_de,
        "de_workers": de_workers,
        "phase_center_arcsec": [x0, y0],
        "sb_pa_deg": seed_pa,
        "sb_inc_deg": seed_inc,
        "v_cube_mid_kms": v_cube_mid,
        "m0_data_extent_arcsec": m0_data_extent,
        "m0_model_extent_arcsec": model_m0_extent,
        "extent_ok": bool(model_m0_extent >= 0.85 * m0_data_extent),
        "fitted": fitted,
        "chi2_cube": float(chi2 if chi2 < 1.0e29 else res.fun),
        "nfev": int(n_eval["n"]),
        "init": init,
        "bounds": bounds,
        "last_error": last_err[0] if last_err else None,
    }
    (work / "kinms_fit_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
