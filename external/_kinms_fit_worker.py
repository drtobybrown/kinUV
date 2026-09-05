#!/usr/bin/env python3
"""Isolated KinMS cube fit worker (external_fitters venv only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import differential_evolution, minimize


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
    """KinMS native Jy/beam → Rayleigh–Jeans K on the imaging cube."""
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


def _make_kinms_cube(params, hdr, n_samps):
    from kinms import KinMS

    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    nv = int(hdr["NAXIS3"])
    cdelt = abs(float(hdr.get("CDELT1", -5.5e-5))) * 3600.0
    dv = abs(float(hdr.get("CDELT3", 10.0)))
    if dv > 200:
        dv = dv / 1000.0
    v0, rt, pa, inc, vsys_opt, gs = params
    r = np.linspace(0.02, 4.5, 50)
    vcirc = float(v0) * (2.0 / np.pi) * np.arctan(r / max(float(rt), 1e-3))
    sb = np.exp(-0.5 * (r / 1.8) ** 2)
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
        sbProf=sb,
        sbRad=r,
        velProf=vcirc,
        velRad=r,
        gasSigma=float(gs),
        intFlux=1.0,
    )
    # KinMS returns (nx, ny, nv); FITS getdata is (nv, ny, nx).
    return np.transpose(np.asarray(cube, dtype=np.float64), (2, 1, 0))


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

    x0 = np.array(
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

    def objective(x, *, n_samps=1.2e5):
        n_eval["n"] += 1
        try:
            cube_jy = _make_kinms_cube(x, hdr, n_samps=n_samps)
            model_k = _jy_beam_to_k(cube_jy, hdr, vel)
            chi2, _ = _scaled_cube_chi2(model_k, data_k, mask3d)
            return chi2
        except Exception as exc:
            last_err[:] = [f"{type(exc).__name__}: {exc}"]
            return 1.0e30

    de = differential_evolution(
        lambda x: objective(x, n_samps=1.2e5),
        bnds,
        seed=42,
        maxiter=10,
        popsize=5,
        polish=False,
        workers=1,
    )
    res = minimize(
        lambda x: objective(x, n_samps=2e5),
        de.x,
        method="L-BFGS-B",
        bounds=bnds,
        options={"maxfun": 30, "ftol": 1e-9},
    )
    if float(de.fun) < float(res.fun):
        res.x = de.x
        res.fun = de.fun
    final_jy = _make_kinms_cube(res.x, hdr, n_samps=3e5)
    final_k = _jy_beam_to_k(final_jy, hdr, vel)
    chi2, flux_scale = _scaled_cube_chi2(final_k, data_k, mask3d)
    final_jy_scaled = final_jy * flux_scale

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
    out_hdr["ORIGIN"] = "KinMS independent cube fit; catalogue priors only"
    fits.PrimaryHDU(data=final_jy_scaled.astype(np.float32), header=out_hdr).writeto(
        work / "model_cube.fits",
        overwrite=True,
    )
    result = {
        "success": bool(res.fun < 1.0e29),
        "message": f"DE nfev={de.nfev}; polish {res.message}",
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
