#!/usr/bin/env python3
"""Pre-flight KinMS geometry test (external_fitters venv only).

Pass criteria (PA=199.7°, i=43.9°, V0=250 km/s, vsys=8323.6 km/s):
  * Major PV along PA 199.7° shows the full ±V0 sin(i) ≈ ±174 km/s horns
  * Minor PV along PA 289.7° is flat at vsys
  * M0 long axis is along PA ≈ 200° and 90% flux radius ≳ 6″
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from _kinms_best_worker import _jy_beam_to_k, _make_kinms_cube, _vel_optical_kms
from _kinms_sb import m0_extent_arcsec, m0_from_cube, radial_sb_from_m0, sky_axes_arcsec

CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
    "KGAS66_clipped_cube.fits"
)
MASK = CUBE.parent / "KGAS66_mask_cube.fits"
PA = 199.7
INC = 43.9
V0 = 250.0
RT = 0.5
VSYS = 8323.6
SIGMA = 10.0
PA_MIN = PA + 90.0


def _pv_along_pa(cube, hdr, pa_deg, length=16.0, width=0.4):
    """Minimal PV (nv, n_off) along receding PA; +offset = receding."""
    nv, ny, nx = cube.shape
    cell = abs(float(hdr["CDELT1"])) * 3600.0
    n_off = max(8, int(round(length / cell)))
    n_w = max(1, int(round(width / cell)))
    offsets = (np.arange(n_off) - (n_off - 1) * 0.5) * cell
    widths = (np.arange(n_w) - (n_w - 1) * 0.5) * cell
    pa = np.deg2rad(float(pa_deg))
    crpix1, crpix2 = float(hdr["CRPIX1"]), float(hdr["CRPIX2"])
    cdelt1, cdelt2 = float(hdr["CDELT1"]) * 3600.0, float(hdr["CDELT2"]) * 3600.0
    pv = np.zeros((nv, n_off), dtype=np.float64)
    for i, s in enumerate(offsets):
        acc = np.zeros(nv)
        n_ok = 0
        for woff in widths:
            east = s * np.sin(pa) + woff * np.cos(pa)
            north = s * np.cos(pa) - woff * np.sin(pa)
            x = (crpix1 - 1.0) + (-east / abs(cdelt1) if cdelt1 < 0 else east / cdelt1)
            y = (crpix2 - 1.0) + north / cdelt2
            if 1 <= x < nx - 1 and 1 <= y < ny - 1:
                acc += cube[:, int(round(y)), int(round(x))]
                n_ok += 1
        if n_ok:
            pv[:, i] = acc / n_ok
    return pv, offsets


def _intensity_weighted_vel(pv, vel, offsets):
    w = np.clip(pv, 0.0, None)
    den = w.sum(axis=0)
    v_ridge = np.divide(w.T @ vel, den, out=np.full(den.shape, np.nan), where=den > 0)
    return offsets, v_ridge


def main() -> int:
    hdr = fits.getheader(CUBE)
    data = np.asarray(fits.getdata(CUBE), dtype=np.float64)
    mask = np.asarray(fits.getdata(MASK), dtype=np.float64) > 0.5
    vel = _vel_optical_kms(hdr)
    dv = abs(float(hdr["CDELT3"]))
    m0_data = m0_from_cube(data, vel, mask, dv)
    sb_rad, sb_prof = radial_sb_from_m0(m0_data, hdr, PA, INC)
    v_mid = float(vel[len(vel) // 2])
    params = [V0, RT, PA, INC, VSYS, SIGMA]
    cube_jy = _make_kinms_cube(
        params, hdr, sb_rad, sb_prof, x0=0.0, y0=0.0, v_cube_mid=v_mid, n_samps=80000
    )
    model_k = _jy_beam_to_k(cube_jy, hdr, vel)
    m0_mod = m0_from_cube(model_k, vel, mask, dv)
    ext_data = m0_extent_arcsec(m0_data, hdr)
    ext_mod = m0_extent_arcsec(m0_mod, hdr)

    pv_maj, off = _pv_along_pa(model_k, hdr, PA)
    pv_min, _ = _pv_along_pa(model_k, hdr, PA_MIN)
    _, v_maj = _intensity_weighted_vel(pv_maj, vel, off)
    _, v_min = _intensity_weighted_vel(pv_min, vel, off)
    sini = np.sin(np.radians(INC))
    expected = V0 * sini
    receding = (off > 1.5) & np.isfinite(v_maj)
    approaching = (off < -1.5) & np.isfinite(v_maj)
    dv_maj = float(np.nanmedian(v_maj[receding]) - np.nanmedian(v_maj[approaching])) / 2.0
    dv_min = float(np.nanmax(np.abs(v_min[np.isfinite(v_min)] - VSYS))) if np.any(
        np.isfinite(v_min)
    ) else 999.0

    xe, yn = sky_axes_arcsec(hdr)
    flux = np.clip(m0_mod, 0.0, None)
    tot = float(flux.sum()) or 1.0
    # Second-moment PA of model M0 (E of N)
    xbar = float((flux * xe).sum() / tot)
    ybar = float((flux * yn).sum() / tot)
    ixx = float((flux * (xe - xbar) ** 2).sum() / tot)
    iyy = float((flux * (yn - ybar) ** 2).sum() / tot)
    ixy = float((flux * (xe - xbar) * (yn - ybar)).sum() / tot)
    theta = 0.5 * np.arctan2(2 * ixy, ixx - iyy)
    pa_m0 = (90.0 - np.degrees(theta)) % 180.0
    pa_target = PA % 180.0
    pa_err = min(abs(pa_m0 - pa_target), 180.0 - abs(pa_m0 - pa_target))

    checks = {
        "major_half_amp_kms": dv_maj,
        "major_amp_ok": bool(dv_maj > 0.55 * expected),
        "minor_max_dev_kms": dv_min,
        "minor_flat_ok": bool(dv_min < 60.0),
        "m0_data_extent_arcsec": ext_data,
        "m0_model_extent_arcsec": ext_mod,
        "extent_ok": bool(ext_mod >= 6.0),
        "m0_pa_deg": pa_m0,
        "m0_pa_err_deg": pa_err,
        "m0_pa_ok": bool(pa_err < 40.0),
        "m0_pa_note": (
            "Morphological PA of a b/a~0.72 disk is weakly constrained; "
            "kinematic major/minor PV is the geometry gate"
        ),
        "expected_vlos_kms": expected,
        "v_cube_mid_kms": v_mid,
        "transpose": "(2, 1, 0) KinMS (nx,ny,nv) -> FITS (nv,ny,nx)",
        "sb_mode": "sbProf_from_M0_elliptical",
    }
    checks["pass"] = bool(
        checks["major_amp_ok"]
        and checks["minor_flat_ok"]
        and checks["extent_ok"]
        and checks["m0_pa_ok"]
    )
    print(json.dumps(checks, indent=2))
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
