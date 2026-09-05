#!/usr/bin/env python3
"""Evaluate KinMS cube χ² on 2-D parameter slices (external_fitters venv).

Does not refit. Reuses the existing sbProf from the mock KinMS work dir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from _kinms_best_worker import (
    _jy_beam_to_k,
    _make_kinms_cube,
    _scaled_cube_chi2,
    _vel_optical_kms,
)


def _eval_grid(pairs, center, hdr, sb_rad, sb_prof, x0, y0, v_mid, n_samps, data_k, mask3d, vel):
    chi2 = np.empty(len(pairs), dtype=np.float64)
    for i, (key_a, val_a, key_b, val_b) in enumerate(pairs):
        p = dict(center)
        p[key_a] = float(val_a)
        p[key_b] = float(val_b)
        x = np.array(
            [
                p["v0_kms"],
                p["r_t_arcsec"],
                p["pa_deg"],
                p["i_deg"],
                p["vsys_optical_kms"],
                p["gas_sigma_kms"],
            ],
            dtype=np.float64,
        )
        cube_jy = _make_kinms_cube(
            x, hdr, sb_rad, sb_prof, x0=x0, y0=y0, v_cube_mid=v_mid, n_samps=n_samps
        )
        model_k = _jy_beam_to_k(cube_jy, hdr, vel)
        chi2[i], _ = _scaled_cube_chi2(model_k, data_k, mask3d)
    return chi2


def main() -> int:
    cfg = json.loads(Path(sys.argv[1]).read_text())
    work = Path(cfg["work"])
    hdr = fits.getheader(cfg["cube"])
    data_k = np.asarray(fits.getdata(cfg["cube"]), dtype=np.float64)
    mask3d = np.asarray(fits.getdata(cfg["mask"]), dtype=np.float64) > 0.5
    vel = _vel_optical_kms(hdr)
    sb = np.load(cfg["sb_profile"])
    sb_rad = np.asarray(sb["sb_rad"], dtype=np.float64)
    sb_prof = np.asarray(sb["sb_prof"], dtype=np.float64)
    center = cfg["center"]
    x0, y0 = float(center.get("dx_arcsec", 0.0)), float(center.get("dy_arcsec", 0.0))
    v_mid = float(vel[len(vel) // 2])
    n_samps = int(cfg.get("n_clouds", 25000))
    out = {"center": center, "n_clouds": n_samps, "slices": {}}
    for name, spec in cfg["slices"].items():
        xa, xb = spec["x_name"], spec["y_name"]
        x = np.asarray(spec["x"], dtype=np.float64)
        y = np.asarray(spec["y"], dtype=np.float64)
        pairs = []
        for yi in y:
            for xi in x:
                pairs.append((xa, xi, xb, yi))
        z = _eval_grid(
            pairs, center, hdr, sb_rad, sb_prof, x0, y0, v_mid, n_samps, data_k, mask3d, vel
        )
        out["slices"][name] = {
            "x_name": xa,
            "y_name": xb,
            "x": x.tolist(),
            "y": y.tolist(),
            "chi2": z.reshape(y.size, x.size).tolist(),
        }
    dest = work / "kinms_chi2_slices.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": "wrote", "path": str(dest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
