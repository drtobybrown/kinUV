#!/usr/bin/env python3
"""Post-warmup G1 eval/s on official Stage A. JSON only. No vis dumps."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/kinuv-jax-cache")

import numpy as np

from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.io.vis import load_kgas066
from kinuv.likelihood.chi2 import chi2
from kinuv.transforms.nufft import BACKEND

MAP = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
OUT = Path("docs/reviews/artifacts/2026-08-30-g1-jax/timing.json")


def main() -> None:
    rec = json.loads((MAP / "stage_a_map.json").read_text())
    data = load_kgas066(NPZ, cube_path=CUBE if CUBE.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO if ICO.is_file() else None)
    params = {
        "flux": rec["flux"],
        "pa_deg": rec["pa_deg"],
        "vsys_kms": rec["vsys_kms"],
        "gas_sigma_kms": rec["gas_sigma_kms"],
        "dx_arcsec": rec["dx_arcsec"],
        "dy_arcsec": rec["dy_arcsec"],
        "v0_kms": rec["v0_kms"],
        "r_t_arcsec": rec["r_t_arcsec"],
    }
    vis = predict_binned(data, params, tmpl, grid, xla=True)
    c0 = float(chi2(data.vis, vis, data.weights, data.s))
    t0 = time.perf_counter()
    vis = predict_binned(data, params, tmpl, grid, xla=True)
    c1 = float(chi2(data.vis, vis, data.weights, data.s))
    dt = time.perf_counter() - t0
    summary = {
        "backend": BACKEND,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "chi2": c1,
        "chi2_warmup": c0,
        "s": float(data.s),
        "seconds_post_warmup": dt,
        "eval_per_s": (1.0 / dt) if dt > 0 else None,
        "s2_fd_eval_per_s": 0.329,
        "beats_s2_fd": bool(dt > 0 and (1.0 / dt) > 0.329),
        "identity_ok": abs(c1 - 168675.6) < 1.0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
