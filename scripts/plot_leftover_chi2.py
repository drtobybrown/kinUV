#!/usr/bin/env python3
"""Leftover vis chi2 of the official 066 Stage A MAP vs baseline length and velocity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kinuv.diagnostics.figures import plot_leftover_chi2
from kinuv.diagnostics.s1 import (
    CANFAR_CUBE_30,
    CANFAR_ICO,
    CANFAR_NPZ,
    MAP_DIR,
    assert_hann_bin_operator,
    leftover_chi2,
)
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.io.vis import load_kgas066

ARTIFACT = Path("docs/reviews/artifacts/2026-08-29-s1-mock")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=CANFAR_NPZ)
    p.add_argument("--map-dir", type=Path, default=MAP_DIR)
    p.add_argument("--out", type=Path, default=ARTIFACT)
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    kernel = assert_hann_bin_operator()
    cube30 = CANFAR_CUBE_30 if CANFAR_CUBE_30.is_file() else None
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    data = load_kgas066(args.npz, cube_path=cube30)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)
    rec = json.loads((args.map_dir / "stage_a_map.json").read_text())
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
    model = predict_binned(data, params, tmpl, grid)
    b_m, per_row, vel, per_chan = leftover_chi2(data, model)
    total = float(np.sum(per_row))
    summary = {
        "map": str(args.map_dir),
        "chi2_sum": total,
        "chi2_map_json": rec["chi2_map"],
        "n_row": int(data.vis.shape[0]),
        "n_chan": int(data.vis.shape[1]),
        "s": float(data.s),
        "pol": "XX",
        "pipeline_kernel": kernel,
        "mean_chi2_per_row": float(np.mean(per_row)),
        "mean_chi2_per_chan": float(np.mean(per_chan)),
        "note": "Spiral leftover in image-plane M0 is SB misspecification, not CLEAN.",
    }
    if abs(total - float(rec["chi2_map"])) >= 1.0:
        raise SystemExit(
            f"leftover chi2_sum={total:.1f} != stage_a chi2_map={rec['chi2_map']:.1f}"
        )
    if int(data.vis.shape[0]) != 881 or int(data.vis.shape[1]) != 95:
        raise SystemExit(f"fit array {data.vis.shape} != (881, 95)")
    (out / "leftover_chi2.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(
        out / "leftover_chi2.npz",
        baseline_m=b_m,
        chi2_row=per_row,
        vel_kms=vel,
        chi2_chan=per_chan,
    )

    plot_leftover_chi2(b_m, per_row, vel, per_chan, out / "leftover_chi2.png")
    print(f"leftover chi2 sum={total:.1f} (JSON {rec['chi2_map']:.1f}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
