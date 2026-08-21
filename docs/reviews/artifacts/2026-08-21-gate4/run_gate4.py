#!/usr/bin/env python3
"""066-12 Gate 4: smoke then sequential λ_reg. No NUTS. No rings on real vis yet."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from kinuv.infer.campaign import calibrate_lambda_reg
from kinuv.infer.map import image_grid_for_vis
from kinuv.infer.stage_b import nuisance_from_params, run_stage_b_map
from kinuv.io.vis import load_kgas066

NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
OUT = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-f47bc9-lambda"
)
STAGE_A = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-f47bc9-map/stage_a_map.json"
)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        rec = calibrate_lambda_reg(smoke=True, out_dir=OUT / "smoke")
        print(json.dumps(rec, indent=2), flush=True)
        v0 = rec["v0_stage_a"][0]
        if abs(v0 - 200.0) > 80.0:
            raise SystemExit(f"SMOKE_V0_FAIL {v0}")
        return
    if mode == "campaign":
        rec = calibrate_lambda_reg(smoke=False, out_dir=OUT)
        print(json.dumps({k: rec[k] for k in rec if k != "rows"}, indent=2), flush=True)
        (OUT / "chosen.json").write_text(
            json.dumps({"chosen_lambda": rec["chosen_lambda"]}, indent=2) + "\n"
        )
        if rec["chosen_lambda"] is None:
            raise SystemExit("LAMBDA_NONE")
        return
    if mode == "stage-b":
        chosen = json.loads((OUT / "chosen.json").read_text())["chosen_lambda"]
        if chosen is None:
            raise SystemExit("LAMBDA_NONE")
        from kinuv.forward.sb import load_sb_template

        cube = CUBE if CUBE.is_file() else None
        data = load_kgas066(NPZ, cube_path=cube)
        grid = image_grid_for_vis(data)
        tmpl = load_sb_template(grid, ico_path=ICO)
        a = json.loads(STAGE_A.read_text())
        rec_b = run_stage_b_map(
            data,
            nuisance_from_params(a),
            tmpl,
            grid,
            lam_reg=float(chosen),
            v0_init=float(a["v0_kms"]),
            rt_init=float(a["r_t_arcsec"]),
            chi2_stage_a=float(a["chi2_map"]),
        )
        payload = {
            "v_knots_kms": list(rec_b.v_knots_kms),
            "r_knots_arcsec": list(rec_b.r_knots_arcsec),
            "lam_reg": rec_b.lam_reg,
            "chi2_map": rec_b.chi2_map,
            "chi2_stage_a": rec_b.chi2_stage_a,
            "aic_stage_a": rec_b.aic_stage_a,
            "aic_stage_b": rec_b.aic_stage_b,
            "keep_stage_a": rec_b.keep_stage_a,
            "delta_chi2": rec_b.delta_chi2,
            "v0_recovered": rec_b.v0_recovered,
            "r_t_recovered": rec_b.r_t_recovered,
            "max_omega": rec_b.max_omega,
            "nfev": rec_b.nfev,
            "success": rec_b.success,
            "message": rec_b.message,
        }
        dest = Path(str(STAGE_A).replace("stage_a_map.json", "stage_b_map.json"))
        dest.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2), flush=True)
        return
    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
