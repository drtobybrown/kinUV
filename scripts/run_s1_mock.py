#!/usr/bin/env python3
"""S1: inject steep inner V_c on real 066 uv; Stage A vis vs CLEAN-cube moments.

Hann+bin only. Frozen i, no h_z. Mock-only r_t box. No Stage B. No NUTS.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

from kinuv.diagnostics.s1 import (
    CANFAR_CUBE_10,
    CANFAR_CUBE_30,
    CANFAR_ICO,
    CANFAR_NPZ,
    PIPELINE_KERNEL,
    S1_RT_ARCSEC,
    S1_RT_BOUNDS_ARCSEC,
    S1_SIGMA_KMS,
    S1_V0_KMS,
    PASS_SIGMA_KMS,
    assert_hann_bin_operator,
    chi2_slice_pa_rt,
    chi2_slice_pa_sigma,
    chi2_slice_sigma_inc,
    dirty_cube_from_truth,
    inject_vis,
    inner_slope_arctan,
    params_from_map,
    quadratic_cov_2d,
    r_eval_arcsec,
    vis_recovery_table,
)
from kinuv.diagnostics.figures import plot_chi2_slices
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import inclination_deg
from kinuv.infer.map import image_grid_for_vis, run_stage_a_map
from kinuv.infer.seeds import vsys_seed_radio_kms
from kinuv.io.vis import load_kgas066

ARTIFACT = Path("docs/reviews/artifacts/2026-08-29-s1-mock")
TRUTH_PA_DEG = 199.73
TRUTH_FLUX_JY = 70.0
N_GRID = 5


def _barolo_status() -> dict:
    exe = shutil.which("BBarolo") or shutil.which("3dbarolo") or shutil.which("barolo")
    if exe is None:
        return {
            "available": False,
            "note": "3DBarolo unavailable; beam-convolved M1/M2 is the cube estimator.",
        }
    return {"available": True, "executable": exe, "note": "on PATH; not run (no par file this wave)"}


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=CANFAR_NPZ)
    p.add_argument("--out", type=Path, default=ARTIFACT)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--skip-slices", action="store_true")
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    kernel = assert_hann_bin_operator()
    if kernel != PIPELINE_KERNEL:
        raise SystemExit(f"pipeline_kernel={kernel!r} != Hann+bin")
    print(f"pipeline_kernel=Hann+bin ({kernel})", flush=True)

    cube30 = CANFAR_CUBE_30 if CANFAR_CUBE_30.is_file() else None
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    data = load_kgas066(args.npz, cube_path=cube30)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)
    print(
        f"fit array {data.vis.shape} N={data.n_bin} s={data.s:.4f} pol=XX "
        f"i={inclination_deg():.2f}° frozen, h_z=absent",
        flush=True,
    )

    truth = {
        "flux": TRUTH_FLUX_JY,
        "pa_deg": TRUTH_PA_DEG,
        "vsys_kms": vsys_seed_radio_kms(),
        "gas_sigma_kms": S1_SIGMA_KMS,
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "v0_kms": S1_V0_KMS,
        "r_t_arcsec": S1_RT_ARCSEC,
    }
    rng = None if args.no_noise else np.random.default_rng(66)
    data_inj, _ = inject_vis(data, truth, tmpl, grid, rng=rng)
    print(
        f"inject r_t={truth['r_t_arcsec']}\" σ={truth['gas_sigma_kms']} km/s "
        f"V0={truth['v0_kms']} noise={'off' if rng is None else 'XX s'}",
        flush=True,
    )

    rec = run_stage_a_map(
        data_inj,
        template=tmpl,
        grid=grid,
        maxiter=int(args.maxiter),
        rt_bounds_arcsec=S1_RT_BOUNDS_ARCSEC,
    )
    table = vis_recovery_table(truth, rec)
    print(
        f"vis MAP PA={rec.pa_deg:.2f}° V0={rec.v0_kms:.2f} r_t={rec.r_t_arcsec:.3f}\" "
        f"σ={rec.gas_sigma_kms:.2f} ΔV0={table['delta_v0_kms']:+.2f} "
        f"Δσ={table['delta_sigma_kms']:+.2f} nfev={rec.nfev} eval={rec.eval_s:.3f}s",
        flush=True,
    )

    cube_kin = {
        "slope_cube_kms_per_arcsec": float("nan"),
        "sigma_m2_kms": float("nan"),
        "note": "10 km/s cube missing",
    }
    if CANFAR_CUBE_10.is_file() and CANFAR_ICO.is_file():
        ico_hdr = fits.getheader(CANFAR_ICO)
        img_hdr = fits.getheader(CANFAR_CUBE_10)
        _, _, _, cube_kin = dirty_cube_from_truth(
            truth, tmpl, grid, data.freqs_native, ico_hdr, img_hdr
        )
        print(
            f"cube M1 slope={cube_kin['slope_cube_kms_per_arcsec']:.2f} "
            f"M2 σ={cube_kin['sigma_m2_kms']:.2f} km/s "
            f"(truth slope={table['slope_truth_kms_per_arcsec']:.2f})",
            flush=True,
        )

    params = params_from_map(rec)
    cov = {}
    if not args.skip_slices:
        pa = np.linspace(rec.pa_deg - 6.0, rec.pa_deg + 6.0, N_GRID)
        sig = np.linspace(
            max(2.5, rec.gas_sigma_kms - 4.0), rec.gas_sigma_kms + 4.0, N_GRID
        )
        rt = np.linspace(max(0.08, rec.r_t_arcsec - 0.2), rec.r_t_arcsec + 0.4, N_GRID)
        i_deg = np.linspace(inclination_deg() - 8.0, inclination_deg() + 8.0, N_GRID)
        print("χ² slices (PA,σ), (σ,i), (PA,r_t)…", flush=True)
        z_pa_s = chi2_slice_pa_sigma(data_inj, tmpl, grid, params, pa, sig)
        z_s_i = chi2_slice_sigma_inc(data_inj, tmpl, grid, params, sig, i_deg)
        z_pa_rt = chi2_slice_pa_rt(data_inj, tmpl, grid, params, pa, rt)
        cov["pa_sigma"] = quadratic_cov_2d(pa, sig, z_pa_s, rec.pa_deg, rec.gas_sigma_kms)
        cov["sigma_i"] = quadratic_cov_2d(sig, i_deg, z_s_i, rec.gas_sigma_kms, inclination_deg())
        cov["pa_rt"] = quadratic_cov_2d(pa, rt, z_pa_rt, rec.pa_deg, rec.r_t_arcsec)
        slope_grid = np.array(
            [inner_slope_arctan(rec.v0_kms, r, r_eval_arcsec()) for r in rt]
        )
        cov["pa_inner_slope_note"] = (
            "PA–r_t slice; inner slope is a function of r_t at fixed V0"
        )
        cov["inner_slope_at_rt_grid"] = slope_grid.tolist()
        plot_chi2_slices(
            pa,
            sig,
            rt,
            i_deg,
            z_pa_s,
            z_s_i,
            z_pa_rt,
            {
                "pa_deg": rec.pa_deg,
                "gas_sigma_kms": rec.gas_sigma_kms,
                "r_t_arcsec": rec.r_t_arcsec,
                "i_deg": inclination_deg(),
            },
            out / "s1_chi2_slices.png",
        )
        np.savez(
            out / "s1_chi2_slices.npz",
            pa=pa,
            sigma=sig,
            r_t=rt,
            i_deg=i_deg,
            chi2_pa_sigma=z_pa_s,
            chi2_sigma_i=z_s_i,
            chi2_pa_rt=z_pa_rt,
        )

    barolo = _barolo_status()
    payload = {
        "pipeline_kernel": "Hann+bin",
        "operator": kernel,
        "pol": "XX",
        "s_empirical": float(data.s),
        "likelihood": "chi2 = s * sum w |d-m|^2; s from XX line-free vis; not Stokes I",
        "i_held_fixed": True,
        "h_z_in_model": False,
        "stage": "A",
        "rt_bounds_arcsec_mock": list(S1_RT_BOUNDS_ARCSEC),
        "rt_bounds_arcsec_production": [0.5, 15.0],
        "n_row": rec.n_row,
        "n_chan": rec.n_chan,
        "n_bin": rec.n_bin,
        "truth": truth,
        "vis_map": params,
        "vis_chi2": rec.chi2_map,
        "delta_chi2": rec.delta_chi2,
        "nfev": rec.nfev,
        "eval_s": rec.eval_s,
        "recovery": table,
        "cube": cube_kin,
        "covariance": cov,
        "barolo": barolo,
        "uv_win": bool(table["pass_v0"] and table["pass_sigma"]),
        "cube_biased": bool(
            np.isfinite(cube_kin.get("sigma_m2_kms", np.nan))
            and cube_kin["sigma_m2_kms"] > truth["gas_sigma_kms"] + PASS_SIGMA_KMS
        ),
    }
    (out / "s1_table.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out / 's1_table.json'}", flush=True)
    print(
        f"S1 vis pass V0={table['pass_v0']} σ={table['pass_sigma']} "
        f"barolo={barolo['available']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
