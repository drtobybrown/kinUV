#!/usr/bin/env python3
"""S2 hybrid: Laplace-MH on the S1 inject, Laplace SBC, real-066 CI table.

sampler: laplace_mh (not autodiff NUTS). Hann+bin. XX empirical s. Stage A only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kinuv.diagnostics.s1 import (
    CANFAR_CUBE_30,
    CANFAR_ICO,
    CANFAR_NPZ,
    MAP_DIR,
    PIPELINE_KERNEL,
    S1_RT_ARCSEC,
    S1_RT_BOUNDS_ARCSEC,
    S1_SIGMA_KMS,
    S1_V0_KMS,
    assert_hann_bin_operator,
    inject_vis,
    params_from_map,
)
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import _lbfgs_one_start, image_grid_for_vis
from kinuv.infer.posterior import (
    PARAM_NAMES,
    SAMPLER_NAME,
    Z68,
    Z95,
    ess_bulk,
    gaussian_interval,
    hessian_at,
    in_interval,
    interval_table,
    laplace_cov,
    make_logp_vec,
    mcmc_intervals,
    mh_sample,
    n_vis_of,
    params_to_vec,
    split_rhat,
    t_dof,
    t_nvis,
)
from kinuv.infer.seeds import vsys_seed_radio_kms
from kinuv.io.vis import load_kgas066

ARTIFACT = Path("docs/reviews/artifacts/2026-08-29-s2")
S1_TABLE = Path("docs/reviews/artifacts/2026-08-29-s1-mock/s1_table.json")
TRUTH_PA_DEG = 199.73
TRUTH_FLUX_JY = 70.0
REPORT = (
    "pa_deg",
    "vsys_kms",
    "flux",
    "gas_sigma_kms",
    "v0_kms",
    "r_t_arcsec",
)


def _truth() -> dict[str, float]:
    return {
        "flux": TRUTH_FLUX_JY,
        "pa_deg": TRUTH_PA_DEG,
        "vsys_kms": vsys_seed_radio_kms(),
        "gas_sigma_kms": S1_SIGMA_KMS,
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "v0_kms": S1_V0_KMS,
        "r_t_arcsec": S1_RT_ARCSEC,
    }


def _load(npz: Path):
    cube30 = CANFAR_CUBE_30 if CANFAR_CUBE_30.is_file() else None
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    data = load_kgas066(npz, cube_path=cube30)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)
    return data, grid, tmpl


def _fit(data, tmpl, grid, start, maxiter: int):
    extra = {"r_t_arcsec": tuple(S1_RT_BOUNDS_ARCSEC)}
    rec = _lbfgs_one_start(
        data, tmpl, grid, start, 0.5, maxiter, start["pa_deg"], extra_bounds=extra
    )
    return params_from_map(rec), rec


def _hits(truth, mean_vec, cov, names=REPORT):
    idx = {n: PARAM_NAMES.index(n) for n in names}
    out68, out95 = {}, {}
    for n in names:
        i = idx[n]
        lo, hi = gaussian_interval(mean_vec[i], cov[i, i], Z68)
        out68[n] = in_interval(truth[n], lo, hi)
        lo95, hi95 = gaussian_interval(mean_vec[i], cov[i, i], Z95)
        out95[n] = in_interval(truth[n], lo95, hi95)
    return out68, out95


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=CANFAR_NPZ)
    p.add_argument("--out", type=Path, default=ARTIFACT)
    p.add_argument("--n-sbc", type=int, default=20)
    p.add_argument("--n-chain", type=int, default=4)
    p.add_argument("--n-warmup", type=int, default=300)
    p.add_argument("--n-draw", type=int, default=1200)
    p.add_argument("--sbc-maxiter", type=int, default=20)
    p.add_argument("--skip-mcmc", action="store_true")
    p.add_argument("--skip-sbc", action="store_true")
    p.add_argument("--skip-real", action="store_true")
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    kernel = assert_hann_bin_operator()
    if kernel != PIPELINE_KERNEL:
        raise SystemExit(f"pipeline_kernel={kernel!r}")
    print(f"pipeline_kernel=Hann+bin sampler={SAMPLER_NAME} (not NUTS)", flush=True)

    data, grid, tmpl = _load(args.npz)
    n_vis = n_vis_of(data)
    print(
        f"fit array {data.vis.shape} n_vis={n_vis} s={data.s:.4f} pol=XX",
        flush=True,
    )
    truth = _truth()

    if S1_TABLE.is_file():
        s1 = json.loads(S1_TABLE.read_text())
        map_params = {k: float(s1["vis_map"][k]) for k in PARAM_NAMES}
        print("S1 MAP start from s1_table.json", flush=True)
    else:
        map_params = dict(truth)
        print("no s1_table; MAP start is inject truth", flush=True)

    data_inj, _ = inject_vis(data, truth, tmpl, grid, rng=np.random.default_rng(66))

    if not args.skip_mcmc:
        print("Hessian at S1 MAP (T=1)...", flush=True)
        hess = hessian_at(data_inj, map_params, tmpl, grid, t=1.0)
        cov = laplace_cov(hess, t=1.0)
        logp = make_logp_vec(data_inj, tmpl, grid, map_params, t=1.0)
        print(
            f"MH n_chain={args.n_chain} n_warmup={args.n_warmup} "
            f"n_draw={args.n_draw}",
            flush=True,
        )
        mh = mh_sample(
            logp,
            params_to_vec(map_params),
            cov,
            n_chain=args.n_chain,
            n_warmup=args.n_warmup,
            n_draw=args.n_draw,
            rng=np.random.default_rng(66),
        )
        rhat = split_rhat(mh.samples)
        ess = ess_bulk(mh.samples)
        ints = mcmc_intervals(mh.samples, PARAM_NAMES)
        mean_vec = params_to_vec({n: ints[n]["mean"] for n in PARAM_NAMES})
        hit68, hit95 = _hits(truth, mean_vec, cov)
        # also truth vs percentile intervals
        pct_hit = {}
        for n in REPORT:
            pct_hit[n] = {
                "in_68": in_interval(truth[n], ints[n]["p16"], ints[n]["p84"]),
                "in_95": in_interval(truth[n], ints[n]["p025"], ints[n]["p975"]),
            }
        payload = {
            "sampler": SAMPLER_NAME,
            "pipeline_kernel": "Hann+bin",
            "pol": "XX",
            "s": float(data.s),
            "t": 1.0,
            "n_vis": n_vis,
            "n_row": int(data.vis.shape[0]),
            "n_chan": int(data.vis.shape[1]),
            "accept": mh.accept,
            "nfev": mh.nfev,
            "eval_s": mh.eval_s,
            "R_hat": {n: float(rhat[i]) for i, n in enumerate(PARAM_NAMES)},
            "ESS": {n: float(ess[i]) for i, n in enumerate(PARAM_NAMES)},
            "intervals": ints,
            "truth": truth,
            "truth_in_percentile": pct_hit,
            "truth_in_laplace68": hit68,
            "truth_in_laplace95": hit95,
            "gate_R_hat": {
                n: bool(rhat[PARAM_NAMES.index(n)] < 1.01)
                for n in ("pa_deg", "vsys_kms", "flux")
            },
            "gate_ESS": {
                n: bool(ess[PARAM_NAMES.index(n)] > 200)
                for n in ("pa_deg", "vsys_kms", "flux")
            },
        }
        _write(out / "s2_mock_mcmc.json", payload)
        print(
            f"MH accept={mh.accept:.3f} eval_s={mh.eval_s:.3f} nfev={mh.nfev} "
            f"R_hat_pa={rhat[1]:.4f} ESS_pa={ess[1]:.1f}",
            flush=True,
        )

    if not args.skip_sbc:
        names = REPORT
        hits68 = {n: 0 for n in names}
        hits95 = {n: 0 for n in names}
        rows = []
        n_sbc = int(args.n_sbc)
        print(f"SBC n={n_sbc} L-BFGS from truth maxiter={args.sbc_maxiter}", flush=True)
        for i in range(n_sbc):
            data_i, _ = inject_vis(
                data, truth, tmpl, grid, rng=np.random.default_rng(1000 + i)
            )
            p_i, rec = _fit(
                data_i, tmpl, grid, truth, maxiter=int(args.sbc_maxiter)
            )
            hess_i = hessian_at(data_i, p_i, tmpl, grid, t=1.0)
            cov_i = laplace_cov(hess_i, t=1.0)
            mean_i = params_to_vec(p_i)
            h68, h95 = _hits(truth, mean_i, cov_i, names)
            for n in names:
                hits68[n] += int(h68[n])
                hits95[n] += int(h95[n])
            rows.append(
                {
                    "draw": i,
                    "chi2": rec.chi2_map,
                    "params": p_i,
                    "hit68": h68,
                    "hit95": h95,
                }
            )
            print(
                f"  sbc {i+1}/{n_sbc} chi2={rec.chi2_map:.1f} "
                f"r_t={p_i['r_t_arcsec']:.3f} hit68_v0={h68['v0_kms']}",
                flush=True,
            )
        rate68 = {n: hits68[n] / n_sbc for n in names}
        rate95 = {n: hits95[n] / n_sbc for n in names}
        sbc = {
            "n_sbc": n_sbc,
            "t": 1.0,
            "pipeline_kernel": "Hann+bin",
            "pol": "XX",
            "rate68": rate68,
            "rate95": rate95,
            "draws": rows,
            "note": "Pass if rates are consistent with 0.68/0.95 (binomial), not a point match.",
        }
        _write(out / "s2_sbc.json", sbc)
        print(f"SBC rate68={rate68} rate95={rate95}", flush=True)

    if not args.skip_real:
        rec = json.loads((MAP_DIR / "stage_a_map.json").read_text())
        real_p = {n: float(rec[n]) for n in PARAM_NAMES}
        print("Real-066 Hessian at official Stage A MAP...", flush=True)
        hess_r = hessian_at(data, real_p, tmpl, grid, t=1.0)
        chi2_map = float(rec["chi2_map"])
        td = t_dof(chi2_map, n_vis)
        tn = t_nvis(chi2_map, n_vis)
        cov1 = laplace_cov(hess_r, t=1.0)
        covd = laplace_cov(hess_r, t=td)
        covn = laplace_cov(hess_r, t=tn)
        mean_r = params_to_vec(real_p)
        tab1 = interval_table(mean_r, cov1, PARAM_NAMES, z=Z68)
        tabd = interval_table(mean_r, covd, PARAM_NAMES, z=Z68)
        tabn = interval_table(mean_r, covn, PARAM_NAMES, z=Z68)
        table = {}
        for n in REPORT:
            w1 = tab1[n]["width"]
            table[n] = {
                "MAP": real_p[n],
                "unscaled_lo": tab1[n]["lo"],
                "unscaled_hi": tab1[n]["hi"],
                "T_dof_lo": tabd[n]["lo"],
                "T_dof_hi": tabd[n]["hi"],
                "T_nvis_lo": tabn[n]["lo"],
                "T_nvis_hi": tabn[n]["hi"],
                "width_ratio_T_dof": tabd[n]["width"] / w1 if w1 else float("nan"),
                "width_ratio_T_nvis": tabn[n]["width"] / w1 if w1 else float("nan"),
            }
        real = {
            "map": str(MAP_DIR),
            "chi2_map": chi2_map,
            "n_vis": n_vis,
            "T_dof": td,
            "T_nvis": tn,
            "chi2_red": td,
            "pol": "XX",
            "s": float(data.s),
            "table68": table,
            "note": (
                "T_dof = chi2 / (2 n_vis) is the product. "
                "T_nvis = chi2 / n_vis is sensitivity. "
                "Structured leftover vs velocity can still over-narrow real CIs."
            ),
        }
        _write(out / "s2_real_ci_table.json", real)
        lines = [
            "# Stage A 68% CI: unscaled vs T_dof vs T_nvis",
            "",
            f"chi2_map = {chi2_map:.1f}; n_vis = {n_vis}; "
            f"T_dof = {td:.4f}; T_nvis = {tn:.4f}",
            "",
            "| param | MAP | unscaled 68% | T_dof 68% | T_nvis 68% | width T_dof / unscaled | width T_nvis / unscaled |",
            "|---|---:|---|---|---|---:|---:|",
        ]
        for n in REPORT:
            r = table[n]
            lines.append(
                f"| `{n}` | {r['MAP']:.4g} | "
                f"[{r['unscaled_lo']:.4g}, {r['unscaled_hi']:.4g}] | "
                f"[{r['T_dof_lo']:.4g}, {r['T_dof_hi']:.4g}] | "
                f"[{r['T_nvis_lo']:.4g}, {r['T_nvis_hi']:.4g}] | "
                f"{r['width_ratio_T_dof']:.3f} | {r['width_ratio_T_nvis']:.3f} |"
            )
        lines.append("")
        (out / "s2_real_ci_table.md").write_text("\n".join(lines) + "\n")
        print(f"T_dof={td:.4f} T_nvis={tn:.4f} wrote {out / 's2_real_ci_table.md'}", flush=True)

    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
