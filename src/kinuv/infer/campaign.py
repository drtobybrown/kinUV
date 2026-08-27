"""OSCMETRIC λ_reg grid on mock 066 vis (066-12). Kinematics only.

Truth is arctan ``(V_0=200, r_t=3″)`` on the Hann+bin fit array plus Gaussian
noise. Walk λ ascending and stop at the first pass. Do not name the 066-4
campaign wrapper here (infer/ must not contain that symbol).
"""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import numpy as np

from kinuv.decisions import requires
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.infer.seeds import stage_a_seeds
from kinuv.infer.stage_b import (
    N_RINGS_DEFAULT,
    StageBResult,
    fit_v0_rt,
    nuisance_from_params,
    predict_binned,
    run_stage_b_map,
)
from kinuv.io.vis import VisData, load_kgas066
from kinuv.profiles.rotation import (
    CALIBRATION_RT_ARCSEC,
    CALIBRATION_V0_KM_S,
    N_RINGS_MAX,
    RECOVERY_RT_ARCSEC,
    RECOVERY_V0_KMS,
    select_lambda_reg,
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
INJECT_FLUX_JY = 60.6
LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_MOCK_DEFAULT = 20


def inject_truth_params() -> dict[str, float]:
    """Frozen nuisance + calibration arctan. Flux from Stage A MAP scale."""
    p = stage_a_seeds()
    p["flux"] = INJECT_FLUX_JY
    p["dx_arcsec"] = 0.0
    p["dy_arcsec"] = 0.0
    p["v0_kms"] = CALIBRATION_V0_KM_S
    p["r_t_arcsec"] = CALIBRATION_RT_ARCSEC
    return p


def add_complex_noise(vis, weights, s, rng: np.random.Generator):
    """Independent Re/Im with ``σ² = 1/(s w)`` so ``E[s w |n|²] = 2``."""
    w = np.asarray(weights, dtype=np.float64)
    vis_c = np.asarray(vis, dtype=np.complex128)
    sigma = np.zeros_like(w)
    good = w > 0.0
    sigma[good] = 1.0 / np.sqrt(float(s) * w[good])
    noise = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)
    out = vis_c.copy()
    out[good] = vis_c[good] + noise[good]
    return out


def mock_visdata(base: VisData, vis_clean, rng: np.random.Generator) -> VisData:
    noisy = add_complex_noise(vis_clean, base.weights, base.s, rng)
    out = copy(base)
    out.vis = noisy
    return out


def _prior_checkpoint(out_dir: Path | None, n_rings: int, n_use: int, smoke: bool):
    """Load ``campaign.json`` if it is the same residual-Ω n_rings / n_mock run."""
    if out_dir is None or smoke:
        return None
    path = Path(out_dir) / "campaign.json"
    if not path.is_file():
        return None
    prior = json.loads(path.read_text())
    if prior.get("omega_mode") != "residual":
        return None
    if int(prior.get("n_rings", -1)) != int(n_rings):
        return None
    if int(prior.get("n_mock", -1)) != int(n_use):
        return None
    return prior


@requires("DEC-066-OSCMETRIC", "DEC-066-VC", "DEC-066-VIS")
def calibrate_lambda_reg(
    *,
    data: VisData | None = None,
    template=None,
    grid=None,
    n_mock: int = N_MOCK_DEFAULT,
    lambdas=LAMBDA_GRID,
    n_rings: int = N_RINGS_DEFAULT,
    seed: int = 66,
    out_dir: Path | None = None,
    smoke: bool = False,
):
    """20 mocks × λ until OSCMETRIC passes. Returns chosen λ or ``None``."""
    if data is None:
        if not NPZ.is_file():
            raise FileNotFoundError(NPZ)
        if not ICO.is_file():
            raise FileNotFoundError(f"Ico required (no exponential fallback): {ICO}")
        cube = CUBE if CUBE.is_file() else None
        data = load_kgas066(NPZ, cube_path=cube)
    if data.vis.shape != (881, 95):
        raise RuntimeError(f"unexpected fit shape {data.vis.shape}")
    if grid is None:
        grid = image_grid_for_vis(data)
    if template is None:
        if not ICO.is_file():
            raise FileNotFoundError(f"Ico required (no exponential fallback): {ICO}")
        template = load_sb_template(grid, ico_path=ICO)
    truth = inject_truth_params()
    nuis = nuisance_from_params(truth)
    vis_clean = predict_binned(
        data,
        nuis,
        template,
        grid,
        v0_kms=truth["v0_kms"],
        r_t_arcsec=truth["r_t_arcsec"],
    )
    n_use = 1 if smoke else int(n_mock)
    lam = np.asarray(list(lambdas), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    mocks = [mock_visdata(data, vis_clean, rng) for _ in range(n_use)]
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    prior = _prior_checkpoint(out, int(n_rings), n_use, bool(smoke))
    stage_a = []
    if prior is not None and len(prior.get("v0_stage_a", [])) == n_use:
        v0_a = np.asarray(prior["v0_stage_a"], dtype=np.float64)
        stage_a = [{"v0_kms": float(v), "chi2": None} for v in v0_a]
        print(f"resume: skip {n_use} Stage A from checkpoint", flush=True)
    else:
        for i, mock in enumerate(mocks):
            rec = fit_v0_rt(mock, nuis, template, grid)
            if rec["delta_chi2"] <= 0.0 or rec["v0_kms"] < 1.0:
                raise RuntimeError(f"mock {i} Stage A failed inject recovery: {rec}")
            stage_a.append(rec)
            print(
                f"mock {i} Stage A V0={rec['v0_kms']:.1f} rt={rec['r_t_arcsec']:.2f} "
                f"dchi={rec['delta_chi2']:.1f}",
                flush=True,
            )
        v0_a = np.array([r["v0_kms"] for r in stage_a], dtype=np.float64)
    max_omega = np.empty((0, n_use), dtype=np.float64)
    v0_b = np.empty((0, n_use), dtype=np.float64)
    rt_b = np.empty((0, n_use), dtype=np.float64)
    chosen = None
    used = []
    rows = list(prior["rows"]) if prior is not None else []
    done = {(float(r["lambda"]), int(r["mock"])): r for r in rows}

    def _checkpoint() -> None:
        if out is None:
            return
        payload = {
            "chosen_lambda": chosen,
            "n_mock": n_use,
            "n_rings": int(n_rings),
            "lambdas_tried": used,
            "v0_stage_a": v0_a.tolist(),
            "rows": rows,
            "smoke": bool(smoke),
            "omega_mode": "residual",
        }
        (out / "campaign.json").write_text(json.dumps(payload, indent=2) + "\n")

    _checkpoint()
    for lam_i in lam:
        om = np.empty(n_use, dtype=np.float64)
        v0s = np.empty(n_use, dtype=np.float64)
        rts = np.empty(n_use, dtype=np.float64)
        for j, mock in enumerate(mocks):
            key = (float(lam_i), j)
            if key in done:
                prev = done[key]
                om[j] = float(prev["max_omega"])
                v0s[j] = float(prev["v0_b"])
                rts[j] = float(prev["rt_b"])
                print(
                    f"resume: skip lambda={lam_i} mock={j} "
                    f"maxΩ={om[j]:.3f} V0={v0s[j]:.1f} rt={rts[j]:.2f}",
                    flush=True,
                )
                continue
            rec_b: StageBResult = run_stage_b_map(
                mock,
                nuis,
                template,
                grid,
                lam_reg=float(lam_i),
                v0_init=truth["v0_kms"],
                rt_init=truth["r_t_arcsec"],
                n_rings=int(n_rings),
                chi2_stage_a=stage_a[j]["chi2"],
            )
            om[j] = rec_b.max_omega
            v0s[j] = rec_b.v0_recovered
            rts[j] = rec_b.r_t_recovered
            row = {
                "lambda": float(lam_i),
                "mock": j,
                "max_omega": rec_b.max_omega,
                "v0_b": rec_b.v0_recovered,
                "rt_b": rec_b.r_t_recovered,
                "v0_a": stage_a[j]["v0_kms"],
                "nfev": rec_b.nfev,
            }
            rows.append(row)
            done[key] = row
            print(
                f"lambda={lam_i} mock={j} maxΩ={rec_b.max_omega:.3f} "
                f"V0={rec_b.v0_recovered:.1f} rt={rec_b.r_t_recovered:.2f}",
                flush=True,
            )
            _checkpoint()
        used.append(float(lam_i))
        max_omega = np.vstack([max_omega, om[None, :]]) if max_omega.size else om[None, :]
        v0_b = np.vstack([v0_b, v0s[None, :]]) if v0_b.size else v0s[None, :]
        rt_b = np.vstack([rt_b, rts[None, :]]) if rt_b.size else rts[None, :]
        if n_use < 2:
            chosen = float(lam_i)
            break
        picked = select_lambda_reg(
            np.asarray(used, dtype=np.float64),
            max_omega,
            v0_b,
            rt_b,
            v0_sigma=RECOVERY_V0_KMS,
            rt_sigma=RECOVERY_RT_ARCSEC,
            v0_stage_a=v0_a,
        )
        if picked is not None:
            chosen = float(picked)
            break
        _checkpoint()
    if chosen is None and int(n_rings) < N_RINGS_MAX and not smoke:
        if out is not None:
            snap = out / f"campaign_n{int(n_rings)}.json"
            if (out / "campaign.json").is_file():
                snap.write_text((out / "campaign.json").read_text())
        return calibrate_lambda_reg(
            data=data,
            template=template,
            grid=grid,
            n_mock=n_mock,
            lambdas=lambdas,
            n_rings=int(n_rings) + 1,
            seed=seed,
            out_dir=out_dir,
            smoke=False,
        )
    _checkpoint()
    return {
        "chosen_lambda": chosen,
        "n_mock": n_use,
        "n_rings": int(n_rings),
        "lambdas_tried": used,
        "v0_stage_a": v0_a.tolist(),
        "rows": rows,
        "smoke": bool(smoke),
        "omega_mode": "residual",
    }
