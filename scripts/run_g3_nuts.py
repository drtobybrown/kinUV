#!/usr/bin/env python3
"""Tiny-mock NUTS, 066 wall projection, optional 066 CPU runs. No GPU. No vis dumps."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

_scratch_py = Path(__file__).resolve().parents[1] / "src/kinuv/scratch.py"
_spec = importlib.util.spec_from_file_location("_kinuv_scratch", _scratch_py)
_scratch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scratch)
_ROOT = _scratch.apply_scratch_env()

import numpy as np

from dataclasses import replace

from kinuv.constants import velocity_to_freq_hz
from kinuv.diagnostics.figures import plot_posterior_corner
from kinuv.infer.chart import PARAM_NAMES, params_to_unconstrained
from kinuv.infer.nuts import (
    SAMPLED_IDX,
    WALL_CAP_S,
    make_potential,
    mixing_ok,
    mixing_sampled,
    physical_sampled_from_z6,
    product_record,
    run_nuts_z6,
    sampled_z_from_physical,
)
from kinuv.infer.posterior import params_to_vec
from kinuv.io.vis import VisData
from kinuv.transforms.grid import ImageGrid, image_grid_from_uv, max_baseline_lambda
from kinuv.transforms.nufft import BACKEND

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "docs/reviews/artifacts/2026-08-30-g3-nuts"
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
G1_EVAL_S = 3.0119
S2_EVAL_S = 0.329
OFFICIAL_PA = 199.72980072503037
APPROACH_PA = 25.2
TINY_WARMUP = 128
TINY_SAMPLES = 256
N066_WARMUP = 200
N066_SAMPLES = 400


def _tiny_data():
    rng = np.random.default_rng(7)
    n_row, n_native, n_guard, n_bin = 12, 16, 1, 2
    u = rng.uniform(8.0, 80.0, n_row)
    v = rng.uniform(-60.0, 60.0, n_row)
    vsys = 8300.0
    dv_nat = 8.0
    vel_n = vsys + (np.arange(n_native) - n_native // 2) * dv_nat
    freqs_n = velocity_to_freq_hz(vel_n)
    n_trim = n_native - 2 * n_guard
    n_fit = n_trim // n_bin
    vis = rng.normal(size=(n_row, n_fit)) + 1j * rng.normal(size=(n_row, n_fit))
    w = np.ones((n_row, n_fit), dtype=np.float64)
    w_n = np.ones((n_row, n_trim), dtype=np.float64)
    vel = vel_n[n_guard:-n_guard].reshape(n_fit, n_bin).mean(axis=1)
    freqs = freqs_n[n_guard:-n_guard].reshape(n_fit, n_bin).mean(axis=1)
    return VisData(
        u_m=u,
        v_m=v,
        vis=vis,
        weights=w,
        freqs=freqs,
        vel=vel,
        freqs_native=freqs_n,
        vel_native=vel_n,
        n_bin=n_bin,
        dv_kms=float(np.median(np.abs(np.diff(vel)))),
        s=0.5136098555284736,
        phase_dir_rad=np.zeros(2),
        line_free_mask=np.ones(n_fit, dtype=bool),
        n_guard=n_guard,
        weights_native=w_n,
    )


def _tiny_template(grid: ImageGrid):
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.cell_arcsec
    xe, yn = np.meshgrid(x, y, indexing="xy")
    r2 = xe**2 + yn**2
    img = np.exp(-0.5 * r2 / 4.0)
    return img / img.sum()


def _dump(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")


def _checkpoint(name: str, rec: dict) -> None:
    _dump(_ROOT / "g3-nuts" / name, rec)


def _tiny_inject():
    from dataclasses import replace

    from kinuv.infer.map import predict_binned

    data = _tiny_data()
    mb = max_baseline_lambda(data.u_m, data.v_m, data.freqs_native)
    grid = image_grid_from_uv(mb, 8.0)
    tmpl = _tiny_template(grid)
    dx, dy = 0.1, -0.05
    params = {
        "flux": 2.0,
        "pa_deg": 200.0,
        "vsys_kms": 8300.0,
        "gas_sigma_kms": 12.0,
        "dx_arcsec": dx,
        "dy_arcsec": dy,
        "v0_kms": 200.0,
        "r_t_arcsec": 2.0,
    }
    model = np.asarray(predict_binned(data, params, tmpl, grid, xla=True))
    rms = float(np.sqrt(np.mean(np.abs(model) ** 2)))
    sig = 1.0e-3 * max(rms, 1e-6)
    rng = np.random.default_rng(11)
    vis = model + sig * (
        rng.standard_normal(model.shape) + 1j * rng.standard_normal(model.shape)
    )
    w = np.full_like(data.weights, 1.0e4)
    w_n = np.full_like(data.weights_native, 1.0e4)
    return replace(data, vis=vis, weights=w, weights_native=w_n), tmpl, grid, params


def run_tiny() -> dict:
    import jax
    import jax.numpy as jnp

    data, tmpl, grid, params = _tiny_inject()
    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    z6 = sampled_z_from_physical(params_to_vec(params))
    U = make_potential(data, tmpl, grid, dx, dy)
    u_jit = jax.jit(U)
    _ = float(u_jit(jnp.asarray(z6)))
    t0 = time.perf_counter()
    _ = float(u_jit(jnp.asarray(z6)))
    dt_fwd = time.perf_counter() - t0
    g = jax.jit(jax.grad(U))
    _ = np.asarray(g(jnp.asarray(z6)))
    t1 = time.perf_counter()
    _ = np.asarray(g(jnp.asarray(z6)))
    dt_grad = time.perf_counter() - t1
    z_draws, mean_steps, _ = run_nuts_z6(
        u_jit,
        z6,
        rng_seed=3,
        num_warmup=TINY_WARMUP,
        num_samples=TINY_SAMPLES,
        num_chains=4,
        jitter=0.05,
    )
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    autodiff_ok = True
    mix_pass = mixing_ok(mix, rhat_max=1.01, ess_min=200.0)
    eval_s = (1.0 / dt_fwd) if dt_fwd > 0 else float("nan")
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=200.0,
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=autodiff_ok,
        mixing_pass=mix_pass,
        leftover_chi2_structured=False,
        r_t_at_floor=False,
        mean_num_steps=mean_steps,
        eval_s=eval_s,
        note=(
            "tiny-mock 4-chain NUTS; not 066; intervals not calibrated; "
            f"fwd_eval_s={eval_s:.4g} vs G1 {G1_EVAL_S} S2 {S2_EVAL_S}; "
            f"t_grad_s={dt_grad:.4g}; mean_num_steps={mean_steps:.3g}"
        ),
    )
    rec["t_grad_s"] = dt_grad
    rec["t_fwd_s"] = dt_fwd
    rec["shape"] = list(phys8.shape)
    rec["backend"] = BACKEND
    rec["mixing_pass"] = mix_pass
    rec["autodiff_ok"] = autodiff_ok
    ART.mkdir(parents=True, exist_ok=True)
    _dump(ART / "tiny_mock_nuts.json", rec)
    plot_posterior_corner(
        rec,
        ART / "tiny_mock_corner.png",
        title="tiny-mock NUTS 6D; not 066; not calibrated; not S2 Laplace",
    )
    _checkpoint(
        "tiny_mix.json",
        {k: rec[k] for k in rec if k != "draws"},
    )
    return rec


def _load_066():
    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import image_grid_for_vis
    from kinuv.io.vis import load_kgas066

    rec = json.loads((MAP / "stage_a_map.json").read_text())
    data = load_kgas066(NPZ, cube_path=CUBE if CUBE.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO if ICO.is_file() else None)
    params = {n: rec[n] for n in PARAM_NAMES}
    return data, tmpl, grid, params, rec


def time_066_grad(data, tmpl, grid, params) -> dict:
    import jax
    import jax.numpy as jnp

    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    z6 = params_to_unconstrained(params)[list(SAMPLED_IDX)]
    U = make_potential(data, tmpl, grid, dx, dy)
    g = jax.jit(jax.grad(U))
    z = jnp.asarray(z6)
    _ = np.asarray(g(z))
    t0 = time.perf_counter()
    _ = np.asarray(g(z))
    dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = float(jax.jit(U)(z))
    dt_fwd = time.perf_counter() - t1
    return {
        "t_grad_s": dt,
        "t_fwd_s": dt_fwd,
        "eval_s": (1.0 / dt_fwd) if dt_fwd > 0 else float("nan"),
        "chi2_identity_note": "see pytest official chi2",
    }


def project_wall(mean_num_steps: float, t_grad_s: float) -> dict:
    n_chain = 4
    n_iter = N066_WARMUP + N066_SAMPLES
    proj = n_chain * n_iter * float(mean_num_steps) * float(t_grad_s)
    return {
        "n_chain": n_chain,
        "n_warmup": N066_WARMUP,
        "n_samples": N066_SAMPLES,
        "mean_num_steps": float(mean_num_steps),
        "t_grad_s": float(t_grad_s),
        "projected_s": proj,
        "wall_cap_s": WALL_CAP_S,
        "under_cap": bool(proj < WALL_CAP_S),
        "formula": "n_chain * (warmup+draw) * mean_num_steps * t_grad_066",
    }


def run_066_pa(data, tmpl, grid, params, pa_init: float, seed: int) -> dict:
    import jax
    import jax.numpy as jnp

    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    start = dict(params)
    start["pa_deg"] = float(pa_init)
    z6 = sampled_z_from_physical(params_to_vec(start))
    U = make_potential(data, tmpl, grid, dx, dy)
    u_jit = jax.jit(U)
    _ = float(u_jit(jnp.asarray(z6)))
    z_draws, mean_steps, _ = run_nuts_z6(
        u_jit,
        z6,
        rng_seed=seed,
        num_warmup=N066_WARMUP,
        num_samples=N066_SAMPLES,
        num_chains=4,
        jitter=0.02,
    )
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    mix_pass = mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=pa_init,
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=True,
        mixing_pass=mix_pass,
        leftover_chi2_structured=True,
        r_t_at_floor=True,
        mean_num_steps=mean_steps,
        eval_s=float("nan"),
        note=(
            f"066 NUTS PA init {pa_init}; 16/50/84 not calibrated; "
            "leftover-vs-velocity and r_t_at_floor still fire; do not quote inner dV/dr; "
            "not S2 Laplace"
        ),
    )
    rec["mixing_pass"] = mix_pass
    return rec


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    tiny = run_tiny()
    print("tiny mixing", json.dumps(tiny["mixing"], indent=2), flush=True)
    print("tiny sampler", tiny["sampler"], "pass", tiny["mixing_pass"], flush=True)
    data, tmpl, grid, params, map_rec = _load_066()
    grad = time_066_grad(data, tmpl, grid, params)
    proj = project_wall(tiny["mean_num_steps"], grad["t_grad_s"])
    summary = {
        "backend": BACKEND,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "tiny_sampler": tiny["sampler"],
        "tiny_mixing_pass": tiny["mixing_pass"],
        "tiny_mean_num_steps": tiny["mean_num_steps"],
        "tiny_eval_s": tiny["eval_s"],
        "g1_eval_s": G1_EVAL_S,
        "s2_eval_s": S2_EVAL_S,
        "066_grad": grad,
        "projection": proj,
        "s": float(map_rec.get("s", params.get("s", 0.5136098555284736))),
        "official_pa_deg": OFFICIAL_PA,
        "approach_pa_deg": APPROACH_PA,
        "gpu": False,
    }
    _dump(ART / "timing_projection.json", summary)
    _checkpoint("timing_projection.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not tiny["mixing_pass"]:
        print("tiny-mock mixing failed; skip 066 sampler: nuts", flush=True)
        return
    if not proj["under_cap"]:
        print(
            f"066 projection {proj['projected_s']:.0f}s > cap {WALL_CAP_S:.0f}s; "
            "no 066 sampler: nuts; no GPU",
            flush=True,
        )
        return
    receding = run_066_pa(data, tmpl, grid, params, OFFICIAL_PA, seed=11)
    _dump(ART / "kgas066_nuts_pa199.73.json", receding)
    plot_posterior_corner(
        receding,
        ART / "kgas066_nuts_pa199.73_corner.png",
        title="066 NUTS PA 199.73; 6 sampled; not calibrated; leftover structured; r_t floor",
    )
    approaching = run_066_pa(data, tmpl, grid, params, APPROACH_PA, seed=22)
    _dump(ART / "kgas066_nuts_pa25.2.json", approaching)
    plot_posterior_corner(
        approaching,
        ART / "kgas066_nuts_pa25.2_corner.png",
        title="066 NUTS PA 25.2; 6 sampled; not calibrated; leftover structured; r_t floor",
    )
    print("066 receding sampler", receding["sampler"], receding["mixing_pass"], flush=True)
    print(
        "066 approaching sampler",
        approaching["sampler"],
        approaching["mixing_pass"],
        flush=True,
    )


if __name__ == "__main__":
    main()
