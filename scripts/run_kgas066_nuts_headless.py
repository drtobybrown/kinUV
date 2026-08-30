#!/usr/bin/env python3
"""Official 066 NUTS worker for a CANFAR headless session. No vis dumps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

_scratch_py = Path(__file__).resolve().parents[1] / "src/kinuv/scratch.py"
_spec = importlib.util.spec_from_file_location("_kinuv_scratch", _scratch_py)
_scratch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scratch)
_scratch.apply_scratch_env()

import numpy as np

from kinuv.infer.chart import PARAM_NAMES, params_to_unconstrained
from kinuv.infer.nuts import (
    SAMPLED_IDX,
    make_potential,
    mixing_ok,
    mixing_sampled,
    physical_sampled_from_z6,
    product_record,
    run_nuts_z6,
    sampled_z_from_physical,
)
from kinuv.infer.posterior import params_to_vec
from kinuv.runner.canfar import RUNS_ROOT, utc_now, write_json, write_status
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
OFFICIAL_PA = 199.72980072503037
N_WARMUP = 200
N_SAMPLES = 600
N_CHAINS = 4


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


def _heartbeat(run_id: str, stop: threading.Event, state: dict) -> None:
    while not stop.wait(30.0):
        write_status(run_id, dict(state, updated_at=utc_now()))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=os.environ.get("KINUV_RUN_ID", "kgas066-nuts"))
    args = p.parse_args()
    run_id = args.run_id
    dest = RUNS_ROOT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "posteriors").mkdir(exist_ok=True)

    state = {
        "state": "RUNNING",
        "step": "load",
        "chain": 0,
        "n_chain": N_CHAINS,
        "n_warmup": N_WARMUP,
        "n_samples": N_SAMPLES,
        "backend": BACKEND,
    }
    write_status(run_id, state)
    data, tmpl, grid, params, map_rec = _load_066()
    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    start = dict(params)
    start["pa_deg"] = float(OFFICIAL_PA)
    z6 = sampled_z_from_physical(params_to_vec(start))

    import jax
    import jax.numpy as jnp

    U = make_potential(data, tmpl, grid, dx, dy)
    u_jit = jax.jit(U)
    state["step"] = "compile"
    write_status(run_id, state)
    _ = float(u_jit(jnp.asarray(z6)))

    stop = threading.Event()
    t = threading.Thread(target=_heartbeat, args=(run_id, stop, state), daemon=True)
    t.start()
    z_parts = []
    step_parts = []
    t0 = time.perf_counter()
    try:
        for c in range(N_CHAINS):
            state["step"] = f"{c}/{N_CHAINS}"
            state["chain"] = c + 1
            write_status(run_id, state)
            z_c, mean_steps, _ = run_nuts_z6(
                u_jit,
                z6,
                rng_seed=11 + c,
                num_warmup=N_WARMUP,
                num_samples=N_SAMPLES,
                num_chains=1,
                jitter=0.02,
            )
            z_parts.append(np.asarray(z_c)[0])
            step_parts.append(mean_steps)
            state["step"] = f"{c + 1}/{N_CHAINS}"
            write_status(run_id, state)
    finally:
        stop.set()

    z_draws = np.stack(z_parts, axis=0)
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    mix_pass = mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=OFFICIAL_PA,
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=True,
        mixing_pass=mix_pass,
        leftover_chi2_structured=True,
        r_t_at_floor=True,
        mean_num_steps=float(np.mean(step_parts)),
        eval_s=float("nan"),
        note=(
            "066 headless NUTS PA 199.73; 4 chains x 600 draws; "
            "16/50/84 not calibrated; leftover structured; r_t floor; "
            "not S2 Laplace; do not quote inner dV/dr"
        ),
    )
    rec["mixing_pass"] = mix_pass
    rec["elapsed_s"] = time.perf_counter() - t0
    rec["backend"] = BACKEND
    rec["s"] = float(map_rec.get("s", data.s))
    write_json(dest / "posteriors" / "kgas066_nuts.json", rec)
    summary = {k: rec[k] for k in rec if k != "draws"}
    write_json(dest / "posteriors" / "summary.json", summary)
    write_status(
        run_id,
        {
            "state": "SUCCEEDED" if mix_pass else "COMPLETED_UNMIXED",
            "step": f"{N_CHAINS}/{N_CHAINS}",
            "mixing_pass": mix_pass,
            "sampler": rec["sampler"],
            "elapsed_s": rec["elapsed_s"],
        },
    )
    (dest / ".trigger_complete").write_text(utc_now() + "\n")
    return 0 if mix_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
