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

from kinuv.infer.chart import PARAM_NAMES
from kinuv.infer.nuts import (
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
from kinuv.runner.checkpoint import dual_checkpoint, flush_scratch_to_arc
from kinuv.runner.log import (
    host_snapshot,
    install_crash_hook,
    logs_dir,
    rss_mb,
    setup_worker_logging,
)
from kinuv.runner.status_md import ping_status_ntfy, write_job_status_md
from kinuv.scratch import kinuv_scratch_root
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


def _heartbeat(run_id: str, stop: threading.Event, state: dict, log) -> None:
    while not stop.wait(30.0):
        rec = dict(state)
        rec["rss_mb"] = rss_mb()
        rec["updated_at"] = utc_now()
        write_status(run_id, rec)
        log.info(
            "heartbeat step=%s chain=%s rss_mb=%s",
            rec.get("step"),
            rec.get("chain"),
            None if rec["rss_mb"] is None else f"{rec['rss_mb']:.0f}",
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=os.environ.get("KINUV_RUN_ID", "kgas066-nuts"))
    args = p.parse_args()
    run_id = args.run_id
    dest = RUNS_ROOT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "posteriors").mkdir(exist_ok=True)
    arc_ckpt = dest / "checkpoints"
    arc_ckpt.mkdir(exist_ok=True)
    scratch_ckpt = kinuv_scratch_root() / "checkpoints"
    scratch_ckpt.mkdir(parents=True, exist_ok=True)
    logs_dir(run_id)
    log = setup_worker_logging(run_id)
    install_crash_hook(
        run_id,
        log,
        on_fail=lambda: flush_scratch_to_arc(scratch_ckpt, arc_ckpt),
    )
    snap = host_snapshot()
    log.info("worker start snapshot=%s", json.dumps(snap, default=str))

    state = {
        "state": "RUNNING",
        "step": "load",
        "chain": 0,
        "n_chain": N_CHAINS,
        "n_warmup": N_WARMUP,
        "n_samples": N_SAMPLES,
        "backend": BACKEND,
        "rss_mb": rss_mb(),
        "hostname": snap.get("hostname"),
        "session_id": snap.get("session_id"),
        "pid": snap.get("pid"),
    }
    write_status(run_id, state)
    data, tmpl, grid, params, map_rec = _load_066()
    log.info(
        "loaded 066 vis.shape=%s backend=%s rss_mb=%s",
        tuple(np.asarray(data.vis).shape),
        BACKEND,
        rss_mb(),
    )
    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    start = dict(params)
    start["pa_deg"] = float(OFFICIAL_PA)
    z6 = sampled_z_from_physical(params_to_vec(start))

    import jax
    import jax.numpy as jnp

    U = make_potential(data, tmpl, grid, dx, dy)
    u_jit = jax.jit(U)
    state["step"] = "compile"
    state["rss_mb"] = rss_mb()
    write_status(run_id, state)
    log.info("compile U at MAP z6 rss_mb=%s", rss_mb())
    u0 = float(u_jit(jnp.asarray(z6)))
    log.info("compiled U=%.6f rss_mb=%s", u0, rss_mb())

    stop = threading.Event()
    t = threading.Thread(target=_heartbeat, args=(run_id, stop, state, log), daemon=True)
    t.start()
    z_parts = []
    step_parts = []
    t0 = time.perf_counter()
    try:
        for c in range(N_CHAINS):
            state["step"] = f"{c}/{N_CHAINS}"
            state["chain"] = c + 1
            state["rss_mb"] = rss_mb()
            write_status(run_id, state)
            log.info(
                "chain %d/%d start warmup=%d samples=%d rss_mb=%s",
                c + 1,
                N_CHAINS,
                N_WARMUP,
                N_SAMPLES,
                rss_mb(),
            )
            tc = time.perf_counter()
            z_c, mean_steps, _ = run_nuts_z6(
                u_jit,
                z6,
                rng_seed=11 + c,
                num_warmup=N_WARMUP,
                num_samples=N_SAMPLES,
                num_chains=1,
                jitter=0.02,
                progress_bar=True,
            )
            elapsed_c = time.perf_counter() - tc
            z_arr = np.asarray(z_c, dtype=np.float64)
            if z_arr.ndim == 3:
                z_arr = z_arr[0]
            z_parts.append(z_arr)
            step_parts.append(mean_steps)
            chain_rec = {
                "chain": c + 1,
                "elapsed_s": elapsed_c,
                "mean_num_steps": float(mean_steps),
                "rss_mb": rss_mb(),
                "updated_at": utc_now(),
                "z6_shape": list(z_arr.shape),
            }
            write_json(logs_dir(run_id) / f"chain_{c + 1}.json", chain_rec)
            try:
                scratch_path, arc_path = dual_checkpoint(
                    scratch_ckpt,
                    arc_ckpt,
                    f"chain_{c + 1}.npz",
                    z6=z_arr,
                    mean_steps=np.asarray(mean_steps),
                )
                log.info(
                    "checkpoint chain %d scratch=%s arc=%s",
                    c + 1,
                    scratch_path,
                    arc_path,
                )
            except OSError:
                log.exception(
                    "checkpoint chain %d failed; draws kept in memory",
                    c + 1,
                )
            state["step"] = f"{c + 1}/{N_CHAINS}"
            state["rss_mb"] = rss_mb()
            write_status(run_id, state)
            log.info(
                "chain %d/%d done elapsed_s=%.1f mean_steps=%.1f rss_mb=%s",
                c + 1,
                N_CHAINS,
                elapsed_c,
                float(mean_steps),
                rss_mb(),
            )
    finally:
        stop.set()
        try:
            n = len(flush_scratch_to_arc(scratch_ckpt, arc_ckpt))
            log.info("flushed %d scratch checkpoints to /arc", n)
        except OSError:
            log.exception("final scratch→arc flush failed")

    z_draws = np.stack(z_parts, axis=0)
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    mix_pass = mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)
    rt = np.asarray(phys8, dtype=np.float64)[..., PARAM_NAMES.index("r_t_arcsec")]
    r_t_at_floor = bool(np.median(rt) <= 0.5 + 1e-6)
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=OFFICIAL_PA,
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=True,
        mixing_pass=mix_pass,
        leftover_chi2_structured=True,
        r_t_at_floor=r_t_at_floor,
        mean_num_steps=float(np.mean(step_parts)),
        eval_s=float("nan"),
        note=(
            "066 headless NUTS PA 199.73; 4 chains x 600 draws; "
            "16/50/84 not calibrated; leftover from MAP not refit; "
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
            "rss_mb": rss_mb(),
        },
    )
    (dest / ".trigger_complete").write_text(utc_now() + "\n")
    try:
        write_job_status_md(
            run_id=run_id,
            session_id=snap.get("session_id") or os.environ.get("SKAHA_SESSION_ID"),
            state="SUCCEEDED" if mix_pass else "COMPLETED_UNMIXED",
            mixing_pass=mix_pass,
            sampler=str(rec["sampler"]),
            elapsed_s=float(rec["elapsed_s"]),
            note=(
                "Agent Run Status written by the worker. "
                "Official MAP unchanged. Do not start G4"
            ),
        )
        ping_status_ntfy()
    except Exception:
        log.exception("STATUS.md patch failed")
    log.info(
        "done mixing_pass=%s sampler=%s elapsed_s=%.1f rss_mb=%s r_t_at_floor=%s",
        mix_pass,
        rec["sampler"],
        rec["elapsed_s"],
        rss_mb(),
        r_t_at_floor,
    )
    return 0 if mix_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
