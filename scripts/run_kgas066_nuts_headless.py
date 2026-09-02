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
from kinuv.runner.canfar import (
    ARTIFACT_G3_REL,
    KIND_PA25,
    OFFICIAL_PA_DEG,
    RUNS_ROOT,
    artifact_dir_for_kind,
    utc_now,
    write_json,
    write_status,
)
from kinuv.runner.checkpoint import dual_checkpoint, flush_scratch_to_arc
from kinuv.runner.log import (
    host_snapshot,
    install_crash_hook,
    logs_dir,
    rss_mb,
    setup_worker_logging,
)
from kinuv.runner.status_md import ping_status_ntfy, write_job_status_md
from kinuv.runner.plots import mean_params, write_leftover_at_params, write_nuts_product_plots
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
OFFICIAL_PA = OFFICIAL_PA_DEG
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
    p.add_argument(
        "--pa-init",
        type=float,
        default=None,
        help="Physical PA start (deg). Default: KINUV_PA_INIT env, else official MAP.",
    )
    p.add_argument(
        "--chain-id",
        type=int,
        default=None,
        help="1-4: run one chain and stop. Default KINUV_CHAIN_ID.",
    )
    args = p.parse_args()
    run_id = args.run_id
    kind = os.environ.get("KINUV_KIND", "nuts")
    chain_raw = args.chain_id if args.chain_id is not None else os.environ.get("KINUV_CHAIN_ID")
    chain_id = int(chain_raw) if chain_raw not in (None, "", "0") else None
    if chain_id is not None and chain_id not in (1, 2, 3, 4):
        raise SystemExit("--chain-id must be 1..4")
    if args.pa_init is not None:
        pa_init = float(args.pa_init)
    else:
        pa_init = float(os.environ.get("KINUV_PA_INIT", str(OFFICIAL_PA)))
    art_env = os.environ.get("KINUV_ARTIFACT_DIR")
    if art_env:
        artifact_dir = Path(art_env)
    else:
        artifact_dir = artifact_dir_for_kind(kind)
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

    n_loop = 1 if chain_id is not None else N_CHAINS
    state = {
        "state": "RUNNING",
        "step": "load",
        "chain": 0,
        "n_chain": n_loop,
        "n_warmup": N_WARMUP,
        "n_samples": N_SAMPLES,
        "backend": BACKEND,
        "rss_mb": rss_mb(),
        "hostname": snap.get("hostname"),
        "session_id": snap.get("session_id"),
        "pid": snap.get("pid"),
        "chain_id": chain_id,
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
    start["pa_deg"] = float(pa_init)
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
        chain_ids = [chain_id] if chain_id is not None else list(range(1, N_CHAINS + 1))
        for c in chain_ids:
            state["step"] = f"{c}/{n_loop}"
            state["chain"] = c
            state["rss_mb"] = rss_mb()
            write_status(run_id, state)
            log.info(
                "chain %d/%d start warmup=%d samples=%d rss_mb=%s",
                c,
                n_loop,
                N_WARMUP,
                N_SAMPLES,
                rss_mb(),
            )
            tc = time.perf_counter()
            z_c, mean_steps, _ = run_nuts_z6(
                u_jit,
                z6,
                rng_seed=11 + (c - 1),
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
                "chain": c,
                "elapsed_s": elapsed_c,
                "mean_num_steps": float(mean_steps),
                "rss_mb": rss_mb(),
                "updated_at": utc_now(),
                "z6_shape": list(z_arr.shape),
                "rng_seed": 11 + (c - 1),
            }
            write_json(logs_dir(run_id) / f"chain_{c}.json", chain_rec)
            try:
                scratch_path, arc_path = dual_checkpoint(
                    scratch_ckpt,
                    arc_ckpt,
                    f"chain_{c}.npz",
                    z6=z_arr,
                    mean_steps=np.asarray(mean_steps),
                )
                log.info(
                    "checkpoint chain %d scratch=%s arc=%s",
                    c,
                    scratch_path,
                    arc_path,
                )
            except OSError:
                log.exception(
                    "checkpoint chain %d failed; draws kept in memory",
                    c,
                )
            state["step"] = f"{c}/{n_loop}"
            state["rss_mb"] = rss_mb()
            write_status(run_id, state)
            log.info(
                "chain %d/%d done elapsed_s=%.1f mean_steps=%.1f rss_mb=%s",
                c,
                n_loop,
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

    if chain_id is not None:
        write_status(
            run_id,
            {
                "state": "SUCCEEDED",
                "step": f"{chain_id}/1",
                "chain": chain_id,
                "mixing_pass": False,
                "sampler": "pending_merge",
                "elapsed_s": time.perf_counter() - t0,
                "rss_mb": rss_mb(),
            },
        )
        (dest / ".trigger_complete").write_text(utc_now() + "\n")
        log.info("single-chain shard done chain_id=%s; merge is host-side", chain_id)
        return 0

    z_draws = np.stack(z_parts, axis=0)
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    mix_pass = mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)
    rt = np.asarray(phys8, dtype=np.float64)[..., PARAM_NAMES.index("r_t_arcsec")]
    r_t_at_floor = bool(abs(float(np.median(rt)) - 0.5) <= 0.01)
    if kind == KIND_PA25 and ARTIFACT_G3_REL in str(artifact_dir):
        raise SystemExit(
            "approaching NUTS must not write docs/reviews/artifacts/2026-08-30-g3-nuts/"
        )
    leftover_structured = False
    try:
        mean_p = mean_params({"sampler": "nuts", "draws": phys8})
        leftover_rec = write_leftover_at_params(
            mean_p, dest / "plots", data=data, tmpl=tmpl, grid=grid
        )
        leftover_structured = bool(leftover_rec["leftover_chi2_structured"])
    except Exception:
        log.exception("leftover at NUTS mean failed; not copying G0 leftover bit")
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=pa_init,
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=True,
        mixing_pass=mix_pass,
        leftover_chi2_structured=leftover_structured,
        r_t_at_floor=r_t_at_floor,
        mean_num_steps=float(np.mean(step_parts)),
        eval_s=float("nan"),
        note=(
            f"066 headless NUTS PA {pa_init:.2f}; 4 chains x 600 draws; "
            "16/50/84 not calibrated; leftover plotted at NUTS mean; "
            "not S2 Laplace; do not quote inner dV/dr"
        ),
    )
    rec["mixing_pass"] = mix_pass
    rec["elapsed_s"] = time.perf_counter() - t0
    rec["backend"] = BACKEND
    rec["s"] = float(map_rec.get("s", data.s))
    rec["kind"] = kind
    write_json(dest / "posteriors" / "kgas066_nuts.json", rec)
    summary = {k: rec[k] for k in rec if k != "draws"}
    write_json(dest / "posteriors" / "summary.json", summary)
    try:
        plotted = write_nuts_product_plots(
            rec,
            dest,
            artifact_dir=artifact_dir,
            data=data,
            tmpl=tmpl,
            grid=grid,
        )
        log.info("product plots pngs=%s", plotted.get("artifact_pngs"))
    except Exception:
        log.exception("product plots failed")
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
            kind=kind,
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
