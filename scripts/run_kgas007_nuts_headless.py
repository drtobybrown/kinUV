#!/usr/bin/env python3
"""KGAS007 CPU NUTS worker. Wavelength vis loader. Does not steal 066-latest.

MAP-θ identity through U must pass (|chi2-122070.76|<1 at i_rad=0.5044)
before sampling. No G3 dest. No official 066 MAP. Wavelength loader only.
"""

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

from kinuv.constants import C_LIGHT_M_S, freq_to_velocity_kms
from kinuv.forward.sb import load_sb_template
from kinuv.infer.chart import PARAM_NAMES
from kinuv.infer.map import image_grid_for_vis, predict_binned
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
from kinuv.io.vis import (
    N_BIN,
    N_GUARD,
    TRIM_MARGIN_NATIVE,
    UV_BIN_M,
    VisData,
    _extend_axis,
    _trim_and_guard_indices,
    bin_uv_plane,
    cube_vopt_window_kms,
    optical_to_radio_kms,
)
from kinuv.likelihood.chi2 import chi2, empirical_s
from kinuv.response.spectral import bin_channels
from kinuv.runner.canfar import (
    ARTIFACT_G3_REL,
    RUNS_ROOT,
    artifact_dir_for_kind,
    utc_now,
    write_json,
    write_status,
)
from kinuv.runner.checkpoint import dual_checkpoint, flush_scratch_to_arc
from kinuv.runner.kind import ARTIFACT_KGAS007_REL, KIND_KGAS007
from kinuv.runner.log import (
    host_snapshot,
    install_crash_hook,
    logs_dir,
    rss_mb,
    setup_worker_logging,
)
from kinuv.runner.plots import mean_params, write_leftover_at_params, write_nuts_product_plots
from kinuv.runner.status_md import ping_status_ntfy, write_job_status_md
from kinuv.scratch import kinuv_scratch_root
from kinuv.transforms.nufft import BACKEND

MAP = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/"
    "kinuv-KGAS007-stage-a-map"
)
PRODUCT = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS007/"
    "kinuv-KGAS007-nuts"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS007.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS7/30kms/"
    "KGAS7_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS7/30kms/"
    "KGAS7_clipped_cube.fits"
)
CHI2_MAP_TARGET = 122070.76
CHI2_MAP_TOL = 1.0
I_RAD_TARGET = 0.5044
N_WARMUP = 200
N_SAMPLES = 600
N_CHAINS = 4


def _load_007(path: Path, cube_path: Path, cat_row: dict) -> tuple[VisData, dict]:
    """007 npz is (u,v) in wavelengths, no time/baseline. Do not invent those keys."""
    z = np.load(path, mmap_mode="r")
    if any(k not in z.files for k in ("u", "v", "vis", "weights", "freqs")):
        raise KeyError(f"{path} missing u/v/vis/weights/freqs")
    if "u_m" in z.files:
        raise RuntimeError("unexpected u_m on 007; refuse 066 copy")
    freqs_all = np.asarray(z["freqs"], dtype=np.float64).ravel()
    f_ref = float(np.mean(freqs_all))
    u_m = np.asarray(z["u"], dtype=np.float64) * C_LIGHT_M_S / f_ref
    v_m = np.asarray(z["v"], dtype=np.float64) * C_LIGHT_M_S / f_ref
    vel_all = freq_to_velocity_kms(freqs_all)
    v_lo_opt, v_hi_opt = cube_vopt_window_kms(cube_path)
    v_lo_line = float(optical_to_radio_kms(v_lo_opt))
    v_hi_line = float(optical_to_radio_kms(v_hi_opt))
    i0, i1, g0, g1, dv_native, extra_lo, extra_hi, dvel = _trim_and_guard_indices(
        vel_all,
        v_lo_line,
        v_hi_line,
        margin=int(TRIM_MARGIN_NATIVE),
        n_guard=int(N_GUARD),
    )
    sl = slice(i0, i1 + 1)
    vis = np.asarray(z["vis"][:, sl], dtype=np.complex128)
    weights = np.asarray(z["weights"][:, sl], dtype=np.float64)
    freqs_trim = freqs_all[sl]
    vel_trim = vel_all[sl]
    freqs_native = freqs_all[g0 : g1 + 1]
    vel_native = vel_all[g0 : g1 + 1]
    freqs_native, vel_native = _extend_axis(
        freqs_native, vel_native, extra_lo, extra_hi, dvel
    )
    u_m, v_m, vis, weights = bin_uv_plane(u_m, v_m, vis, weights, float(UV_BIN_M))
    vis_b, w_b, vel_b, freqs_b, _ = bin_channels(
        vis, weights, vel_trim, freqs_trim, int(N_BIN)
    )
    dv_kms = (
        float(np.median(np.abs(np.diff(vel_b))))
        if vel_b.size > 1
        else float(N_BIN) * dv_native
    )
    line_free = (vel_b < v_lo_line) | (vel_b > v_hi_line)
    s = empirical_s(vis_b, w_b, line_free)
    ra = float(cat_row.get("ra_deg", 0.0))
    dec = float(cat_row.get("dec_deg", 0.0))
    data = VisData(
        u_m=u_m,
        v_m=v_m,
        vis=vis_b,
        weights=w_b,
        freqs=freqs_b,
        vel=vel_b,
        freqs_native=freqs_native,
        vel_native=vel_native,
        n_bin=int(N_BIN),
        dv_kms=dv_kms,
        s=s,
        phase_dir_rad=np.array([np.radians(ra), np.radians(dec)], dtype=np.float64),
        line_free_mask=line_free,
        n_guard=int(N_GUARD),
        weights_native=weights,
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
    )
    meta = {
        "npz_keys": list(z.files),
        "uv_stored": "wavelengths",
        "uv_ref_hz": f_ref,
        "time_average": False,
        "uv_bin_m": float(UV_BIN_M),
    }
    return data, meta


def map_theta_chi2(data, tmpl, grid, params, i_rad):
    """MAP-θ vis χ² through predict_binned inside U (same i_rad)."""
    model = predict_binned(data, params, tmpl, grid, i_rad=i_rad, xla=True)
    return float(np.asarray(chi2(data.vis, model, data.weights, float(data.s))))


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
    p.add_argument("--run-id", default=os.environ.get("KINUV_RUN_ID", "kgas007-nuts"))
    p.add_argument("--pa-init", type=float, default=None)
    p.add_argument("--chain-id", type=int, default=None)
    args = p.parse_args()
    run_id = args.run_id
    kind = os.environ.get("KINUV_KIND", KIND_KGAS007)
    if "kgas007" not in str(kind).lower():
        raise SystemExit("007 worker requires KINUV_KIND containing kgas007")
    chain_raw = args.chain_id if args.chain_id is not None else os.environ.get("KINUV_CHAIN_ID")
    chain_id = int(chain_raw) if chain_raw not in (None, "", "0") else None
    if chain_id is not None and chain_id not in (1, 2, 3, 4):
        raise SystemExit("--chain-id must be 1..4")
    rec = json.loads((MAP / "stage_a_map.json").read_text())
    i_rad = float(rec["i_rad_frozen"])
    if abs(i_rad - I_RAD_TARGET) > 1.0e-3:
        raise SystemExit(f"i_rad={i_rad} is not the frozen 0.5044 coordinate")
    if args.pa_init is not None:
        pa_init = float(args.pa_init)
    else:
        pa_init = float(os.environ.get("KINUV_PA_INIT", rec["pa_deg"]))
    art_env = os.environ.get("KINUV_ARTIFACT_DIR")
    if art_env:
        artifact_dir = Path(art_env)
    else:
        artifact_dir = artifact_dir_for_kind(kind)
    if ARTIFACT_G3_REL in str(artifact_dir):
        raise SystemExit(
            "007 NUTS must not write docs/reviews/artifacts/2026-08-30-g3-nuts/"
        )
    if ARTIFACT_KGAS007_REL not in str(artifact_dir) and art_env is None:
        raise SystemExit("007 NUTS dest is docs/reviews/artifacts/2026-09-05-kgas007-nuts/")
    dest = RUNS_ROOT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "posteriors").mkdir(exist_ok=True)
    PRODUCT.mkdir(parents=True, exist_ok=True)
    if PRODUCT.resolve() == MAP.resolve():
        raise SystemExit("007 NUTS must not overwrite the Stage A MAP tree")
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
        "galaxy": "KGAS007",
        "kind": kind,
        "i_rad": i_rad,
    }
    write_status(run_id, state)
    write_json(
        PRODUCT / "status.json",
        {
            "galaxy": "KGAS007",
            "kind": kind,
            "run_id": run_id,
            "chain_id": chain_id,
            "stage_a_map": str(MAP / "stage_a_map.json"),
            "quote_inner_slope": False,
            "intervals_calibrated": False,
            "updated_at": utc_now(),
        },
    )
    cat = rec["catalogue_overrides"]
    if not NPZ.is_file():
        raise SystemExit(f"missing 007 vis {NPZ}")
    if not ICO.is_file():
        raise SystemExit(f"missing 007 Ico {ICO}")
    if not CUBE.is_file():
        raise SystemExit(f"missing 007 cube {CUBE}")
    data, load_meta = _load_007(NPZ, CUBE, cat)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO)
    params = {n: float(rec[n]) for n in PARAM_NAMES}
    log.info(
        "loaded 007 vis.shape=%s s=%.6f backend=%s rss_mb=%s load=%s",
        tuple(np.asarray(data.vis).shape),
        float(data.s),
        BACKEND,
        rss_mb(),
        json.dumps(load_meta),
    )
    if tuple(np.asarray(data.vis).shape) != (956, 66):
        raise SystemExit(f"007 vis shape {data.vis.shape} is not 956x66")
    c_map = map_theta_chi2(data, tmpl, grid, params, i_rad)
    ident = {
        "chi2_map_theta": c_map,
        "chi2_target": CHI2_MAP_TARGET,
        "abs_delta": abs(c_map - CHI2_MAP_TARGET),
        "i_rad": i_rad,
        "n_row": int(data.vis.shape[0]),
        "n_chan": int(data.vis.shape[1]),
        "pass": abs(c_map - CHI2_MAP_TARGET) < CHI2_MAP_TOL,
        "quote_inner_slope": False,
    }
    write_json(dest / "identity_chi2.json", ident)
    write_json(PRODUCT / "identity_chi2.json", ident)
    log.info("MAP-theta identity %s", json.dumps(ident))
    if not ident["pass"]:
        raise SystemExit(f"MAP-theta identity failed chi2={c_map}")

    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    start = dict(params)
    start["pa_deg"] = float(pa_init)
    z6 = sampled_z_from_physical(params_to_vec(start))

    import jax
    import jax.numpy as jnp

    U = make_potential(data, tmpl, grid, dx, dy, i_rad=i_rad)
    u_jit = jax.jit(U)
    state["step"] = "compile"
    state["rss_mb"] = rss_mb()
    write_status(run_id, state)
    log.info("compile U at MAP z6 i_rad=%.6f rss_mb=%s", i_rad, rss_mb())
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
                "galaxy": "KGAS007",
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
                "galaxy": "KGAS007",
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
    leftover_structured = None
    try:
        mean_p = mean_params({"sampler": "nuts", "draws": phys8})
        leftover_rec = write_leftover_at_params(
            mean_p,
            dest / "plots",
            data=data,
            tmpl=tmpl,
            grid=grid,
            i_rad=i_rad,
        )
        leftover_structured = bool(leftover_rec["leftover_chi2_structured"])
        leftover_rec["quote_inner_slope"] = False
        leftover_rec["leftover_gate"] = (
            "vel-structured" if leftover_structured else "not-vel-structured"
        )
        write_json(dest / "plots" / "leftover_chi2.json", leftover_rec)
    except Exception:
        log.exception("leftover at NUTS mean failed; omit leftover key")
    rec_out = product_record(
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
            f"007 headless NUTS PA {pa_init:.2f}; i_rad={i_rad:.4f}; "
            "4 chains x 600 draws; 16/50/84 not calibrated; "
            "do not quote inner dV/dr"
        ),
    )
    rec_out["mixing_pass"] = mix_pass
    rec_out["elapsed_s"] = time.perf_counter() - t0
    rec_out["backend"] = BACKEND
    rec_out["s"] = float(rec.get("s", data.s))
    rec_out["kind"] = kind
    rec_out["galaxy"] = "KGAS007"
    rec_out["i_rad_frozen"] = i_rad
    rec_out["quote_inner_slope"] = False
    rec_out["intervals_calibrated"] = False
    rec_out["dec_066_target_amended"] = True
    rec_out["infer_mock_recovery"] = "waived-this-card"
    write_json(dest / "posteriors" / "kgas007_nuts.json", rec_out)
    write_json(PRODUCT / "kgas007_nuts.json", rec_out)
    summary = {k: rec_out[k] for k in rec_out if k != "draws"}
    write_json(dest / "posteriors" / "summary.json", summary)
    write_json(PRODUCT / "summary.json", summary)
    try:
        plotted = write_nuts_product_plots(
            rec_out,
            dest,
            artifact_dir=artifact_dir,
            data=data,
            tmpl=tmpl,
            grid=grid,
            leftover=False,
            imaging=False,
            i_rad=i_rad,
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
            "sampler": rec_out["sampler"],
            "elapsed_s": rec_out["elapsed_s"],
            "rss_mb": rss_mb(),
            "galaxy": "KGAS007",
        },
    )
    (dest / ".trigger_complete").write_text(utc_now() + "\n")
    try:
        write_job_status_md(
            run_id=run_id,
            session_id=snap.get("session_id") or os.environ.get("SKAHA_SESSION_ID"),
            state="SUCCEEDED" if mix_pass else "COMPLETED_UNMIXED",
            mixing_pass=mix_pass,
            sampler=str(rec_out["sampler"]),
            elapsed_s=float(rec_out["elapsed_s"]),
            kind=kind,
            note=(
                "007 NUTS job. DEC-067 items 3-4 left as 066-only. "
                "Official MAP unchanged. Do not start G4"
            ),
        )
        ping_status_ntfy()
    except Exception:
        log.exception("STATUS.md patch failed")
    log.info(
        "done mixing_pass=%s sampler=%s elapsed_s=%.1f rss_mb=%s r_t_at_floor=%s",
        mix_pass,
        rec_out["sampler"],
        rec_out["elapsed_s"],
        rss_mb(),
        r_t_at_floor,
    )
    return 0 if mix_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
