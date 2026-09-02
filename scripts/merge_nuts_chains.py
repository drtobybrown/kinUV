#!/usr/bin/env python3
"""Merge four 1-chain NUTS shards. Mixing ESS>400. Does not steal KGAS066-latest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinuv.infer.chart import PARAM_NAMES  # noqa: E402
from kinuv.infer.nuts import (  # noqa: E402
    mixing_ok,
    mixing_sampled,
    physical_sampled_from_z6,
    product_record,
)
from kinuv.runner.canfar import RUNS_ROOT, utc_now, write_json  # noqa: E402
from kinuv.runner.kind import (  # noqa: E402
    ARTIFACT_G3_REL,
    ARTIFACT_GPU,
    SERIAL_CPU_WALL_S,
    TENX_WALL_S,
)
from kinuv.runner.plots import write_nuts_product_plots  # noqa: E402


def _load_chain(path: Path, chain_id: int) -> tuple[np.ndarray, float]:
    npz = path / "checkpoints" / f"chain_{chain_id}.npz"
    if not npz.is_file():
        raise SystemExit(f"missing {npz}")
    data = np.load(npz)
    z6 = np.asarray(data["z6"], dtype=np.float64)
    if z6.ndim == 3:
        z6 = z6[0]
    rec_path = path / "logs" / f"chain_{chain_id}.json"
    elapsed = float("nan")
    if rec_path.is_file():
        elapsed = float(json.loads(rec_path.read_text())["elapsed_s"])
    return z6, elapsed


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs=4, help="four run dirs, chain 1..4 order")
    p.add_argument(
        "--artifact-dir",
        default=str(ARTIFACT_GPU),
        help="must not be 2026-08-30-g3-nuts",
    )
    args = p.parse_args()
    artifact_dir = Path(args.artifact_dir)
    if ARTIFACT_G3_REL in str(artifact_dir):
        raise SystemExit("merge must not write 2026-08-30-g3-nuts")
    t0 = time.perf_counter()
    z_parts = []
    elapsed = []
    for i, raw in enumerate(args.run_dirs, start=1):
        dest = Path(raw)
        if not dest.is_absolute():
            dest = RUNS_ROOT / raw
        z6, el = _load_chain(dest, i)
        z_parts.append(z6)
        elapsed.append(el)
    z_draws = np.stack(z_parts, axis=0)
    map_path = Path(
        "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
        "kinuv-KGAS066-uvsign-map/stage_a_map.json"
    )
    rec_map = json.loads(map_path.read_text())
    dx, dy = rec_map["dx_arcsec"], rec_map["dy_arcsec"]
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    mix_pass = mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)
    merge_s = time.perf_counter() - t0
    finite = [e for e in elapsed if np.isfinite(e)]
    t_run = (max(finite) if finite else float("nan")) + merge_s
    ratio = SERIAL_CPU_WALL_S / t_run if t_run and t_run > 0 else float("nan")
    tenx = bool(np.isfinite(t_run) and t_run <= TENX_WALL_S)
    rt = np.asarray(phys8)[..., PARAM_NAMES.index("r_t_arcsec")]
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=float(rec_map["pa_deg"]),
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=True,
        mixing_pass=mix_pass,
        leftover_chi2_structured=False,
        r_t_at_floor=bool(abs(float(np.median(rt)) - 0.5) <= 0.01),
        mean_num_steps=float("nan"),
        eval_s=float("nan"),
        note=(
            "GPU 4×1-chain merge; 16/50/84 not calibrated; "
            "speed smoke not a new official MAP; do not quote inner dV/dr"
        ),
    )
    rec["mixing_pass"] = mix_pass
    rec["kind"] = "nuts-gpu"
    rec["chain_elapsed_s"] = elapsed
    rec["merge_s"] = merge_s
    rec["t_run_s"] = t_run
    rec["serial_cpu_wall_s"] = SERIAL_CPU_WALL_S
    rec["wall_speedup"] = ratio
    rec["tenx_wall"] = tenx
    rec["submit_to_done_s"] = None
    state = "SUCCEEDED" if mix_pass else "COMPLETED_UNMIXED"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "kgas066_nuts.json", rec)
    write_json(
        artifact_dir / "summary.json",
        {k: rec[k] for k in rec if k != "draws"},
    )
    write_json(
        artifact_dir / "wall.json",
        {
            "t_run_s": t_run,
            "serial_cpu_wall_s": SERIAL_CPU_WALL_S,
            "tenx_wall_s": TENX_WALL_S,
            "wall_speedup": ratio,
            "tenx_wall": tenx,
            "chain_elapsed_s": elapsed,
            "merge_s": merge_s,
            "mixing_pass": mix_pass,
            "state": state,
            "utc": utc_now(),
        },
    )
    dest0 = Path(args.run_dirs[0])
    if not dest0.is_absolute():
        dest0 = RUNS_ROOT / args.run_dirs[0]
    try:
        write_nuts_product_plots(
            rec,
            dest0,
            artifact_dir=artifact_dir,
            leftover=False,
            imaging=False,
        )
    except Exception:
        pass
    print(
        json.dumps(
            {
                "state": state,
                "sampler": rec["sampler"],
                "mixing_pass": mix_pass,
                "t_run_s": t_run,
                "tenx_wall": tenx,
                "wall_speedup": ratio,
            },
            indent=2,
        )
    )
    return 0 if mix_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
