#!/usr/bin/env python3
"""Merge 3–4 1-chain CPU NUTS shards. ``sampler: nuts`` needs four finite + mix."""

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
    NUTS_SAMPLER,
    chain_physically_ok,
    mixing_ok,
    mixing_sampled,
    physical_sampled_from_z6,
    product_record,
)
from kinuv.runner.canfar import RUNS_ROOT, utc_now, write_json  # noqa: E402
from kinuv.runner.kind import ARTIFACT_G3  # noqa: E402
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
    p.add_argument(
        "run_dirs",
        nargs="+",
        help="3 or 4 run dirs; chain id is 1..n in order (or --chain-ids)",
    )
    p.add_argument("--artifact-dir", default=str(ARTIFACT_G3))
    p.add_argument("--pa-init", type=float, default=None)
    p.add_argument(
        "--chain-ids",
        default=None,
        help="comma list matching run_dirs (default 1,2,3[,4])",
    )
    args = p.parse_args()
    if len(args.run_dirs) < 3:
        raise SystemExit("need at least 3 run dirs")
    artifact_dir = Path(args.artifact_dir)
    if args.chain_ids:
        chain_ids = [int(x) for x in str(args.chain_ids).split(",")]
        if len(chain_ids) != len(args.run_dirs):
            raise SystemExit("--chain-ids length must match run_dirs")
    else:
        chain_ids = list(range(1, len(args.run_dirs) + 1))
    t0 = time.perf_counter()
    map_path = Path(
        "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
        "kinuv-KGAS066-uvsign-map/stage_a_map.json"
    )
    rec_map = json.loads(map_path.read_text())
    pa_init = float(args.pa_init if args.pa_init is not None else rec_map["pa_deg"])
    dx, dy = rec_map["dx_arcsec"], rec_map["dy_arcsec"]
    kept_z = []
    kept_elapsed = []
    dropped = []
    for raw, cid in zip(args.run_dirs, chain_ids, strict=True):
        dest = Path(raw)
        if not dest.is_absolute():
            dest = RUNS_ROOT / raw
        z6, el = _load_chain(dest, cid)
        phys = physical_sampled_from_z6(z6[None, ...], dx, dy)[0]
        if not chain_physically_ok(phys):
            dropped.append({"run": str(dest), "chain_id": cid})
            continue
        kept_z.append(z6)
        kept_elapsed.append(el)
    n_kept = len(kept_z)
    if n_kept == 0:
        raise SystemExit("all shards dropped as exploded/non-finite")
    z_draws = np.stack(kept_z, axis=0)
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    mix = mixing_sampled(phys8)
    mix_pass = bool(
        n_kept == 4
        and mixing_ok(mix, rhat_max=1.01, ess_min=400.0, ess_tail_min=400.0)
    )
    merge_s = time.perf_counter() - t0
    finite = [e for e in kept_elapsed if np.isfinite(e)]
    t_run = (max(finite) if finite else float("nan")) + merge_s
    rt = np.asarray(phys8)[..., PARAM_NAMES.index("r_t_arcsec")]
    rec = product_record(
        draws8=phys8,
        mix=mix,
        pa_init_deg=pa_init,
        dx_map=dx,
        dy_map=dy,
        autodiff_ok=True,
        mixing_pass=mix_pass,
        leftover_chi2_structured=None,
        r_t_at_floor=bool(abs(float(np.median(rt)) - 0.5) <= 0.01),
        mean_num_steps=float("nan"),
        eval_s=float("nan"),
        note=(
            "066 CPU NUTS merge; leftover unevaluated; 16/50/84 not calibrated; "
            "do not quote inner dV/dr"
        ),
    )
    rec["mixing_pass"] = mix_pass
    rec["kind"] = "nuts"
    rec["chain_elapsed_s"] = kept_elapsed
    rec["merge_s"] = merge_s
    rec["t_run_s"] = t_run
    rec["dropped_shards"] = dropped
    rec["n_kept"] = n_kept
    state = "SUCCEEDED" if rec["sampler"] == NUTS_SAMPLER else "COMPLETED_UNMIXED"
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
            "chain_elapsed_s": kept_elapsed,
            "merge_s": merge_s,
            "mixing_pass": mix_pass,
            "state": state,
            "sampler": rec["sampler"],
            "n_kept": n_kept,
            "dropped_shards": dropped,
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
                "n_kept": n_kept,
                "dropped": len(dropped),
                "t_run_s": t_run,
            },
            indent=2,
        )
    )
    return 0 if rec["sampler"] == NUTS_SAMPLER else 1


if __name__ == "__main__":
    raise SystemExit(main())
