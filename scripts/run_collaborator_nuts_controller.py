#!/usr/bin/env python3
"""Durable detached controller for collaborator-delivery NUTS workers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def cpu_set_for(pid: int) -> str:
    """Return a taskset-compatible description of an adopted worker affinity."""
    cpus = sorted(os.sched_getaffinity(pid))
    ranges = []
    start = previous = cpus[0]
    for cpu in cpus[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def terminal_record(job: dict, exit_code: int):
    return {
        name: job[name]
        for name in ("target", "chain", "seed", "pid", "started_utc", "cpu_set")
    } | {"exit_code": exit_code, "completed_utc": utc_now()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("KGAS066", "KGAS007"),
        default=("KGAS066", "KGAS007"),
    )
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--controller-log", type=Path, default=None)
    parser.add_argument("--worker-log-root", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument(
        "--cpu-sets",
        default="0-3,4-7,8-11,12-15,16-19,20-23,24-27,28-31",
        help="comma-separated taskset CPU ranges, one per concurrent worker",
    )
    args = parser.parse_args()
    cpu_sets = [item.strip() for item in args.cpu_sets.split(",") if item.strip()]
    if len(cpu_sets) < args.max_workers:
        raise ValueError("cpu-sets must provide one affinity set per worker")
    args.durable_root.mkdir(parents=True, exist_ok=True)
    controller_log = args.controller_log or args.durable_root / "controller.log"
    controller_log.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    seed_bases = {"KGAS066": 9100, "KGAS007": 9200}
    for target in args.targets:
        selected = args.map_root / target / "selected_map.json"
        if not selected.is_file():
            raise FileNotFoundError(selected)
        for chain in range(1, 5):
            jobs.append({"target": target, "chain": chain, "seed": seed_bases[target] + chain, "selected": selected})
    active = {}
    completed = []
    status_path = args.status_file or args.durable_root / "controller_status.json"
    queued = []
    for job in jobs:
        worker_status = args.durable_root / job["target"] / f"chain-{job['chain']}" / "status.json"
        state = json.loads(worker_status.read_text(encoding="utf-8")) if worker_status.is_file() else None
        if state and state.get("state") in {"SUCCEEDED", "FAILED"}:
            job.update({
                "pid": int(state.get("pid", -1)),
                "started_utc": state.get("started_utc", "unknown"),
                "cpu_set": state.get("cpu_set", "completed"),
            })
            completed.append(terminal_record(job, int(state.get("exit_code", state["state"] != "SUCCEEDED"))))
        elif state and process_alive(int(state.get("pid", -1))):
            pid = int(state["pid"])
            job.update({
                "pid": pid,
                "process": None,
                "log_handle": None,
                "started_utc": state.get("started_utc", "unknown"),
                "cpu_set": cpu_set_for(pid),
            })
            active[(job["target"], job["chain"])] = job
        else:
            queued.append(job)
    jobs = queued
    with controller_log.open("a", encoding="ascii") as log:
        log.write(
            f"{utc_now()} controller start pid={os.getpid()} queued={len(jobs)} "
            f"adopted={len(active)} completed={len(completed)}\n"
        )
        log.flush()
        while jobs or active:
            while jobs and len(active) < args.max_workers:
                job = jobs.pop(0)
                target = job["target"]
                chain = job["chain"]
                worker_root = args.durable_root / target
                worker_root.mkdir(parents=True, exist_ok=True)
                stdout_root = args.worker_log_root or worker_root
                worker_stdout = stdout_root / target / f"chain-{chain}.stdout.log" if args.worker_log_root else stdout_root / f"chain-{chain}.stdout.log"
                worker_stdout.parent.mkdir(parents=True, exist_ok=True)
                worker_log = worker_stdout.open("ab")
                worker_command = [
                    str(args.python), "scripts/run_collaborator_nuts_chain.py",
                    "--selected-map", str(job["selected"]), "--chain-id", str(chain),
                    "--seed", str(job["seed"]), "--scratch", str(args.scratch_root / target),
                    "--durable", str(worker_root), "--warmup", str(args.warmup),
                    "--samples", str(args.samples), "--chunk", "100",
                    "--target-accept", "0.90", "--warmup-max-tree-depth", "7",
                    "--max-tree-depth", "8",
                ]
                if args.worker_log_root:
                    worker_command.extend(["--log-dir", str(args.worker_log_root / target / f"chain-{chain}")])
                occupied = {row["cpu_set"] for row in active.values()}
                cpu_set = next(item for item in cpu_sets if item not in occupied)
                command = ["taskset", "-c", cpu_set, *worker_command]
                env = os.environ.copy()
                env.update({"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4", "TF_NUM_INTRAOP_THREADS": "4", "TF_NUM_INTEROP_THREADS": "1", "JAX_NUM_THREADS": "4", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4", "PYTHONUNBUFFERED": "1"})
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], env=env, stdin=subprocess.DEVNULL, stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update({"pid": process.pid, "process": process, "log_handle": worker_log, "started_utc": utc_now(), "cpu_set": cpu_set})
                active[(target, chain)] = job
                log.write(f"{utc_now()} launch target={target} chain={chain} seed={job['seed']} pid={process.pid} cpu_set={cpu_set}\n")
                log.flush()
            time.sleep(10.0)
            for key, job in list(active.items()):
                worker_status = args.durable_root / job["target"] / f"chain-{job['chain']}" / "status.json"
                state = json.loads(worker_status.read_text(encoding="utf-8")) if worker_status.is_file() else {}
                code = None
                if state.get("state") in {"SUCCEEDED", "FAILED"}:
                    code = int(state.get("exit_code", state["state"] != "SUCCEEDED"))
                elif job["process"] is not None:
                    code = job["process"].poll()
                elif not process_alive(job["pid"]):
                    code = 1
                if code is None:
                    continue
                if job["log_handle"] is not None:
                    job["log_handle"].close()
                completed.append(terminal_record(job, code))
                del active[key]
                log.write(f"{utc_now()} exit target={key[0]} chain={key[1]} pid={job['pid']} code={code}\n")
                log.flush()
            write_json(status_path, {
                "state": "RUNNING" if jobs or active else "SUCCEEDED" if all(row["exit_code"] == 0 for row in completed) else "FAILED",
                "controller_pid": os.getpid(), "updated_utc": utc_now(),
                "queued": [{k: row[k] for k in ("target", "chain", "seed")} for row in jobs],
                "active": [{k: row[k] for k in ("target", "chain", "seed", "pid", "started_utc", "cpu_set")} for row in active.values()],
                "completed": completed,
            })
        result = 0 if all(row["exit_code"] == 0 for row in completed) else 1
        log.write(f"{utc_now()} controller complete exit_code={result}\n")
        log.flush()
        return result


if __name__ == "__main__":
    raise SystemExit(main())
