#!/usr/bin/env python3
"""Wait for detached NUTS completion, then finalize a posterior candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")
    os.replace(temporary, path)


def aggregate_chain_statuses(nuts_root: Path, targets: tuple[str, ...]) -> dict:
    completed = []
    active = []
    queued = []
    for target in targets:
        for chain in range(1, 5):
            path = nuts_root / target / f"chain-{chain}" / "status.json"
            if not path.is_file():
                queued.append({"target": target, "chain": chain})
                continue
            item = json.loads(path.read_text(encoding="utf-8"))
            row = {
                key: item.get(key)
                for key in ("target", "chain_id", "seed", "pid", "started_utc")
            }
            row["chain"] = row.pop("chain_id")
            if item.get("state") in {"SUCCEEDED", "FAILED"}:
                row["exit_code"] = int(item.get("exit_code", item["state"] != "SUCCEEDED"))
                row["completed_utc"] = item.get("completed_utc")
                completed.append(row)
            else:
                row["state"] = item.get("state", "UNKNOWN")
                active.append(row)
    terminal_count = len(completed)
    failed = any(row["exit_code"] != 0 for row in completed)
    state = "FAILED" if failed else "SUCCEEDED" if terminal_count == 4 * len(targets) else "RUNNING"
    return {"state": state, "completed": completed, "active": active, "queued": queued}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--nuts-root", type=Path, required=True)
    parser.add_argument("--map-candidate-root", type=Path, required=True)
    parser.add_argument("--posterior-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--controller-status", type=Path, action="append", default=[])
    parser.add_argument("--targets", nargs="+", default=("KGAS066", "KGAS007"))
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    status_path = args.status_file or args.nuts_root.parent / "postprocess_status.json"
    log_path = args.log_file or args.nuts_root.parent / "postprocess.log"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state = {"state": "WAITING_FOR_NUTS", "pid": os.getpid(), "started_utc": now()}
    write_status(status_path, state)
    with log_path.open("a", encoding="ascii") as log:
        while True:
            status_paths = args.controller_status
            if not status_paths:
                controller = aggregate_chain_statuses(args.nuts_root, tuple(args.targets))
                controller["chain_status_source"] = "direct"
            else:
                controllers = [json.loads(path.read_text(encoding="utf-8")) for path in status_paths if path.is_file()]
                if len(controllers) != len(status_paths):
                    controller = {"state": "RUNNING", "completed": [], "active": [], "queued": []}
                    controllers = []
                if not controllers:
                    write_status(args.nuts_root / "controller_status.json", controller)
                    log.write(f"{now()} heartbeat state=RUNNING completed=0 active=0 queued=0\n")
                    log.flush()
                    time.sleep(60.0)
                    continue
                states = {item["state"] for item in controllers}
                controller = {
                    "state": "FAILED" if "FAILED" in states else "SUCCEEDED" if states == {"SUCCEEDED"} else "RUNNING",
                    "completed": [row for item in controllers for row in item["completed"]],
                    "active": [row for item in controllers for row in item["active"]],
                    "queued": [row for item in controllers for row in item["queued"]],
                    "target_controller_statuses": [str(path) for path in status_paths],
                }
            write_status(args.nuts_root / "controller_status.json", controller)
            log.write(f"{now()} heartbeat state={controller['state']} completed={len(controller['completed'])} active={len(controller['active'])} queued={len(controller['queued'])}\n")
            log.flush()
            if controller["state"] != "RUNNING":
                break
            time.sleep(60.0)
        if controller["state"] != "SUCCEEDED":
            state.update({"state": "BLOCKED_NUTS_FAILED", "completed_utc": now()})
            write_status(status_path, state)
            return 1
        commands = [
            [str(args.python), "scripts/finalize_collaborator_posterior.py", "--map-root", str(args.map_root), "--nuts-root", str(args.nuts_root), "--output-root", str(args.posterior_root)],
            [str(args.python), "scripts/render_collaborator_posterior.py", "--map-candidate-root", str(args.map_candidate_root), "--posterior-root", str(args.posterior_root), "--output-root", str(args.candidate_root)],
        ]
        state["state"] = "POSTPROCESSING"
        write_status(status_path, state)
        for command in commands:
            result = subprocess.run(command, cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "TERM": "dumb", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
            if result.returncode != 0:
                state.update({"state": "POSTPROCESS_FAILED", "exit_code": result.returncode, "failed_command": command, "completed_utc": now()})
                write_status(status_path, state)
                return result.returncode
        state.update({"state": "POSTERIOR_CANDIDATE_READY", "exit_code": 0, "completed_utc": now(), "candidate_root": str(args.candidate_root)})
        write_status(status_path, state)
        log.write(f"{now()} postprocess complete\n")
        log.flush()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
