#!/usr/bin/env python3
"""Race fast and legacy posterior campaigns, finalize the first accepted result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


TARGETS = ("KGAS066", "KGAS007")
CANFAR = Path("/arc/home/thbrown/.local/bin/canfar")
CANFAR_PYTHON = Path("/usr/bin/python3")
SESSION_RE = re.compile(r"\(ID:\s*([A-Za-z0-9]+)\)")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def campaign_status(root: Path, required_draws: dict[str, int]) -> dict:
    rows = []
    for target in TARGETS:
        for chain in range(1, 5):
            status_path = root / target / f"chain-{chain}" / "status.json"
            if not status_path.is_file():
                rows.append(
                    {
                        "target": target,
                        "chain": chain,
                        "state": "QUEUED",
                        "completed_warmup": 0,
                        "completed_draws": 0,
                        "required_draws": required_draws[target],
                        "updated_utc": None,
                    }
                )
                continue
            item = json.loads(status_path.read_text(encoding="utf-8"))
            state = item.get("state", "UNKNOWN")
            completed = int(item.get("completed_draws", 0))
            if state == "SUCCEEDED" and completed < required_draws[target]:
                state = "AWAITING_EXTENSION"
            rows.append(
                {
                    "target": target,
                    "chain": chain,
                    "state": state,
                    "completed_warmup": int(item.get("completed_warmup", 0)),
                    "completed_draws": completed,
                    "required_draws": required_draws[target],
                    "updated_utc": item.get("updated_utc", item.get("completed_utc")),
                }
            )
    failed = [row for row in rows if row["state"] == "FAILED"]
    complete = all(row["state"] == "SUCCEEDED" for row in rows)
    return {"state": "FAILED" if failed else "SUCCEEDED" if complete else "RUNNING", "chains": rows}


def controller_document(status: dict) -> dict:
    completed = []
    active = []
    queued = []
    for row in status["chains"]:
        base = {"target": row["target"], "chain": row["chain"]}
        if row["state"] == "SUCCEEDED":
            completed.append({**base, "exit_code": 0})
        elif row["state"] == "FAILED":
            completed.append({**base, "exit_code": 1})
        elif row["state"] == "QUEUED":
            queued.append(base)
        else:
            active.append({**base, "state": row["state"]})
    return {"state": status["state"], "completed": completed, "active": active, "queued": queued}


def run_logged(command: list[str], *, cwd: Path, log) -> int:
    log.write(f"{now()} command={' '.join(command)}\n")
    log.flush()
    result = subprocess.run(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={
            **os.environ,
            "TERM": "dumb",
            "NO_COLOR": "1",
            "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        },
    )
    return int(result.returncode)


def finalize_attempt(
    name: str,
    target: str,
    root: Path,
    *,
    repo: Path,
    python: Path,
    map_root: Path,
    output_root: Path,
    log,
) -> tuple[bool, dict]:
    posterior = output_root / name / "posterior"
    command = [
        str(python),
        "scripts/finalize_collaborator_posterior.py",
        "--map-root",
        str(map_root),
        "--nuts-root",
        str(root),
        "--output-root",
        str(posterior),
        "--targets",
        target,
    ]
    if run_logged(command, cwd=repo, log=log) != 0:
        return False, {"state": "FINALIZATION_FAILED", "posterior_root": str(posterior)}
    summary = json.loads((posterior / target / "summary.json").read_text(encoding="utf-8"))
    accepted = bool(summary["gates"]["accepted"])
    return accepted, {
        "state": "ACCEPTED" if accepted else "GATES_FAILED",
        "posterior_root": str(posterior),
        "target": target,
        "gates": summary["gates"],
    }


def extension_eligible(gates: dict) -> bool:
    return bool(
        gates["divergences"] == 0
        and gates["bfmi_min"] >= 0.30
        and gates["tree_depth_saturation_count"] == 0
    )


def submit_extension(target: str, chain: int, root: Path, commit: str, repo: Path) -> dict:
    name = f"kinuv-{target}-{commit[:6]}-c{chain}-extend1000"
    command = [
        str(CANFAR_PYTHON),
        str(CANFAR),
        "create",
        "headless",
        "skaha/astroml:latest",
        "--name",
        name,
        "--",
        "/bin/bash",
        str(repo / "scripts/run_collaborator_nuts_chain_headless.sh"),
        target,
        str(chain),
        str(root),
        commit,
        "200",
        "1000",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = SESSION_RE.search(output)
    return {
        "target": target,
        "chain": chain,
        "name": name,
        "session_id": match.group(1) if match else None,
        "returncode": int(result.returncode),
        "ok": result.returncode == 0 and match is not None,
        "submitted_utc": now(),
    }


def dispatch_session_ids(path: Path, target: str) -> list[str]:
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = list(document.get("sessions", [])) + list(document.get("replacement_sessions", []))
    replacement = document.get("postprocess_replacement")
    if replacement:
        entries.append(replacement)
    return sorted(
        {
            row["session_id"]
            for row in entries
            if row.get("session_id") and row.get("target") == target
        }
    )


def terminate_sessions(session_ids: list[str]) -> dict:
    if not session_ids:
        return {"session_ids": [], "returncode": 0, "terminated": True}
    command = [str(CANFAR_PYTHON), str(CANFAR), "delete", "--force", *session_ids]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
    )
    return {
        "session_ids": session_ids,
        "returncode": int(result.returncode),
        "terminated": result.returncode == 0,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--race-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--map-candidate-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    state_path = args.race_root / "RACE_STATUS.json"
    log_path = args.race_root / "race.log"
    state = {
        "schema_version": "kinuv-posterior-race-v1",
        "state": "RUNNING",
        "started_utc": now(),
        "pid": os.getpid(),
        "code_commit": args.code_commit,
        "fast_required_draws": {target: 500 for target in TARGETS},
        "fast_extensions": [],
        "evaluations": {},
        "target_winners": {},
    }
    write_json(state_path, state)
    args.race_root.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="ascii") as log:
        while True:
            fast = campaign_status(args.fast_root, state["fast_required_draws"])
            legacy = campaign_status(args.legacy_root, {target: 1000 for target in TARGETS})
            state.update({"updated_utc": now(), "fast": fast, "legacy": legacy})
            write_json(state_path, state)
            log.write(
                f"{now()} heartbeat fast={fast['state']} "
                f"fast_draws={sum(x['completed_draws'] for x in fast['chains'])} "
                f"legacy={legacy['state']} "
                f"legacy_draws={sum(x['completed_draws'] for x in legacy['chains'])}\n"
            )
            log.flush()
            statuses = {"fast": (args.fast_root, fast), "legacy": (args.legacy_root, legacy)}
            for target in TARGETS:
                if target in state["target_winners"]:
                    continue
                for name in ("fast", "legacy"):
                    root, status = statuses[name]
                    target_rows = [row for row in status["chains"] if row["target"] == target]
                    if not all(row["state"] == "SUCCEEDED" for row in target_rows):
                        continue
                    signature = [row["completed_draws"] for row in target_rows]
                    evaluation_key = f"{name}:{target}"
                    if state["evaluations"].get(evaluation_key, {}).get("draw_signature") == signature:
                        continue
                    accepted, evaluation = finalize_attempt(
                        name,
                        target,
                        root,
                        repo=repo,
                        python=args.python,
                        map_root=args.map_root,
                        output_root=args.race_root,
                        log=log,
                    )
                    evaluation["draw_signature"] = signature
                    evaluation["evaluated_utc"] = now()
                    state["evaluations"][evaluation_key] = evaluation
                    write_json(state_path, state)
                    if accepted:
                        candidate = args.race_root / name / "posterior-candidate"
                        render = [
                            str(args.python),
                            "scripts/render_collaborator_posterior.py",
                            "--map-candidate-root",
                            str(args.map_candidate_root),
                            "--posterior-root",
                            evaluation["posterior_root"],
                            "--output-root",
                            str(candidate),
                            "--targets",
                            target,
                        ]
                        if run_logged(render, cwd=repo, log=log) != 0:
                            state.update({"state": "RENDER_FAILED", "failed_target": target, "completed_utc": now()})
                            write_json(state_path, state)
                            return 1
                        loser = "legacy" if name == "fast" else "fast"
                        dispatch = (args.legacy_root if loser == "legacy" else args.fast_root) / "dispatch.json"
                        loser_ids = dispatch_session_ids(dispatch, target)
                        if loser == "fast":
                            loser_ids.extend(
                                row["session_id"]
                                for row in state["fast_extensions"]
                                if row.get("session_id") and row["target"] == target
                            )
                        state["target_winners"][target] = {
                            "campaign": name,
                            "candidate_root": str(candidate / target),
                            "loser_termination": terminate_sessions(sorted(set(loser_ids))),
                            "completed_utc": now(),
                        }
                        write_json(state_path, state)
                        break
                    if (
                        name == "fast"
                        and extension_eligible(evaluation["gates"])
                        and state["fast_required_draws"][target] == 500
                    ):
                        submissions = [
                            submit_extension(target, chain, args.fast_root, args.code_commit, repo)
                            for chain in range(1, 5)
                        ]
                        state["fast_extensions"].extend(submissions)
                        if not all(row["ok"] for row in submissions):
                            state.update({"state": "EXTENSION_SUBMIT_FAILED", "failed_target": target, "completed_utc": now()})
                            write_json(state_path, state)
                            return 1
                        state["fast_required_draws"][target] = 1000
                        write_json(state_path, state)
                        break
            if len(state["target_winners"]) == len(TARGETS):
                state.update({"state": "WINNERS_READY", "completed_utc": now()})
                write_json(state_path, state)
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
