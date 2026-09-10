#!/usr/bin/env python3
"""Dispatch grouped real-visibility training refits to flexible CANFAR sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[2]
CANFAR = Path.home() / ".local/bin/canfar"
CANFAR_PYTHON = Path("/usr/bin/python3")
SESSION_RE = re.compile(r"\(ID:\s*([A-Za-z0-9]+)\)")
TARGETS = ("KGAS066", "KGAS007")
MODEL_SOURCE_PATHS = (
    "src/kinuv/infer/unified.py",
    "src/kinuv/profiles/unified.py",
    "experiments/unified_foundation/unified_map_runner.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_winners(values: list[str]) -> dict[str, Path]:
    winners = {}
    for value in values:
        target, separator, raw_path = value.partition("=")
        if not separator or target not in TARGETS or target in winners:
            raise ValueError("--winner must specify each target once as TARGET=/path/result.json")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        winners[target] = path
    if set(winners) != set(TARGETS):
        raise ValueError("winners for KGAS066 and KGAS007 are required")
    return winners


def submit(*, name: str, image: str, command: list[str], dry_run: bool) -> dict:
    argv = [str(CANFAR_PYTHON), str(CANFAR), "create", "headless", image, "--name", name, "--", *command]
    if dry_run:
        return {"ok": True, "session_id": None, "argv": argv, "dry_run": True}
    completed = subprocess.run(
        argv, check=False, capture_output=True, text=True, timeout=240,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
    )
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    match = SESSION_RE.search(combined)
    return {
        "ok": completed.returncode == 0 and match is not None,
        "session_id": match.group(1) if match else None,
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "argv": argv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--winner", action="append", required=True)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--image", default="skaha/astroml:latest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    winners = parse_winners(args.winner)
    run_root = args.run_root.resolve()
    dispatch_path = run_root / "dispatch.json"
    if dispatch_path.exists() and not args.dry_run:
        raise FileExistsError(dispatch_path)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if branch != "dev":
        raise RuntimeError(f"dispatch requires dev branch, found {branch!r}")
    tracked_dirty = subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=REPO, text=True
    ).splitlines()
    protected = {
        "src/kinuv/infer/unified.py",
        "src/kinuv/profiles/unified.py",
        "experiments/unified_foundation/unified_map_runner.py",
        "experiments/unified_foundation/grouped_real_visibility_benchmark.py",
    }
    overlap = sorted(set(tracked_dirty) & protected)
    if overlap:
        raise RuntimeError(f"refusing dispatch with relevant tracked changes: {overlap}")
    winner_commits = {
        json.loads(path.read_text(encoding="utf-8"))["git"]["commit"]
        for path in winners.values()
    }
    if len(winner_commits) != 1:
        raise RuntimeError(f"winner model commits differ: {sorted(winner_commits)}")
    winner_commit = next(iter(winner_commits))
    same_model = subprocess.run(
        ["git", "diff", "--quiet", winner_commit, commit, "--", *MODEL_SOURCE_PATHS],
        cwd=REPO,
        check=False,
    )
    if same_model.returncode != 0:
        raise RuntimeError(f"archived model source differs from winner commit {winner_commit}")
    worker = REPO / "experiments/unified_foundation/grouped_real_refit_worker.py"
    shell = REPO / "experiments/unified_foundation/grouped_real_refit_headless.sh"
    worker_hash = sha256(worker)
    record = {
        "schema_version": "kinuv-unified-real-grouped-refit-dispatch-v1",
        "state": "DISPATCHING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "winner_model_commit": winner_commit,
        "branch": branch,
        "archived_relevant_source_clean": True,
        "excluded_unrelated_tracked_changes": tracked_dirty,
        "source": {
            "worker": {"path": str(worker), "sha256": worker_hash},
            "headless_shell": {"path": str(shell), "sha256": sha256(shell)},
            "launcher": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        },
        "conditioning": {
            "unified": "14D kinematics/geometry/flux/dispersion refit on each training partition from full-data MAP; morphology fixed from full cube",
            "kinms": "frozen full-data best fit",
            "fully_end_to_end_leakage_free": False,
        },
        "execution": {"topology": "one flexible session per target-fold", "n_folds": 5, "maxiter": args.maxiter},
        "run_root": str(run_root),
        "winners": {target: str(path) for target, path in winners.items()},
        "sessions": [],
    }
    atomic_json(dispatch_path, record)
    tag = re.sub(r"[^A-Za-z0-9]+", "-", run_root.name).strip("-")[-9:]
    for target in TARGETS:
        for fold in range(5):
            name = f"kinuv-rrefit-{target[-3:]}-f{fold}-{commit[:7]}-{tag}"
            result = submit(
                name=name, image=args.image,
                command=["/bin/bash", str(shell), target, str(fold), str(run_root), str(winners[target]), commit, str(worker), worker_hash, str(args.maxiter)],
                dry_run=args.dry_run,
            )
            record["sessions"].append({"target": target, "fold_id": fold, "name": name, **result})
            atomic_json(dispatch_path, record)
            if not result["ok"]:
                record["state"] = "PARTIAL_SUBMIT_FAILURE"
                atomic_json(dispatch_path, record)
                return 1
    record["state"] = "DRY_RUN" if args.dry_run else "SUBMITTED"
    atomic_json(dispatch_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
