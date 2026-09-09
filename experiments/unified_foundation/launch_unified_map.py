#!/usr/bin/env python3
"""Submit frozen-source unified MAP multistarts as flexible CANFAR sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
CANFAR = Path.home() / ".local/bin/canfar"
CANFAR_PYTHON = Path("/usr/bin/python3")
SESSION_RE = re.compile(r"\(ID:\s*([A-Za-z0-9]+)\)")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def submit(*, name: str, image: str, command: list[str], dry_run: bool) -> dict:
    argv = [
        str(CANFAR_PYTHON),
        str(CANFAR),
        "create",
        "headless",
        image,
        "--name",
        name,
        "--",
        *command,
    ]
    if dry_run:
        return {"ok": True, "session_id": None, "argv": argv, "dry_run": True}
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("KGAS066", "KGAS007"),
        default=("KGAS066", "KGAS007"),
    )
    parser.add_argument(
        "--starts", nargs="+", type=int, choices=(1, 2, 3, 4), default=(1, 2, 3, 4)
    )
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--image", default="skaha/astroml:latest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).strip()
    if dirty and not args.dry_run:
        raise RuntimeError("refusing dispatch from a dirty checkout")
    dispatch_path = args.run_root / "dispatch.json"
    if dispatch_path.exists() and not args.dry_run:
        raise RuntimeError(f"refusing duplicate dispatch: {dispatch_path}")

    record = {
        "schema_version": "kinuv-unified-map-dispatch-v1",
        "state": "DISPATCHING",
        "created_utc": now(),
        "code_commit": commit,
        "source_dirty": bool(dirty),
        "image": args.image,
        "allocation": {
            "class": "flexible",
            "cpu_requested": None,
            "memory_gb_requested": None,
            "platform_ceiling": {"cpu": 16, "memory_gb": 32},
        },
        "execution": {
            "topology": "one-flexible-headless-session-per-target-start",
            "targets": list(args.targets),
            "starts": list(args.starts),
            "maxiter": args.maxiter,
            "candidate_only": True,
        },
        "run_root": str(args.run_root.resolve()),
        "sessions": [],
    }
    atomic_json(dispatch_path, record)
    short = commit[:7]
    tag = re.sub(r"[^A-Za-z0-9]+", "-", args.run_root.name).strip("-")[-12:]
    for target in args.targets:
        for start in args.starts:
            name = f"kinuv-umap-{target[-3:]}-{short}-s{start}-{tag}"
            result = submit(
                name=name,
                image=args.image,
                command=[
                    "/bin/bash",
                    str(REPO / "experiments/unified_foundation/unified_map_headless.sh"),
                    target,
                    str(start),
                    str(args.run_root.resolve()),
                    commit,
                    str(args.maxiter),
                ],
                dry_run=args.dry_run,
            )
            record["sessions"].append(
                {
                    "role": "unified-map-start",
                    "target": target,
                    "start": start,
                    "name": name,
                    **result,
                }
            )
            atomic_json(dispatch_path, record)
            if not result["ok"]:
                record["state"] = "PARTIAL_SUBMIT_FAILURE"
                atomic_json(dispatch_path, record)
                return 1
    record["state"] = "DRY_RUN" if args.dry_run else "SUBMITTED"
    atomic_json(dispatch_path, record)
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
