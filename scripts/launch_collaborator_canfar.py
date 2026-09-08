#!/usr/bin/env python3
"""Submit the collaborator posterior as flexible CANFAR headless sessions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
CANFAR = Path.home() / ".local/bin/canfar"
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


def submit(*, name: str, image: str, command: list[str], dry_run: bool) -> dict:
    argv = [
        str(CANFAR_PYTHON), str(CANFAR), "create", "headless", image,
        "--name", name, "--", *command,
    ]
    if dry_run:
        return {"ok": True, "session_id": None, "argv": argv, "dry_run": True}
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    match = SESSION_RE.search(output)
    return {
        "ok": result.returncode == 0 and match is not None,
        "session_id": match.group(1) if match else None,
        "returncode": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "argv": argv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-root",
        type=Path,
        default=PROJECT / "results/incoming/collaborator-delivery-20260908/nuts-headless-attempt3",
    )
    parser.add_argument("--image", default="skaha/astroml:latest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    short = commit[:6]
    dispatch_path = args.attempt_root / "dispatch.json"
    if dispatch_path.exists() and not args.dry_run:
        raise RuntimeError(f"refusing duplicate dispatch: {dispatch_path}")
    record = {
        "schema_version": "kinuv-collaborator-canfar-dispatch-v1",
        "created_utc": now(),
        "code_commit": commit,
        "image": args.image,
        "allocation": {
            "class": "flexible",
            "cpu_requested": None,
            "memory_gb_requested": None,
            "platform_ceiling": {"cpu": 16, "memory_gb": 32},
        },
        "attempt_root": str(args.attempt_root.resolve()),
        "sessions": [],
    }
    write_json(dispatch_path, record)
    for target in ("KGAS066", "KGAS007"):
        name = f"kinuv-{target}-{short}-nuts-a3"
        result = submit(
            name=name,
            image=args.image,
            command=[
                "/bin/bash", str(REPO / "scripts/run_collaborator_nuts_headless.sh"),
                target, str(args.attempt_root.resolve()), commit,
            ],
            dry_run=args.dry_run,
        )
        record["sessions"].append({"role": "nuts", "target": target, "name": name, **result})
        write_json(dispatch_path, record)
        if not result["ok"]:
            record["state"] = "PARTIAL_SUBMIT_FAILURE"
            write_json(dispatch_path, record)
            return 1
    monitor_name = f"kinuv-collab-{short}-post-a3"
    monitor = submit(
        name=monitor_name,
        image=args.image,
        command=[
            "/bin/bash", str(REPO / "scripts/run_collaborator_postprocess_headless.sh"),
            str(args.attempt_root.resolve()), commit,
        ],
        dry_run=args.dry_run,
    )
    record["sessions"].append({"role": "postprocess", "target": None, "name": monitor_name, **monitor})
    record["state"] = "DRY_RUN" if args.dry_run else "SUBMITTED" if monitor["ok"] else "PARTIAL_SUBMIT_FAILURE"
    write_json(dispatch_path, record)
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if monitor["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
