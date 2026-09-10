#!/usr/bin/env python3
"""Snapshot and submit one flexible Phase-4 benchmark session per target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[2]
CANFAR = Path.home() / ".local/bin/canfar"
SESSION_RE = re.compile(r"\(ID:\s*([A-Za-z0-9]+)\)")
SOURCE_NAMES = ("phase4_synthetic_benchmark.py", "phase4_synthetic_headless.sh")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload):
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


def _submit(name, image, command, dry_run):
    argv = ["/usr/bin/python3", str(CANFAR), "create", "headless", image, "--name", name, "--", *command]
    if dry_run:
        return {"ok": True, "session_id": None, "argv": argv, "dry_run": True}
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kinms-archive", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=("KGAS066", "KGAS007"), default=("KGAS066", "KGAS007"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(7401,))
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--image", default="skaha/astroml:latest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.kinms_archive = args.kinms_archive.resolve()
    if not args.kinms_archive.is_file() and not args.dry_run:
        raise FileNotFoundError(args.kinms_archive)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    production_dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "external"], cwd=REPO, text=True
    ).strip()
    if production_dirty:
        raise RuntimeError("refusing dispatch with dirty production src/external files")
    if (args.run_root / "dispatch.json").exists():
        raise FileExistsError(args.run_root / "dispatch.json")
    source_dir = args.run_root / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    source_records = {}
    for name in SOURCE_NAMES:
        source = Path(__file__).parent / name
        destination = source_dir / name
        shutil.copy2(source, destination)
        source_records[name] = {"sha256": _sha256(destination), "bytes": destination.stat().st_size}
    record = {
        "schema_version": "kinuv-unified-phase4-dispatch-v1",
        "state": "DISPATCHING",
        "created_utc": _now(),
        "production_code_commit": commit,
        "production_source_dirty": False,
        "experiment_source": source_records,
        "kinms_environment_archive": {
            "path": str(args.kinms_archive),
            "sha256": _sha256(args.kinms_archive) if args.kinms_archive.is_file() else None,
        },
        "allocation": {
            "class": "flexible",
            "cpu_requested": None,
            "memory_gb_requested": None,
            "platform_ceiling": {"cpu": 16, "memory_gb": 32},
        },
        "targets": list(args.targets),
        "seeds": list(args.seeds),
        "maxiter": args.maxiter,
        "sessions": [],
    }
    dispatch_path = args.run_root / "dispatch.json"
    _write_json(dispatch_path, record)
    for target in args.targets:
        name = f"kinuv-p4-{target[-3:]}-{commit[:7]}"
        result = _submit(
            name,
            args.image,
            [
                "/bin/bash",
                str(source_dir / "phase4_synthetic_headless.sh"),
                target,
                str(args.run_root),
                commit,
                str(args.kinms_archive),
                str(args.maxiter),
                *[str(seed) for seed in args.seeds],
            ],
            args.dry_run,
        )
        record["sessions"].append({"target_id": target, "name": name, **result})
        _write_json(dispatch_path, record)
        if not result["ok"]:
            record["state"] = "PARTIAL_SUBMIT_FAILURE"
            _write_json(dispatch_path, record)
            return 1
    record["state"] = "DRY_RUN" if args.dry_run else "SUBMITTED"
    _write_json(dispatch_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
