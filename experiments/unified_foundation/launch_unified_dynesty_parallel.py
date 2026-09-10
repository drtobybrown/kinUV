#!/usr/bin/env python3
"""Submit fresh shared-JIT thread-pool Dynesty replicates to flexible CANFAR."""

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
CANFAR = Path.home() / ".local/bin/canfar"
SESSION_RE = re.compile(r"\(ID:\s*([A-Za-z0-9]+)\)")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def submit(name, command, image, cpu, memory, dry_run):
    argv = ["/usr/bin/python3", str(CANFAR), "create", "headless", image, "--name", name]
    if cpu is not None:
        argv.extend(("--cpu", str(cpu)))
    if memory is not None:
        argv.extend(("--memory", str(memory)))
    argv.extend(("--", *command))
    if dry_run:
        return {"ok": True, "session_id": None, "argv": argv, "dry_run": True}
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        timeout=240,
        env={**os.environ, "TERM": "dumb", "NO_COLOR": "1"},
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--map-commit", required=True)
    parser.add_argument("--targets", nargs="+", choices=("KGAS066", "KGAS007"), default=("KGAS066", "KGAS007"))
    parser.add_argument("--replicates", type=int, default=2, choices=(2,))
    parser.add_argument("--workers", type=int, default=4, choices=(4, 8, 16, 32))
    parser.add_argument("--n-effective", type=int, default=2000)
    parser.add_argument("--slices", type=int, default=17)
    parser.add_argument("--dlogz-init", type=float, default=0.01)
    parser.add_argument("--resume-root", type=Path)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--memory", type=int)
    parser.add_argument("--image", default="skaha/astroml:latest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if dirty and not args.dry_run:
        raise RuntimeError("refusing to dispatch from a dirty checkout")
    if args.run_root.exists() and any(args.run_root.iterdir()):
        raise FileExistsError("parallel race requires a fresh immutable run root")

    dispatch_path = args.run_root / "dispatch.json"
    record = {
        "schema_version": "kinuv-unified-dynesty-parallel-dispatch-v1",
        "state": "DISPATCHING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sampler": "dynesty-rslice",
        "code_commit": commit,
        "map_commit": args.map_commit,
        "run_root": str(args.run_root.resolve()),
        "allocation": {
            "class": "fixed" if args.cpu is not None or args.memory is not None else "flexible",
            "cpu_requested": args.cpu,
            "memory_requested": args.memory,
        },
        "parallel": {
            "executor": "ThreadPoolExecutor",
            "workers": args.workers,
            "queue_size": args.workers,
            "shared_prewarmed_jit": True,
        },
        "n_effective": args.n_effective,
        "slices": args.slices,
        "dlogz_init": args.dlogz_init,
        "resume_root": str(args.resume_root.resolve()) if args.resume_root else None,
        "sessions": [],
    }
    atomic_json(dispatch_path, record)
    script = REPO / "experiments/unified_foundation/unified_dynesty_parallel_headless.sh"
    for target in args.targets:
        map_result = args.map_root / target / "start-4" / "result.json"
        if not map_result.is_file() and not args.dry_run:
            raise FileNotFoundError(map_result)
        for replicate in range(1, args.replicates + 1):
            seed = 960000 + (66 if target == "KGAS066" else 7) * 10 + replicate
            command = [
                "/bin/bash",
                str(script),
                target,
                str(replicate),
                str(seed),
                str(args.run_root.resolve()),
                commit,
                str(map_result.resolve()),
                args.map_commit,
                str(args.workers),
                str(args.n_effective),
                str(args.slices),
                str(args.dlogz_init),
                str(args.resume_root.resolve()) if args.resume_root else "",
            ]
            name = f"kinuv-udp-{target[-3:]}-{commit[:7]}-r{replicate}"
            response = submit(name, command, args.image, args.cpu, args.memory, args.dry_run)
            record["sessions"].append(
                {"target": target, "replicate": replicate, "seed": seed, "name": name, **response}
            )
            atomic_json(dispatch_path, record)
            if not response["ok"]:
                record["state"] = "PARTIAL_SUBMIT_FAILURE"
                atomic_json(dispatch_path, record)
                return 1
    record["state"] = "DRY_RUN" if args.dry_run else "SUBMITTED"
    atomic_json(dispatch_path, record)
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
