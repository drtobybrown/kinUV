#!/usr/bin/env python3
"""Submit a CANFAR headless job. Do not wait for NUTS to finish."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinuv.runner.canfar import (  # noqa: E402
    DEFAULT_IMAGE,
    FALLBACK_IMAGE,
    REPO as KINUV_REPO,
    ensure_cert,
    headless_job_env,
    make_run_id,
    pa_init_deg_for_kind,
    point_latest,
    run_dir,
    steal_latest,
    submit_headless,
    utc_now,
    write_manifest,
    write_status,
)
from kinuv.runner.log import logs_dir  # noqa: E402


def git_sha6() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short=6", "HEAD"],
        cwd=KINUV_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    sha = (proc.stdout or "unknown").strip()
    return sha[:6]


def session_name(kind: str = "nuts") -> str:
    return f"kinuv-KGAS066-{git_sha6()}-{kind}"[:63]


def start_watcher(run_id: str, session_id: str) -> int | None:
    """Copy platform logs onto /arc until the session vanishes."""
    logd = logs_dir(run_id)
    out = logd / "watcher.out"
    cmd = [
        sys.executable,
        str(KINUV_REPO / "scripts/watch_headless.py"),
        "--run-id",
        run_id,
        "--session-id",
        session_id,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(KINUV_REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with out.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(KINUV_REPO),
            env=env,
        )
    return proc.pid


def main() -> int:
    p = argparse.ArgumentParser(description="Launch kinUV headless (DEC-067-RUNNER)")
    p.add_argument(
        "--run-id",
        default=None,
        help="default: {KGASID}-{YYYYMMDDTHHMMSSZ}-nuts",
    )
    p.add_argument("--galaxy", default="KGAS066")
    p.add_argument("--kind", default="nuts")
    p.add_argument(
        "--pa-init",
        type=float,
        default=None,
        help="Physical PA start (deg). Default 25.2 for kind nuts-pa25, else MAP 199.73. "
        "Delivery is KINUV_PA_INIT on --env, not the manifest.",
    )
    p.add_argument("--gpu", type=int, default=0, help="GPUs; 0 = omit (CPU jax venv)")
    p.add_argument("--cpu", type=int, default=0, help="CPU cores; 0 = flexible")
    p.add_argument("--memory", type=int, default=0, help="RAM GB; 0 = flexible")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--skip-pull", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-watch", action="store_true")
    args = p.parse_args()

    gpu = int(args.gpu) if int(args.gpu) > 0 else None
    cpu = int(args.cpu) if int(args.cpu) > 0 else None
    memory = int(args.memory) if int(args.memory) > 0 else None
    run_id = args.run_id or make_run_id(args.galaxy, args.kind)
    name = session_name(args.kind)
    entry = str(KINUV_REPO / "scripts/canfar_entrypoint.sh")
    cert = ensure_cert()
    pa_init = (
        float(args.pa_init)
        if args.pa_init is not None
        else pa_init_deg_for_kind(args.kind)
    )
    env = headless_job_env(
        run_id=run_id,
        galaxy=args.galaxy,
        kind=args.kind,
        gpu=gpu,
        skip_pull=args.skip_pull,
        repo=KINUV_REPO,
        runs_root=run_dir(run_id).parent,
        pa_init=pa_init,
    )

    manifest = {
        "run_id": run_id,
        "galaxy": args.galaxy,
        "kind": args.kind,
        "session_name": name,
        "git_commit": git_sha6(),
        "image": args.image,
        "gpu": gpu,
        "cpu": cpu,
        "memory_gb": memory,
        "flexible": cpu is None and memory is None,
        "cert": {k: cert[k] for k in cert if k != "stderr"},
        "created_at": utc_now(),
        "command": ["/bin/bash", entry, run_id],
        "warmup": 200,
        "num_samples": 600,
        "num_chains": 4,
        "pa_init_deg": pa_init,
        "point_latest": steal_latest(args.kind),
        "artifact_dir": env["KINUV_ARTIFACT_DIR"],
    }
    write_manifest(run_id, manifest)
    write_status(
        run_id,
        {"state": "SUBMITTING", "step": "0/4", "session_id": None},
    )
    point_latest_path = None
    if steal_latest(args.kind):
        point_latest_path = point_latest(args.galaxy, run_id)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    result = submit_headless(
        name=name,
        command=["/bin/bash", entry, run_id],
        image=args.image,
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        env=env,
    )
    if (not result["ok"]) and args.image == DEFAULT_IMAGE:
        result = submit_headless(
            name=name,
            command=["/bin/bash", entry, run_id],
            image=FALLBACK_IMAGE,
            gpu=gpu,
            cpu=cpu,
            memory=memory,
            env=env,
        )
        manifest["image"] = FALLBACK_IMAGE
        manifest["image_fallback"] = True
    manifest["session_id"] = result.get("session_id")
    manifest["submit_ok"] = bool(result.get("ok"))
    manifest["submit_stdout"] = (result.get("stdout") or "")[-2000:]
    manifest["submit_stderr"] = (result.get("stderr") or "")[-2000:]
    write_manifest(run_id, manifest)
    write_status(
        run_id,
        {
            "state": "PENDING" if result.get("ok") else "FAILED_SUBMIT",
            "step": "0/4",
            "session_id": result.get("session_id"),
        },
    )
    dest = run_dir(run_id)
    logs_dir(run_id)
    (dest / "stream.log").write_text(manifest.get("submit_stdout") or "")
    watcher_pid = None
    if result.get("ok") and result.get("session_id") and not args.no_watch:
        watcher_pid = start_watcher(run_id, result["session_id"])
        manifest["watcher_pid"] = watcher_pid
        write_manifest(run_id, manifest)
    print(
        json.dumps(
            {
                "ok": result.get("ok"),
                "session_id": result.get("session_id"),
                "run_dir": str(dest),
                "latest": None if point_latest_path is None else str(point_latest_path),
                "kind": args.kind,
                "pa_init_deg": pa_init,
                "name": name,
                "image": manifest["image"],
                "gpu": gpu,
                "cpu": cpu,
                "memory_gb": memory,
                "flexible": cpu is None and memory is None,
                "watcher_pid": watcher_pid,
            },
            indent=2,
        )
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
