#!/usr/bin/env python3
"""Submit a CANFAR headless job. Do not wait for NUTS to finish."""

from __future__ import annotations

import argparse
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
    run_dir,
    submit_headless,
    utc_now,
    write_manifest,
    write_status,
)


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


def main() -> int:
    p = argparse.ArgumentParser(description="Launch kinUV headless (DEC-067-RUNNER)")
    p.add_argument("--run-id", default="kgas066-nuts")
    p.add_argument("--galaxy", default="KGAS066")
    p.add_argument("--gpu", type=int, default=0, help="GPUs; 0 = omit (CPU jax venv)")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--skip-pull", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    gpu = int(args.gpu) if int(args.gpu) > 0 else None
    name = session_name("nuts")
    entry = str(KINUV_REPO / "scripts/canfar_entrypoint.sh")
    cert = ensure_cert()
    env = {
        "KINUV_RUN_ID": args.run_id,
        "KINUV_GALAXY": args.galaxy,
        "JAX_PLATFORMS": "cpu" if gpu is None else "cuda",
        "JAX_ENABLE_X64": "1",
    }
    if args.skip_pull:
        env["KINUV_SKIP_PULL"] = "1"

    manifest = {
        "run_id": args.run_id,
        "galaxy": args.galaxy,
        "session_name": name,
        "git_commit": git_sha6(),
        "image": args.image,
        "gpu": gpu,
        "flexible": True,
        "cert": {k: cert[k] for k in cert if k != "stderr"},
        "created_at": utc_now(),
        "command": ["/bin/bash", entry, args.run_id],
        "warmup": 200,
        "num_samples": 600,
        "num_chains": 4,
        "pa_init_deg": 199.72980072503037,
    }
    write_manifest(args.run_id, manifest)
    write_status(
        args.run_id,
        {"state": "SUBMITTING", "step": "0/4", "session_id": None},
    )
    if args.dry_run:
        print(json_dump := __import__("json").dumps(manifest, indent=2))
        return 0

    result = submit_headless(
        name=name,
        command=["/bin/bash", entry, args.run_id],
        image=args.image,
        gpu=gpu,
        env=env,
    )
    if (not result["ok"]) and args.image == DEFAULT_IMAGE:
        result = submit_headless(
            name=name,
            command=["/bin/bash", entry, args.run_id],
            image=FALLBACK_IMAGE,
            gpu=gpu,
            env=env,
        )
        manifest["image"] = FALLBACK_IMAGE
        manifest["image_fallback"] = True
    manifest["session_id"] = result.get("session_id")
    manifest["submit_ok"] = bool(result.get("ok"))
    manifest["submit_stdout"] = (result.get("stdout") or "")[-2000:]
    manifest["submit_stderr"] = (result.get("stderr") or "")[-2000:]
    write_manifest(args.run_id, manifest)
    write_status(
        args.run_id,
        {
            "state": "PENDING" if result.get("ok") else "FAILED_SUBMIT",
            "step": "0/4",
            "session_id": result.get("session_id"),
        },
    )
    dest = run_dir(args.run_id)
    (dest / "stream.log").write_text(manifest.get("submit_stdout") or "")
    print(
        __import__("json").dumps(
            {
                "ok": result.get("ok"),
                "session_id": result.get("session_id"),
                "run_dir": str(dest),
                "name": name,
                "image": manifest["image"],
                "gpu": gpu,
            },
            indent=2,
        )
    )
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
