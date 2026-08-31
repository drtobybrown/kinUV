"""CANFAR CLI wrappers for headless NUTS (DEC-067-RUNNER, DEC-OPS-AUTH).

The live ``canfar`` CLI has no ``ps --json``. Parse ``create`` / ``info`` text.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from kinuv.decisions import requires

CANFAR_BIN = os.environ.get("CANFAR_BIN", "canfar")
PROJECT_ROOT = Path(
    os.environ.get(
        "KINUV_PROJECT",
        "/arc/projects/KILOGAS/analysis/toby_sandbox",
    )
)
# Durable products live on the project volume, never $HOME. /scratch is ephemeral.
RUNS_ROOT = Path(os.environ.get("KINUV_RUNS", str(PROJECT_ROOT / "kinuv_runs")))
DEFAULT_IMAGE = "skaha/astroml:latest"
FALLBACK_IMAGE = "skaha/base-notebook:latest"
REPO = PROJECT_ROOT / "kinUV"
CERT_PATH = Path.home() / ".ssl" / "cadcproxy.pem"

_SESSION_ID_RE = re.compile(r"\(ID:\s*([A-Za-z0-9]+)\)")
_STATUS_RE = re.compile(r"Status\s+(\S+)")
_NAME_RE = re.compile(r"Name\s+(\S+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_dir(run_id: str) -> Path:
    return RUNS_ROOT / str(run_id)


def write_json(path: Path, rec: dict) -> None:
    """Atomic JSON write with fsync so a SIGKILL still leaves the last heartbeat."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(rec, indent=2) + "\n").encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(str(tmp), str(path))
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def fsync_path(path: Path) -> None:
    """Best-effort file+dir fsync so an OOM still leaves the last checkpoint on /arc."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def archive_run(run_id: str) -> Path | None:
    """Rename ``<run_id>/`` so a relaunch does not overwrite the previous attempt."""
    src = run_dir(run_id)
    if not src.exists():
        return None
    dest = RUNS_ROOT / f"{run_id}.archive.{utc_now().replace(':', '')}"
    src.rename(dest)
    return dest


def parse_session_id(create_text: str) -> str | None:
    """Parse ``Successfully created session 'name' (ID: abc123)``."""
    m = _SESSION_ID_RE.search(create_text)
    return m.group(1) if m else None


def parse_info_status(info_text: str) -> dict:
    status_m = _STATUS_RE.search(info_text)
    name_m = _NAME_RE.search(info_text)
    return {
        "state": status_m.group(1) if status_m else "unknown",
        "name": name_m.group(1) if name_m else None,
        "raw": info_text,
    }


def _run(argv: list[str], *, timeout: float | None = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def cert_days_left() -> float | None:
    if not CERT_PATH.is_file():
        return None
    proc = _run(
        ["openssl", "x509", "-enddate", "-noout", "-in", str(CERT_PATH)],
        timeout=15.0,
    )
    if proc.returncode != 0:
        return None
    line = " ".join((proc.stdout or "").strip().split())
    if "=" not in line:
        return None
    from datetime import datetime as dt

    end = dt.strptime(line.split("=", 1)[1], "%b %d %H:%M:%S %Y %Z")
    end = end.replace(tzinfo=timezone.utc)
    return (end - datetime.now(timezone.utc)).total_seconds() / 86400.0


def ensure_cert() -> dict:
    """Refresh CADC cert when missing or < 2 days left (DEC-OPS-AUTH)."""
    left = cert_days_left()
    if left is not None and left > 2.0:
        return {"ok": True, "refreshed": False, "days_left": left}
    cadc = shutil.which("cadc-get-cert")
    if cadc is None:
        return {"ok": left is not None, "refreshed": False, "days_left": left}
    proc = _run([cadc, "--days-valid", "10"], timeout=20.0)
    return {
        "ok": proc.returncode == 0 or (left is not None and left > 0),
        "refreshed": proc.returncode == 0,
        "days_left": cert_days_left(),
        "stderr": (proc.stderr or "")[:500],
    }


@requires("DEC-067-RUNNER", "DEC-OPS-AUTH")
def submit_headless(
    *,
    name: str,
    command: list[str],
    image: str = DEFAULT_IMAGE,
    gpu: int | None = None,
    cpu: int | None = None,
    memory: int | None = None,
    env: dict[str, str] | None = None,
) -> dict:
    """``canfar create headless``. Flexible CPU/RAM unless ``cpu``/``memory`` set.

    GPU only if ``gpu`` is set. Pin RAM after a session vanishes under flexible.
    """
    argv = [CANFAR_BIN, "create", "headless", image, "--name", name]
    if cpu:
        argv.extend(["--cpu", str(int(cpu))])
    if memory:
        argv.extend(["--memory", str(int(memory))])
    if gpu:
        argv.extend(["--gpu", str(int(gpu))])
    if env:
        for k, v in env.items():
            argv.extend(["--env", f"{k}={v}"])
    argv.append("--")
    argv.extend(command)
    proc = _run(argv, timeout=180.0)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sid = parse_session_id(text)
    return {
        "ok": proc.returncode == 0 and bool(sid),
        "session_id": sid,
        "returncode": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
        "image": image,
        "name": name,
        "gpu": gpu,
        "cpu": cpu,
        "memory": memory,
        "argv": argv,
    }


@requires("DEC-067-RUNNER")
def get_status(session_id: str) -> dict:
    proc = _run([CANFAR_BIN, "info", str(session_id)], timeout=60.0)
    parsed = parse_info_status((proc.stdout or "") + "\n" + (proc.stderr or ""))
    parsed["ok"] = proc.returncode == 0
    parsed["session_id"] = session_id
    return parsed


@requires("DEC-067-RUNNER")
def stream_logs(session_id: str, dest: Path | None = None) -> str:
    proc = _run([CANFAR_BIN, "logs", str(session_id)], timeout=60.0)
    text = proc.stdout or ""
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
    return text


@requires("DEC-067-RUNNER")
def stream_events(session_id: str) -> str:
    proc = _run([CANFAR_BIN, "events", str(session_id)], timeout=60.0)
    return proc.stdout or ""


@requires("DEC-067-RUNNER")
def cleanup(session_id: str) -> dict:
    proc = _run([CANFAR_BIN, "delete", str(session_id), "--force"], timeout=60.0)
    return {
        "ok": proc.returncode == 0,
        "session_id": session_id,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


def write_manifest(run_id: str, rec: dict) -> Path:
    path = run_dir(run_id) / "manifest.json"
    write_json(path, rec)
    return path


def write_status(run_id: str, rec: dict) -> Path:
    rec = dict(rec)
    rec.setdefault("updated_at", utc_now())
    path = run_dir(run_id) / "status.json"
    write_json(path, rec)
    return path
