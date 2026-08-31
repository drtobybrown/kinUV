"""Job-owned logs on ``/arc``. Platform ``canfar logs`` expire in ~1 hour."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from kinuv.runner.canfar import RUNS_ROOT, run_dir, utc_now, write_status


def logs_dir(run_id: str) -> Path:
    path = run_dir(run_id) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_log_path(run_id: str) -> Path:
    return run_dir(run_id) / "worker.log"


def run_log_path(run_id: str) -> Path:
    return logs_dir(run_id) / "run.log"


def append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = line if line.endswith("\n") else line + "\n"
    with path.open("a", encoding="utf-8", buffering=1) as fh:
        fh.write(text)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_snapshot() -> dict:
    return {
        "utc": utc_now(),
        "pid": os.getpid(),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "session_id": os.environ.get("SKAHA_SESSION_ID"),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
        "rss_mb": rss_mb(),
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "python": sys.executable,
        "cwd": os.getcwd(),
    }


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        try:
            os.fsync(self.stream.fileno())
        except OSError:
            pass


def setup_worker_logging(run_id: str) -> logging.Logger:
    """Structured logger → ``logs/run.log``. Stdout is teed separately to ``worker.log``."""
    path = run_log_path(run_id)
    log = logging.getLogger(f"kinuv.run.{run_id}")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = time.gmtime
    fh = _FlushFileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def install_crash_hook(run_id: str, log: logging.Logger, on_fail=None) -> None:
    prev = sys.excepthook

    def _flush() -> None:
        if on_fail is None:
            return
        try:
            on_fail()
        except Exception:
            log.exception("checkpoint flush failed")

    def _hook(exc_type, exc, tb) -> None:
        _flush()
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        crash = run_dir(run_id) / "crash.log"
        append_log(crash, text)
        snap = host_snapshot()
        write_status(
            run_id,
            {
                "state": "CRASHED",
                "error": f"{exc_type.__name__}: {exc}"[:800],
                **{k: snap[k] for k in ("pid", "hostname", "session_id", "rss_mb")},
            },
        )
        log.error("unhandled exception\n%s", text)
        prev(exc_type, exc, tb)

    sys.excepthook = _hook

    def _on_signal(signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        log.warning("signal %s", name)
        _flush()
        write_status(
            run_id,
            {
                "state": "SIGNAL",
                "signal": name,
                "rss_mb": rss_mb(),
            },
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def runs_root() -> Path:
    return RUNS_ROOT
