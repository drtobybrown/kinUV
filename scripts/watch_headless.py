#!/usr/bin/env python3
"""Copy CANFAR platform logs onto /arc until the session is gone (~1 h retention)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinuv.runner.canfar import (  # noqa: E402
    get_status,
    run_dir,
    stream_events,
    stream_logs,
    utc_now,
    write_json,
)
from kinuv.runner.log import append_log, logs_dir  # noqa: E402

DONE_STATES = frozenset(
    {"SUCCEEDED", "COMPLETED_UNMIXED", "CRASHED", "FAILED", "SIGNAL", "FAILED_SUBMIT"}
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _snapshot(path: Path, text: str) -> None:
    path.write_text(text)
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Persist canfar logs/info/events to the run dir")
    p.add_argument("--run-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--interval", type=float, default=20.0)
    p.add_argument("--max-hours", type=float, default=24.0)
    p.add_argument("--gone-polls", type=int, default=8)
    args = p.parse_args()

    dest = run_dir(args.run_id)
    dest.mkdir(parents=True, exist_ok=True)
    logd = logs_dir(args.run_id)
    platform = dest / "logs" / "platform.log"
    watcher_json = logd / "watcher.json"
    trigger = dest / ".trigger_complete"
    t0 = time.time()
    last = {"logs": "", "info": "", "events": ""}
    gone = 0
    polls = 0
    append_log(
        platform,
        f"{utc_now()} watcher start session={args.session_id} interval={args.interval}s",
    )

    while True:
        polls += 1
        elapsed_h = (time.time() - t0) / 3600.0
        rec = {
            "state": "WATCHING",
            "session_id": args.session_id,
            "polls": polls,
            "updated_at": utc_now(),
            "elapsed_h": elapsed_h,
        }
        if trigger.is_file():
            rec["state"] = "TRIGGER"
            write_json(watcher_json, rec)
            append_log(platform, f"{utc_now()} stop: .trigger_complete")
            return 0
        status_path = dest / "status.json"
        if status_path.is_file():
            try:
                worker_state = json.loads(status_path.read_text()).get("state")
            except (OSError, json.JSONDecodeError):
                worker_state = None
            if worker_state in DONE_STATES:
                rec["state"] = f"WORKER_{worker_state}"
                write_json(watcher_json, rec)
                append_log(platform, f"{utc_now()} stop: worker state {worker_state}")
                return 0

        info = get_status(args.session_id)
        raw_info = str(info.get("raw") or "")
        logs = stream_logs(args.session_id)
        events = stream_events(args.session_id)
        missing = (not info.get("ok")) or ("404" in raw_info) or ("not found" in raw_info.lower())
        rec["canfar_state"] = info.get("state")
        rec["canfar_ok"] = bool(info.get("ok"))
        rec["gone_polls"] = gone if missing else 0

        for label, text, filename in (
            ("info", raw_info, "canfar-info.txt"),
            ("logs", logs, "canfar-logs.txt"),
            ("events", events, "canfar-events.txt"),
        ):
            digest = _digest(text)
            if digest != last[label]:
                _snapshot(logd / filename, text)
                append_log(
                    platform,
                    f"{utc_now()} {label} changed sha={digest} bytes={len(text.encode('utf-8'))}",
                )
                if text.strip():
                    append_log(platform, text.rstrip() + "\n")
                last[label] = digest

        if missing:
            gone += 1
            rec["state"] = "SESSION_MISSING"
            append_log(
                platform,
                f"{utc_now()} session missing ({gone}/{args.gone_polls}) info={raw_info[:200]!r}",
            )
            if gone >= int(args.gone_polls):
                rec["state"] = "GONE"
                write_json(watcher_json, rec)
                append_log(platform, f"{utc_now()} stop: session gone")
                return 0
        else:
            gone = 0

        write_json(watcher_json, rec)
        if elapsed_h >= float(args.max_hours):
            append_log(platform, f"{utc_now()} stop: max-hours {args.max_hours}")
            return 0
        time.sleep(float(args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
