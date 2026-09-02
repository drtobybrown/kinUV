"""Patch ``docs/architecture/STATUS.md`` Agent Run Status from a finished job.

Does not rewrite Architecture mailbox history. YAML ``pending`` may be cleared.
Git commit is left to the submit-host / parent; the worker only writes the file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from kinuv.runner.canfar import REPO, utc_now

STATUS_REL = Path("docs/architecture/STATUS.md")
_BULLET_RE = re.compile(r"^(\* \*\*)(.+?)(:\*\*\s*)(.*)$")
_KEYS = (
    "Phase",
    "Last Action",
    "Decisions Made",
    "Blockers / Gates",
    "Next Step",
)


def status_md_path() -> Path:
    root = Path(os.environ.get("KINUV_REPO", str(REPO)))
    return root / STATUS_REL


def _set_pending(front: str, pending: list[str] | None) -> str:
    if pending is None:
        return front
    if not pending:
        repl = "pending: []"
    else:
        inner = ", ".join(json.dumps(x) for x in pending)
        repl = f"pending: [{inner}]"
    new, n = re.subn(r"^pending:\s*.*$", repl, front, count=1, flags=re.M)
    return new if n else front


def patch_agent_run_status(
    path: Path,
    bullets: dict[str, str],
    *,
    pending: list[str] | None = None,
) -> None:
    """Replace Agent Run Status bullets. Stop at ``# Architecture mailbox``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    start = text.find("## Agent Run Status")
    end = text.find("# Architecture mailbox")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("STATUS.md missing Agent Run Status or Architecture mailbox")
    head = _set_pending(text[:start], pending)
    mid = text[start:end]
    tail = text[end:]
    out: list[str] = []
    seen: set[str] = set()
    for line in mid.splitlines(True):
        m = _BULLET_RE.match(line.rstrip("\n"))
        if m and m.group(2) in bullets:
            key = m.group(2)
            out.append(f"* **{key}:** {bullets[key]}\n")
            seen.add(key)
        else:
            out.append(line)
    missing = [k for k in _KEYS if k in bullets and k not in seen]
    if missing:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.startswith("* **"):
                insert_at = i
                break
        extra = [f"* **{k}:** {bullets[k]}\n" for k in missing]
        out = out[:insert_at] + extra + out[insert_at:]
    mid_s = "".join(out)
    if not mid_s.endswith("\n"):
        mid_s += "\n"
    if not tail.startswith("\n") and not mid_s.endswith("\n\n"):
        mid_s += "\n"
    path.write_text(head + mid_s + tail, encoding="utf-8")


def write_job_status_md(
    *,
    run_id: str,
    session_id: str | None,
    state: str,
    mixing_pass: bool,
    sampler: str,
    elapsed_s: float,
    note: str = "",
    kind: str | None = None,
) -> Path | None:
    """Best-effort patch of the repo mailbox from a headless worker."""
    path = status_md_path()
    if not path.is_file():
        return None
    sid = session_id or "unknown"
    mix = "pass" if mixing_pass else "FAIL"
    approaching = "pa25" in str(run_id).lower() or "pa25" in str(kind or "").lower()
    if approaching:
        phase = f"066 NUTS PA 25.2 {state} (`{run_id}`)"
        next_step = (
            "Copy posteriors into docs/reviews/artifacts/"
            "2026-09-02-kgas066-leftover-and-modes/pa25/. "
            "Official MAP unchanged. Do not start G4"
        )
        default_note = (
            "Approaching-PA job wrote run-dir status.json + Agent Run Status. "
            "Official MAP unchanged. Do not start G4"
        )
    else:
        phase = f"G3 066 NUTS {state} (`{run_id}`)"
        next_step = (
            "Copy posteriors into docs/reviews/artifacts/2026-08-30-g3-nuts/, "
            "6D corner. Official MAP unchanged. Do not start G4"
        )
        default_note = (
            "Job wrote run-dir status.json + Agent Run Status. "
            "Official MAP unchanged. Do not start G4"
        )
    bullets = {
        "Phase": phase,
        "Last Action": (
            f"Session `{sid}` {state} mixing={mix} sampler={sampler} "
            f"elapsed_s={elapsed_s:.0f} utc={utc_now()}"
        ),
        "Decisions Made": note or default_note,
        "Blockers / Gates": (
            "none"
            if mixing_pass
            else "mixing failed; do not treat 066 JSON as calibrated NUTS"
        ),
        "Next Step": next_step,
    }
    pending: list[str] | None = [] if state in {"SUCCEEDED", "COMPLETED_UNMIXED"} else None
    patch_agent_run_status(path, bullets, pending=pending)
    return path


def ping_status_ntfy() -> None:
    script = Path(os.environ.get("KINUV_REPO", str(REPO))) / ".cursor" / "notify.sh"
    if not script.is_file():
        return
    try:
        subprocess.run(
            ["bash", str(script)],
            check=False,
            timeout=15,
            capture_output=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
