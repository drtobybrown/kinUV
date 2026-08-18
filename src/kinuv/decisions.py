"""Load ``DEC-*`` ids from disk and bind ``@requires`` (DEC-066-INDEX)."""

from __future__ import annotations

import re
from functools import wraps
from pathlib import Path
from typing import Callable

_DECISIONS_DIR = Path(__file__).resolve().parents[2] / "docs" / "decisions"
_ID_RE = re.compile(r"^id:\s*(\S+)", re.M)
_STATUS_RE = re.compile(r"^status:\s*(\S+)", re.M)

_ALLOWED_STATUS = frozenset({"accepted", "proposed"})


def load_decision_index(directory: Path | None = None) -> dict[str, str]:
    """Map DEC id → status. Duplicate ids or missing frontmatter raise."""
    root = directory or _DECISIONS_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"decisions directory missing: {root}")
    index: dict[str, str] = {}
    for path in sorted(root.glob("DEC-*.md")):
        text = path.read_text(encoding="utf-8")
        id_m = _ID_RE.search(text)
        st_m = _STATUS_RE.search(text)
        if id_m is None or st_m is None:
            raise ValueError(f"{path.name}: missing id/status frontmatter")
        dec_id, status = id_m.group(1), st_m.group(1)
        if dec_id in index:
            raise ValueError(f"duplicate DEC id {dec_id}")
        index[dec_id] = status
    return index


def requires(*dec_ids: str) -> Callable:
    """Fail at call time if a named DEC is missing, superseded, or conflicted."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(*args, **kwargs):
            index = load_decision_index()
            for dec_id in dec_ids:
                if dec_id not in index:
                    raise LookupError(f"unknown decision {dec_id}")
                status = index[dec_id]
                if status not in _ALLOWED_STATUS:
                    raise LookupError(
                        f"{dec_id} has status {status!r}, not accepted/proposed"
                    )
            return fn(*args, **kwargs)

        wrapped._kinuv_requires = dec_ids  # type: ignore[attr-defined]
        return wrapped

    return decorator
