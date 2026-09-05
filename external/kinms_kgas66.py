#!/usr/bin/env python3
"""Standalone KinMS comparator for KGAS066. Not a kinUV likelihood.

Vis chi2 = s * sum w |d-m|^2 is the fit. Do not import this module from
src/kinuv or scripts. Do not pip-install into kinuv-venv-recovery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEST = (
    Path(__file__).resolve().parents[1]
    / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
)


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    rec = {
        "tool": "KinMS",
        "ran": False,
        "cube_dir": str(CUBE),
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "leftover_gate": "SB-dominated",
        "note": (
            "Comparator only. vis chi2 is the kinUV fit. "
            "This file is the only licensed KinMS import site."
        ),
    }
    try:
        import kinms  # noqa: F401
    except Exception as exc:
        rec["status"] = "missing"
        rec["error"] = type(exc).__name__
        (DEST / "kinms.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        return 0
    rec["status"] = "imported_no_fit"
    rec["kinms_module"] = getattr(sys.modules.get("kinms"), "__file__", None)
    rec["note"] += (
        " Interpreter already had KinMS; no cube posterior was run this card "
        "(no isolated env; recovery venv must not be pip-mutated)."
    )
    (DEST / "kinms.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
