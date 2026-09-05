#!/usr/bin/env python3
"""3DBarolo CLI comparator for 066 S3. Vis chi2 is the fit. No KinMS import."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
)
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"


def _which() -> str | None:
    for name in ("BBarolo", "3dbarolo", "bbarolo"):
        found = shutil.which(name)
        if found:
            return found
    return None


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    exe = _which()
    cubes = sorted(CUBE.glob("*.fits")) if CUBE.is_dir() else []
    rec = {
        "tool": "3DBarolo",
        "ran": False,
        "executable": exe,
        "cube_dir": str(CUBE),
        "cube_fits": [str(p) for p in cubes],
        "note": (
            "Comparator only. vis chi2 = s * sum w |d-m|^2 is the kinUV fit. "
            "3DBarolo was not on PATH for S1 (2026-08-29)."
        ),
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "leftover_gate": "SB-dominated",
    }
    if exe is None:
        rec["status"] = "missing_on_path"
        (DEST / "barolo.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        return 0
    # Do not invent a parameter file or overwrite science products. Record PATH only
    # unless a user-supplied BBarolo param file exists next to the cube.
    param = CUBE / "barolo.param"
    if not param.is_file():
        rec["status"] = "on_path_no_paramfile"
        rec["ran"] = False
        (DEST / "barolo.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps(rec, indent=2))
        return 0
    work = DEST / "barolo-run"
    work.mkdir(exist_ok=True)
    proc = subprocess.run(
        [exe, "-p", str(param)],
        cwd=str(work),
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    rec["ran"] = proc.returncode == 0
    rec["returncode"] = proc.returncode
    rec["status"] = "ran" if rec["ran"] else "failed"
    (work / "stdout.txt").write_text(proc.stdout or "")
    (work / "stderr.txt").write_text(proc.stderr or "")
    (DEST / "barolo.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: rec[k] for k in rec if k != "cube_fits"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
