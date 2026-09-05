#!/usr/bin/env python3
"""Write the 066 S3 comparator table (kinUV vis vs KinMS cube). No KinMS import."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
LIVE = DEST / "live_fitters"
FITTERS = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    venv_py = FITTERS / "venv" / "bin" / "python"
    runner = REPO / "external" / "run_image_benchmarks.py"
    if venv_py.is_file() and runner.is_file():
        subprocess.run([str(venv_py), str(runner)], check=False, cwd=str(REPO))
    elif runner.is_file():
        subprocess.run([sys.executable, str(runner)], check=False, cwd=str(REPO))
    kin = {}
    if (DEST / "kinms.json").is_file():
        kin = json.loads((DEST / "kinms.json").read_text())
    elif (LIVE / "kinms.json").is_file():
        kin = json.loads((LIVE / "kinms.json").read_text())
    table_path = DEST / "s3_table.json"
    if table_path.is_file():
        table = json.loads(table_path.read_text())
    else:
        table = {}
    table.setdefault("header_note", "vis chi2 is the fit; quote_inner_slope: false")
    table["kinms"] = kin
    table["quote_inner_slope"] = False
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps({"dest": str(DEST), "kinms": kin.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
