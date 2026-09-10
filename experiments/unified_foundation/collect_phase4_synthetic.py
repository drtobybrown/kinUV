#!/usr/bin/env python3
"""Collect completed per-target Phase-4 benchmark summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    targets = []
    missing = []
    for target_id in ("KGAS066", "KGAS007"):
        path = root / "targets" / target_id / "summary.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            targets.append(
                {
                    "target_id": target_id,
                    "aggregate": row["aggregate"],
                    "gates": row["gates"],
                    "gate_pass": row["gate_pass"],
                    "summary_path": str(path),
                }
            )
        else:
            missing.append(target_id)
    complete = not missing
    payload = {
        "schema_version": "kinuv-unified-phase4-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "COMPLETE" if complete else "INCOMPLETE",
        "targets": targets,
        "missing_targets": missing,
        "registered_gates": {
            "per_target_inner_u_rmse_ratio_max": 0.90,
            "per_target_smooth_control_unified_over_legacy_max": 1.00,
            "turnover_policy": "resolved within 0.25 BMAJ or explicitly unresolved",
        },
        "gate_pass": bool(complete and all(row["gate_pass"] for row in targets)),
    }
    _write_json(root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
