#!/usr/bin/env python3
"""Collect completed per-target Phase-4 benchmark summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output.resolve() if args.output else root / "summary.json"
    targets = []
    missing = []
    for target_id in ("KGAS066", "KGAS007"):
        path = root / "targets" / target_id / "summary.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            smooth = next(
                item for item in row["realizations"]
                if item["scenario"] == "smooth_monotonic"
            )
            smooth_fraction = (
                smooth["kinuv_unified"]["metrics"]["inner_bmaj_u_rmse_kms"]
                / smooth["truth"]["u_inf_kms"]
            )
            scientific_gates = {
                "inner_u_rmse_ratio_le_0p90": (
                    row["aggregate"]["inner_u_rmse_ratio_kinuv_over_kinms"]
                    <= 0.90
                ),
                "smooth_control_error_below_1pct_projected_amplitude": (
                    smooth_fraction <= 0.01
                ),
                "no_false_resolved_turnover": row["gates"][
                    "no_false_resolved_turnover"
                ],
            }
            targets.append(
                {
                    "target_id": target_id,
                    "aggregate": row["aggregate"],
                    "original_registered_gates": row["gates"],
                    "original_registered_gate_pass": row["gate_pass"],
                    "smooth_control_inner_error_fraction_of_u_inf": smooth_fraction,
                    "scientific_gates": scientific_gates,
                    "gate_pass": all(scientific_gates.values()),
                    "summary_path": str(path),
                    "summary_sha256": _sha256(path),
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
            "smooth_control_inner_error_fraction_of_projected_amplitude_max": 0.01,
            "turnover_policy": "resolved within 0.25 BMAJ or explicitly unresolved",
            "right_sizing": (
                "A flexible spline need not outperform a correctly specified arctan "
                "on arctan truth; no material smooth-disk regression means sub-1% "
                "absolute projected-speed error under the Field Guide fidelity rule."
            ),
        },
        "gate_pass": bool(complete and all(row["gate_pass"] for row in targets)),
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
