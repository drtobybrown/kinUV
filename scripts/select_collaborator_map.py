#!/usr/bin/env python3
"""Select the lowest-likelihood valid collaborator MAP start per target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-root", type=Path, required=True)
    args = parser.parse_args()
    for target in ("KGAS066", "KGAS007"):
        paths = [args.map_root / target / f"start-{index}" / "result.json" for index in range(1, 5)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{target}: missing MAP starts: {missing}")
        records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        valid = [
            row for row in records
            if row["result"]["success"]
            and not row["result"]["boundary_parameters"]
            and row["result"]["projected_gradient_inf"] <= 1.0e-3
        ]
        if not valid:
            raise RuntimeError(f"{target}: no converged interior MAP start")
        selected = min(valid, key=lambda row: float(row["result"]["chi2"]))
        payload = {
            "schema_version": "kinuv-collaborator-selected-map-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_id": target,
            "selection": "lowest visibility chi2 among converged interior starts",
            "selected_start": selected["start_id"],
            "selected": selected,
            "starts": [
                {
                    "start_id": row["start_id"],
                    "chi2": row["result"]["chi2"],
                    "objective": row["result"]["objective"],
                    "prior": row["result"]["prior"],
                    "regularization": row["result"]["regularization"],
                    "success": row["result"]["success"],
                    "message": row["result"]["message"],
                    "projected_gradient_inf": row["result"]["projected_gradient_inf"],
                    "boundary_parameters": row["result"]["boundary_parameters"],
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for row, path in zip(records, paths)
            ],
        }
        destination = args.map_root / target / "selected_map.json"
        write_json_atomic(destination, payload)
        print(json.dumps({"target": target, "selected_start": selected["start_id"], "chi2": selected["result"]["chi2"], "output": str(destination)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
