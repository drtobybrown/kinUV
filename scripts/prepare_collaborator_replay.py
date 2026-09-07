#!/usr/bin/env python3
"""Bind selected collaborator MAP fits into an exact S4 replay checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


TARGETS = ("KGAS066", "KGAS007")


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
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
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
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for target in TARGETS:
        selected_path = args.map_root / target / "selected_map.json"
        selected_doc = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = selected_doc["selected"]
        source_path = Path(selected["inputs"]["checkpoint"]["path"])
        checkpoint = json.loads(source_path.read_text(encoding="utf-8"))
        fit = selected["result"]
        candidate = fit["candidate"]
        checkpoint.update(
            {
                "schema_version": "kinuv-collaborator-map-checkpoint-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "accepted": True,
                "fits": [fit],
                "selection": {
                    "preferred_candidate": candidate,
                    "two_zone_dispersion_parent": (
                        "supported_rings"
                        if selected.get("two_zone_uses_rings", False)
                        else "joint_emissivity"
                    ),
                    "selection_basis": "lowest visibility chi2 among four bounded joint MAP starts",
                    "selected_start": selected_doc["selected_start"],
                },
                "collaborator_delivery": {
                    "selected_map": str(selected_path.resolve()),
                    "selected_map_sha256": sha256(selected_path),
                    "source_checkpoint": str(source_path.resolve()),
                    "source_checkpoint_sha256": sha256(source_path),
                    "emissivity_fixed": True,
                    "fit_geometry_jointly": True,
                },
            }
        )
        destination = args.output_root / target / "ablations.json"
        write_json_atomic(destination, checkpoint)
        records.append(
            {
                "target_id": target,
                "candidate": candidate,
                "selected_start": selected_doc["selected_start"],
                "chi2": fit["chi2"],
                "path": str(destination.resolve()),
                "sha256": sha256(destination),
            }
        )
        print(
            json.dumps(
                {
                    "target": target,
                    "candidate": candidate,
                    "selected_start": selected_doc["selected_start"],
                    "output": str(destination),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_json_atomic(
        args.output_root / "summary.json",
        {
            "schema_version": "kinuv-collaborator-map-checkpoint-summary-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "targets": records,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
