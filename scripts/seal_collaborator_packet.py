#!/usr/bin/env python3
"""Install and checksum-seal the immutable MAP-only collaborator packet."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


TARGETS = ("KGAS066", "KGAS007")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(root: Path, manifest_path: Path) -> int:
    manifest = load_json(manifest_path)
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or file_record(path) != {
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            raise ValueError(f"manifest verification failed: {path}")
    return len(manifest["files"])


def git_state(repo: Path) -> dict:
    return {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, text=True
        ).strip(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repo, text=True
            ).strip()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--nuts-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--subbeam-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    state = git_state(repo)
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("packet sealing requires a clean dev checkout")

    synthetic_count = verify_manifest(args.synthetic_root, args.synthetic_root / "MANIFEST.json")
    subbeam_count = verify_manifest(args.subbeam_root, args.subbeam_root / "MANIFEST.json")
    controller_path = args.nuts_root / "controller_status.json"
    controller = load_json(controller_path)
    target_records = []
    installed_roots = []
    for target in TARGETS:
        source = args.candidate_root / target
        render_count = verify_manifest(source, source / "MANIFEST.json")
        selected_path = args.map_root / target / "selected_map.json"
        selected = load_json(selected_path)
        destination = args.production_root / target / "meeting_candidate" / args.run_id
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        shutil.copy2(selected_path, destination / "best_model" / "selected_map.json")
        packet_manifest = {
            "schema_version": "kinuv-collaborator-map-packet-target-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "MAP_ONLY_CANDIDATE",
            "posterior_status": controller["state"],
            "target_id": target,
            "git": state,
            "selected_start": selected["selected_start"],
            "selected_chi2": selected["selected"]["result"]["chi2"],
            "render_manifest_entries_verified": render_count,
            "files": {
                path.relative_to(destination).as_posix(): file_record(path)
                for path in sorted(destination.rglob("*"))
                if path.is_file() and path.name != "PACKET_MANIFEST.json"
            },
        }
        write_json_atomic(destination / "PACKET_MANIFEST.json", packet_manifest)
        installed_roots.append(destination)
        target_records.append(
            {
                "target_id": target,
                "path": str(destination.resolve()),
                "manifest": file_record(destination / "PACKET_MANIFEST.json"),
                "selected_start": selected["selected_start"],
                "selected_chi2": selected["selected"]["result"]["chi2"],
                "selected_map": file_record(selected_path),
            }
        )

    packet_root = args.production_root / "meeting_packets" / args.run_id
    if packet_root.exists():
        raise FileExistsError(packet_root)
    packet_root.mkdir(parents=True)
    shutil.copy2(controller_path, packet_root / "nuts_controller_status_at_seal.json")
    index = {
        "schema_version": "kinuv-collaborator-meeting-packet-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "status": "MAP_ONLY_CANDIDATE",
        "posterior_status": controller["state"],
        "git": state,
        "targets": target_records,
        "synthetic_evidence": {
            "s4": {
                "root": str(args.synthetic_root.resolve()),
                "manifest": file_record(args.synthetic_root / "MANIFEST.json"),
                "verified_entries": synthetic_count,
            },
            "s5_subbeam": {
                "root": str(args.subbeam_root.resolve()),
                "manifest": file_record(args.subbeam_root / "MANIFEST.json"),
                "verified_entries": subbeam_count,
            },
            "scores_modified": False,
            "fits_rerun": False,
        },
        "nuts": {
            "live_root": str(args.nuts_root.resolve()),
            "controller_pid": controller["controller_pid"],
            "state_at_seal": controller["state"],
            "status_snapshot": file_record(packet_root / "nuts_controller_status_at_seal.json"),
        },
        "claims": {
            "real_data": "visibility MAP; restored cubes are supporting diagnostics",
            "synthetic": "registered thin axisymmetric arctan experiments only",
            "kgas066_turnover_error_ratio": 0.02493,
            "kgas066_inner_beam_rmse_ratio": 0.01438,
            "kgas007_turnover_error_ratio": 0.27665,
            "kgas007_inner_beam_rmse_ratio": 0.13749,
        },
    }
    write_json_atomic(packet_root / "INDEX.json", index)
    lines = [
        f"# Collaborator packet: {args.run_id}",
        "",
        "Status: **MAP_ONLY_CANDIDATE**. The posterior campaign was running when this immutable packet was sealed.",
        "",
        "| Target | Selected start | Visibility chi-square | Candidate path |",
        "|---|---:|---:|---|",
    ]
    for row in target_records:
        lines.append(
            f"| {row['target_id']} | {row['selected_start']} | {row['selected_chi2']:.6f} | `{row['path']}` |"
        )
    lines += [
        "",
        "The real-data panels are diagnostics of the selected visibility MAP and do not treat the restored KinMS cube as truth. Synthetic scores and seed identities are inherited unchanged from the checksum-verified S4/S5 records. Posterior intervals are absent from this packet and must not be inferred from the MAP curves.",
        "",
    ]
    (packet_root / "README.md").write_text("\n".join(lines), encoding="ascii")
    write_json_atomic(
        packet_root / "MANIFEST.json",
        {
            "schema_version": "kinuv-collaborator-meeting-packet-manifest-v1",
            "git": state,
            "files": {
                path.relative_to(packet_root).as_posix(): file_record(path)
                for path in sorted(packet_root.rglob("*"))
                if path.is_file() and path.name != "MANIFEST.json"
            },
        },
    )
    for root in [*installed_roots, packet_root]:
        for path in root.rglob("*"):
            os.chmod(path, 0o440 if path.is_file() else 0o550)
        os.chmod(root, 0o550)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": "MAP_ONLY_CANDIDATE",
                "posterior_status": controller["state"],
                "targets": len(target_records),
                "synthetic_entries_verified": synthetic_count + subbeam_count,
                "packet": str(packet_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
