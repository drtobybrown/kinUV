#!/usr/bin/env python3
"""Repair native-radio kinUV overlays in accepted production PVD figures.

This command only re-renders the PVD PDF/PNG pairs from frozen checkpoint
products.  It archives each complete target tree before replacement and then
rebuilds the two production manifests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import matplotlib

matplotlib.use("Agg")
import numpy as np

from kinuv.diagnostics.delivery import (
    _native_radio_curve_on_optical_lsrk_offsets,
    render_pvd,
)
from package_final_production import build_profiles, load_cube, load_json


REPO = Path(__file__).resolve().parents[1]
TARGETS = ("KGAS066", "KGAS007")
PVD_PATHS = (
    Path("benchmarks/pvd_kinuv_vs_kinms.pdf"),
    Path("benchmarks/pvd_kinuv_vs_kinms.png"),
    Path("plots/pv_diagrams.pdf"),
    Path("plots/pv_diagrams.png"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def git_state() -> dict:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    tracked_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            text=True,
        ).strip()
    )
    if branch != "dev" or tracked_dirty:
        raise RuntimeError("repair requires clean tracked files on dev")
    return {"branch": branch, "commit": commit, "dirty": False}


def verify_manifest(target_root: Path, manifest_path: Path) -> int:
    document = load_json(manifest_path)
    failures = []
    for relative, expected in document["files"].items():
        path = target_root / relative
        actual = record(path) if path.is_file() else None
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise RuntimeError(f"manifest verification failed: {failures[:3]}")
    return len(document["files"])


def archive_target(target_root: Path, archive_root: Path) -> tuple[Path, int]:
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / target_root.name / "20260909_pre-pvd-frame-overlay-repair.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(target_root, arcname=target_root.name)
    with tarfile.open(archive, "r:gz") as stream:
        members = sum(member.isfile() for member in stream.getmembers())
    expected = sum(path.is_file() for path in target_root.rglob("*"))
    if members != expected:
        raise RuntimeError(f"archive member count {members} != source count {expected}")
    return archive, members


def render_target(target_root: Path, output: Path) -> tuple[dict, dict]:
    best = target_root / "best_model"
    parameters = load_json(best / "parameters.json")
    selected = load_json(best / "selected_map.json")
    replay = load_json(best / "replay.json")
    geometry = parameters["diagnostic_geometry_lsrk_optical"]
    kinms_document = load_json(target_root / "benchmarks" / "kinms_fit_result.json")
    kinms = kinms_document.get("fitted", kinms_document)
    config = load_json(best / "config.json")
    data, header = load_cube(Path(config["diagnostic_cube"]))
    mask, _ = load_cube(Path(config["diagnostic_mask"]))
    model, _ = load_cube(best / "model_on_science_grid.fits")
    kinms_cube, _ = load_cube(target_root / "benchmarks" / "kinms_model_k.fits")
    with np.load(target_root / "benchmarks" / "moments.npz", allow_pickle=False) as archive:
        moment0 = np.asarray(archive["data_moment0"])
    profiles = build_profiles(
        target_root.name, target_root, header, parameters, selected, kinms, moment0
    )
    output.mkdir(parents=True, exist_ok=True)
    render_pvd(
        target_root.name,
        (data, model, kinms_cube),
        mask > 0.5,
        header,
        geometry,
        "MAP",
        profiles,
        output,
    )

    native_vsys = float(parameters["fitted_native_parameters"]["vsys_kms"])
    correction = float(replay["frame"]["frequency_equivalent_correction_kms"])
    radius = np.asarray(profiles["radius"])
    projected = np.asarray(profiles["kinuv_projected"])
    endpoint_offset = float(radius[-1])
    transformed = _native_radio_curve_on_optical_lsrk_offsets(
        np.array([-endpoint_offset, 0.0, endpoint_offset]),
        radius,
        projected,
        native_vsys,
        correction,
    )
    reporting_vsys = float(transformed[1])
    native_speed = float(projected[-1])
    metrics = {
        "radius_arcsec": endpoint_offset,
        "native_projected_speed_kms": native_speed,
        "reporting_vsys_optical_lsrk_kms": reporting_vsys,
        "approaching_velocity_optical_lsrk_kms": float(transformed[0]),
        "receding_velocity_optical_lsrk_kms": float(transformed[2]),
        "approaching_offset_magnitude_kms": reporting_vsys - float(transformed[0]),
        "receding_offset_kms": float(transformed[2]) - reporting_vsys,
        "maximum_naive_overlay_error_kms": float(
            np.max(
                np.abs(
                    transformed
                    - (
                        reporting_vsys
                        + np.array([-native_speed, 0.0, native_speed])
                    )
                )
            )
        ),
    }
    sources = {
        str(path.relative_to(target_root)): record(path)
        for path in (
            best / "parameters.json",
            best / "replay.json",
            best / "rotation_curve.npz",
            best / "model_on_science_grid.fits",
            target_root / "benchmarks" / "kinms_model_k.fits",
            target_root / "benchmarks" / "moments.npz",
        )
    }
    sources.update(
        {
            "external_diagnostic_cube": {
                "path": str(Path(config["diagnostic_cube"]).resolve()),
                **record(Path(config["diagnostic_cube"])),
            },
            "external_diagnostic_mask": {
                "path": str(Path(config["diagnostic_mask"]).resolve()),
                **record(Path(config["diagnostic_mask"])),
            },
        }
    )
    return metrics, sources


def replace_pvds(target_root: Path, rendered: Path) -> tuple[dict, dict]:
    before = {str(path): record(target_root / path) for path in PVD_PATHS}
    for suffix in ("pdf", "png"):
        source = rendered / f"pvd_kinuv_vs_kinms.{suffix}"
        for relative in (
            Path(f"benchmarks/pvd_kinuv_vs_kinms.{suffix}"),
            Path(f"plots/pv_diagrams.{suffix}"),
        ):
            temporary = target_root / relative.with_name(relative.name + ".tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target_root / relative)
    after = {str(path): record(target_root / path) for path in PVD_PATHS}
    for suffix in ("pdf", "png"):
        if after[f"benchmarks/pvd_kinuv_vs_kinms.{suffix}"] != after[f"plots/pv_diagrams.{suffix}"]:
            raise RuntimeError(f"benchmark and plot {suffix} copies differ")
    return before, after


def rebuild_manifests(target_root: Path, state: dict, created: str) -> tuple[int, int]:
    best_manifest = target_root / "best_model" / "manifest.json"
    root_manifest = target_root / "manifest.json"
    excluded = {"manifest.json", "MANIFEST.json"}
    files = {
        path.relative_to(target_root).as_posix(): record(path)
        for path in sorted(target_root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    old_best = load_json(best_manifest)
    old_best.update({"created_utc": created, "git": state, "files": files})
    write_json_atomic(best_manifest, old_best)
    files["best_model/manifest.json"] = record(best_manifest)
    old_root = load_json(root_manifest)
    old_root.update({"created_utc": created, "git": state, "files": files})
    write_json_atomic(root_manifest, old_root)
    return verify_manifest(target_root, best_manifest), verify_manifest(target_root, root_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    args = parser.parse_args()
    state = git_state()
    created = datetime.now(timezone.utc).isoformat()

    roots = {target: args.production_root / target for target in args.targets}
    for root in roots.values():
        verify_manifest(root, root / "manifest.json")
        verify_manifest(root, root / "best_model" / "manifest.json")

    with tempfile.TemporaryDirectory(prefix="kinuv-pvd-repair-", dir=args.production_root.parent) as temporary:
        temporary_root = Path(temporary)
        rendered = {}
        for target, root in roots.items():
            output = temporary_root / target
            metrics, sources = render_target(root, output)
            rendered[target] = (output, metrics, sources)

        for target, root in roots.items():
            archive, archive_members = archive_target(root, args.archive_root)
            output, metrics, sources = rendered[target]
            before, after = replace_pvds(root, output)
            provenance = {
                "schema_version": "kinuv-pvd-overlay-repair-v1",
                "created_utc": created,
                "target_id": target,
                "git": state,
                "operation": "PVD-only endpoint-wise native radio/TOPO to optical-LSRK overlay conversion",
                "fit_or_rescore_performed": False,
                "cube_mask_or_profile_changed": False,
                "archive": {
                    "path": str(archive.resolve()),
                    "members": archive_members,
                    **record(archive),
                },
                "sources": sources,
                "endpoint_metrics": metrics,
                "replaced_files_before": before,
                "replaced_files_after": after,
            }
            write_json_atomic(root / "provenance" / "PVD_OVERLAY_REPAIR.json", provenance)
            best_count, root_count = rebuild_manifests(root, state, created)
            print(
                json.dumps(
                    {
                        "target": target,
                        "archive": str(archive.resolve()),
                        "best_manifest_files": best_count,
                        "root_manifest_files": root_count,
                        "endpoint_metrics": metrics,
                        "pvd_files": after,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    main()
