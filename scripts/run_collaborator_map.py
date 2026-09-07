#!/usr/bin/env python3
"""Run one bounded collaborator-delivery MAP start for one target."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.infer.s2 import propagate_equal_weight_ar1
from kinuv.infer.s3 import (
    build_positive_emissivity_basis,
    chart_from_parameters,
    fit_s3_candidate,
)
from kinuv.io.vis import load_target_vis


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
RECOVERY = WORKSPACE / "results/validation/crossdomain-recovery-s4-remediation-20260907-r1"
COVARIANCE = WORKSPACE / "results/validation/crossdomain-recovery-s2-20260907-r1/metrics.json"


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


def git_state() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    if branch != "dev" or dirty:
        raise RuntimeError("collaborator MAP requires a clean dev checkout")
    return {"commit": commit, "branch": branch, "dirty": dirty}


def covariance_for(target_id: str, n_bin: int):
    metrics = json.loads(COVARIANCE.read_text(encoding="utf-8"))
    row = next(item for item in metrics["targets"] if item["target_id"] == target_id)
    source = next(iter(row["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin)


def selected_checkpoint(target_id: str):
    path = RECOVERY / "s3" / target_id / "ablations.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    name = record["selection"]["preferred_candidate"]
    selected = next(row for row in record["fits"] if row["candidate"] == name)
    return path, record, selected


def apply_start(parameters: dict, target_id: str, start_id: int) -> dict:
    p = json.loads(json.dumps(parameters))
    if start_id == 1:
        return p
    if start_id == 2:
        if target_id == "KGAS066":
            p["arctan_u_kms"] = 205.0
            p["inclination_deg"] = 55.0
        else:
            p["u_knots_kms"] = (1.11 * np.asarray(p["u_knots_kms"])).tolist()
        return p
    if start_id == 3:
        if target_id == "KGAS066":
            p["arctan_u_kms"] = 210.0
            p["inclination_deg"] = 65.0
        else:
            p["u_knots_kms"] = (1.14 * np.asarray(p["u_knots_kms"])).tolist()
            p["inclination_deg"] = min(80.0, float(p["inclination_deg"]) + 10.0)
        return p
    if start_id == 4:
        p["vsys_kms"] = float(p["vsys_kms"]) - 10.0
        return p
    raise ValueError("start-id must be 1..4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--start-id", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()
    state = git_state()
    target_id = args.target
    config_path = REPO / "configs/targets" / f"{target_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_path, checkpoint, selected = selected_checkpoint(target_id)
    parameters = selected["parameters"]
    start_parameters = apply_start(parameters, target_id, args.start_id)

    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load_metadata = load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )
    grid = image_grid_for_vis(data)
    base_template = load_sb_template(grid, Path(config["template_ico"]))
    s2 = json.loads(Path(checkpoint["s2_summary_path"]).read_text(encoding="utf-8"))
    s2_parameters = next(
        row for row in s2["targets"] if row["target_id"] == target_id
    )["best_joint"]["parameters"]
    basis = build_positive_emissivity_basis(
        base_template,
        grid,
        np.radians(float(s2_parameters["pa_deg"])),
        np.radians(float(s2_parameters["inclination_deg"])),
    )
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    vsys_seed = float(config["stage_a"]["parameter_seed"]["vsys_kms"])
    knot_radii = np.asarray(checkpoint["velocity_support"]["knot_radii_arcsec"])
    fixed_weights = np.asarray(parameters["emissivity_weights"], dtype=np.float64)
    z0 = chart_from_parameters(
        start_parameters,
        vsys_seed_kms=vsys_seed,
        dv_kms=data.dv_kms,
        bmaj_arcsec=bmaj,
    )
    two_zone_uses_rings = (
        selected["candidate"] == "two_zone_dispersion"
        and checkpoint["selection"]["two_zone_dispersion_parent"] == "supported_rings"
    )
    result, optimum = fit_s3_candidate(
        data,
        base_template,
        basis,
        knot_radii,
        grid,
        covariance_for(target_id, data.n_bin),
        z0,
        candidate=selected["candidate"],
        pa_seed_deg=float(config["geometry"]["pa_seed_deg"]),
        vsys_seed_kms=vsys_seed,
        bmaj_arcsec=bmaj,
        maxiter=args.maxiter,
        two_zone_uses_rings=two_zone_uses_rings,
        fixed_emissivity_weights=fixed_weights,
    )
    payload = {
        "schema_version": "kinuv-collaborator-map-start-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "target_id": target_id,
        "start_id": args.start_id,
        "start_parameters": start_parameters,
        "selected_parent": selected["candidate"],
        "two_zone_uses_rings": two_zone_uses_rings,
        "emissivity_fixed": True,
        "emissivity_weights": fixed_weights.tolist(),
        "result": result.to_dict(),
        "optimum_chart": optimum.tolist(),
        "load": load_metadata,
        "inputs": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256(checkpoint_path)},
            "covariance": {"path": str(COVARIANCE.resolve()), "sha256": sha256(COVARIANCE)},
            "visibility": {"path": config["visibility_npz"], "sha256": sha256(Path(config["visibility_npz"]))},
        },
    }
    destination = args.output_root / target_id / f"start-{args.start_id}" / "result.json"
    write_json_atomic(destination, payload)
    print(json.dumps({
        "target": target_id,
        "start": args.start_id,
        "chi2": result.chi2,
        "success": result.success,
        "boundary": list(result.boundary_parameters),
        "output": str(destination),
    }, sort_keys=True, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
