#!/usr/bin/env python3
"""Run the registered S2 turnover grid and twelve-start geometry fits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

_repo = Path(__file__).resolve().parents[1]
_scratch_spec = importlib.util.spec_from_file_location(
    "_kinuv_scratch", _repo / "src/kinuv/scratch.py"
)
_scratch_module = importlib.util.module_from_spec(_scratch_spec)
_scratch_spec.loader.exec_module(_scratch_module)
_scratch_module.apply_scratch_env()

from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.infer.s2 import (
    build_s2_value_gradient,
    deterministic_start_grid,
    fit_s2_start,
    propagate_equal_weight_ar1,
)
from kinuv.io.vis import load_target_vis


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True
            ).strip()
        ),
    }


def emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True), flush=True)


def covariance_for_target(metrics: dict, target_id: str, n_bin: int):
    target = next(row for row in metrics["targets"] if row["target_id"] == target_id)
    if target["covariance"]["selected"] != "C1":
        raise ValueError(f"{target_id}: this S2 run expects the frozen selected C1 model")
    source = next(iter(target["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin)


def checkpoint(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def circular_pa_separation_deg(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def run_target(config_path: Path, covariance_metrics: dict, output: Path, maxiter: int):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_id = config["target_id"]
    target_dir = output / target_id
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_dir / "geometry.json"
    if checkpoint_path.exists():
        record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        record = {
            "schema_version": "kinuv-s2-geometry-v1",
            "target_id": target_id,
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "fixed_fits": [],
            "joint_fits": [],
        }

    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load_metadata = load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )
    grid = image_grid_for_vis(data)
    template = load_sb_template(grid, ico_path=config["template_ico"])
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    pa_seed = float(config["geometry"]["pa_seed_deg"])
    parameter_seed = config["stage_a"]["parameter_seed"]
    vsys_seed = float(parameter_seed["vsys_kms"])
    ratios = [float(value) for value in config["s2"]["turnover_over_bmaj_grid"]]
    covariance = covariance_for_target(covariance_metrics, target_id, data.n_bin)
    value_gradient = build_s2_value_gradient(
        data,
        template,
        grid,
        covariance,
        vsys_seed_kms=vsys_seed,
        bmaj_arcsec=bmaj,
    )
    record.update(
        {
            "visibility_path": str(Path(config["visibility_npz"]).resolve()),
            "visibility_sha256": sha256(Path(config["visibility_npz"])),
            "load": load_metadata,
            "n_row": int(data.vis.shape[0]),
            "n_channel": int(data.vis.shape[1]),
            "native_frame": config["spectral_frame"]["likelihood_frame"],
            "reporting_frame": config["spectral_frame"]["reporting_frame"],
            "bmaj_ref_arcsec": bmaj,
            "covariance": covariance.__dict__,
            "turnover_grid": ratios,
            "maxiter": int(maxiter),
        }
    )
    done_fixed = {
        (float(row["fixed_turnover_over_bmaj"]), int(row["start_id"]))
        for row in record["fixed_fits"]
    }
    for ratio in ratios:
        starts = deterministic_start_grid(pa_seed, vsys_seed, bmaj, ratio)
        for start in starts:
            key = (ratio, int(start["start_id"]))
            if key in done_fixed:
                continue
            start["flux"] = float(parameter_seed["flux"])
            start["gas_sigma_kms"] = float(parameter_seed["gas_sigma_kms"])
            result = fit_s2_start(
                data,
                template,
                grid,
                start,
                covariance,
                pa_seed_deg=pa_seed,
                vsys_seed_kms=vsys_seed,
                bmaj_arcsec=bmaj,
                fixed_turnover_over_bmaj=ratio,
                maxiter=maxiter,
                value_gradient=value_gradient,
            )
            row = result.to_dict()
            record["fixed_fits"].append(row)
            checkpoint(checkpoint_path, record)
            emit(
                {
                    "target": target_id,
                    "phase": "fixed",
                    "turnover_over_bmaj": ratio,
                    "start_id": result.start_id,
                    "chi2": result.chi2,
                    "projected_gradient_inf": result.projected_gradient_inf,
                    "success": result.success,
                }
            )

    done_joint = {int(row["start_id"]) for row in record["joint_fits"]}
    for start_id in range(12):
        if start_id in done_joint:
            continue
        candidates = [row for row in record["fixed_fits"] if int(row["start_id"]) == start_id]
        best_fixed = min(candidates, key=lambda row: row["chi2"])
        p = best_fixed["parameters"]
        start = {
            "start_id": start_id,
            "flux": p["flux"],
            "pa_deg": p["pa_deg"],
            "vsys_kms": p["vsys_kms"],
            "gas_sigma_kms": p["gas_sigma_kms"],
            "dx_arcsec": p["dx_arcsec"],
            "dy_arcsec": p["dy_arcsec"],
            "u_kms": p["u_kms"],
            "inclination_deg": p["inclination_deg"],
            "r_t_arcsec": p["r_t_arcsec"],
        }
        result = fit_s2_start(
            data,
            template,
            grid,
            start,
            covariance,
            pa_seed_deg=pa_seed,
            vsys_seed_kms=vsys_seed,
            bmaj_arcsec=bmaj,
            fixed_turnover_over_bmaj=None,
            maxiter=maxiter,
            value_gradient=value_gradient,
        )
        record["joint_fits"].append(result.to_dict())
        checkpoint(checkpoint_path, record)
        emit(
            {
                "target": target_id,
                "phase": "joint",
                "start_id": result.start_id,
                "chi2": result.chi2,
                "projected_gradient_inf": result.projected_gradient_inf,
                "success": result.success,
            }
        )

    joint = record["joint_fits"]
    best = min(joint, key=lambda row: row["chi2"])
    n_complex = int(data.vis.size)
    cluster_delta_chi2 = 2.0 * n_complex * 1.0e-3
    likelihood_cluster = [
        row
        for row in joint
        if row["chi2"] - best["chi2"] <= cluster_delta_chi2
    ]
    converged_cluster = [
        row for row in likelihood_cluster if row["projected_gradient_inf"] <= 1.0e-3
    ]
    consensus = []
    for row in converged_cluster:
        p = row["parameters"]
        q = best["parameters"]
        checks = (
            abs(p["u_kms"] - q["u_kms"]) <= max(1.0, 0.01 * max(q["u_kms"], 1.0)),
            abs(p["turnover_over_bmaj"] - q["turnover_over_bmaj"]) <= 0.01,
            circular_pa_separation_deg(p["pa_deg"], q["pa_deg"]) <= 1.0,
            abs(p["vsys_kms"] - q["vsys_kms"]) <= 0.1 * data.dv_kms,
            abs(p["gas_sigma_kms"] - q["gas_sigma_kms"])
            <= max(0.2, 0.02 * q["gas_sigma_kms"]),
        )
        if all(checks):
            consensus.append(row)
    turnover = float(best["parameters"]["turnover_over_bmaj"])
    endpoint = min(abs(turnover - min(ratios)), abs(turnover - max(ratios))) <= 1.0e-4
    record["best_joint"] = best
    promoted_boundary = [
        name
        for name in best["boundary_parameters"]
        if name not in {"cos_inclination"}
    ]
    frame_drift = float(
        config["spectral_frame"]["frequency_correction_peak_to_peak_kms"]
    )
    record["gates"] = {
        "fixed_fit_count_72": len(record["fixed_fits"]) == 72,
        "joint_fit_count_12": len(joint) == 12,
        "top_likelihood_cluster_consensus": len(consensus) >= 2,
        "best_projected_gradient": best["projected_gradient_inf"] <= 1.0e-3,
        "no_promoted_boundary_pressure": len(promoted_boundary) == 0,
        "turnover_not_endpoint": not endpoint,
        "spectral_frame_drift_below_1_kms": frame_drift < 1.0,
    }
    record["cluster_definition"] = {
        "maximum_delta_reduced_chi2": 1.0e-3,
        "maximum_delta_chi2": cluster_delta_chi2,
        "required_independent_starts": 2,
        "projected_velocity_relative_tolerance": 0.01,
        "turnover_over_bmaj_absolute_tolerance": 0.01,
        "pa_tolerance_deg": 1.0,
        "systemic_tolerance_fit_channels": 0.1,
        "dispersion_relative_tolerance": 0.02,
        "inclination_note": "not a promoted intrinsic quantity under an isotropic orientation prior",
    }
    record["likelihood_cluster_start_ids"] = [row["start_id"] for row in likelihood_cluster]
    record["consensus_start_ids"] = [row["start_id"] for row in consensus]
    record["promoted_boundary_parameters"] = promoted_boundary
    record["accepted"] = all(record["gates"].values())
    checkpoint(checkpoint_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_configs", type=Path, nargs="+")
    parser.add_argument("--covariance-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()
    state = git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("S2 geometry requires a clean exact commit on dev")
    args.output.mkdir(parents=True, exist_ok=True)
    covariance_metrics = json.loads(args.covariance_metrics.read_text(encoding="utf-8"))
    targets = [
        run_target(path, covariance_metrics, args.output, args.maxiter)
        for path in args.target_configs
    ]
    summary = {
        "schema_version": "kinuv-s2-geometry-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "covariance_metrics": str(args.covariance_metrics.resolve()),
        "covariance_metrics_sha256": sha256(args.covariance_metrics),
        "targets": [
            {
                "target_id": row["target_id"],
                "accepted": row["accepted"],
                "gates": row["gates"],
                "best_joint": row["best_joint"],
            }
            for row in targets
        ],
        "accepted": all(row["accepted"] for row in targets),
    }
    summary_path = args.output / "summary.json"
    checkpoint(summary_path, summary)
    manifest = {
        "schema_version": "kinuv-s2-geometry-manifest-v1",
        "code_commit": state["commit"],
        "files": {},
    }
    for path in sorted(args.output.rglob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        manifest["files"][str(path.relative_to(args.output))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    checkpoint(args.output / "MANIFEST.json", manifest)
    emit({"stage": "S2-geometry", "accepted": summary["accepted"]})
    if not summary["accepted"]:
        raise SystemExit("S2 geometry gates did not all pass")


if __name__ == "__main__":
    main()
