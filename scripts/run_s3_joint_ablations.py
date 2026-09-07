#!/usr/bin/env python3
"""Run sequential S3 visibility-model ablations on configured targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

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
from kinuv.infer.s2 import propagate_equal_weight_ar1
from kinuv.infer.s3 import (
    CANDIDATES,
    build_positive_emissivity_basis,
    build_s3_objective,
    fit_s3_candidate,
    initial_chart,
    supported_knot_radii,
)
from kinuv.io.vis import load_target_vis


REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
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


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _covariance(metrics: dict, target_id: str, n_bin: int):
    row = next(item for item in metrics["targets"] if item["target_id"] == target_id)
    source = next(iter(row["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin)


def _s2_target(summary: dict, target_id: str) -> dict:
    row = next(item for item in summary["targets"] if item["target_id"] == target_id)
    if not row["accepted"]:
        raise ValueError(f"{target_id}: S2 geometry is not accepted")
    return row


def _template_inputs(config: dict) -> dict:
    paths = {
        "visibility": Path(config["visibility_npz"]),
        "template_ico": Path(config["template_ico"]),
        "fit_window_cube": Path(config["fit_window_cube"]),
    }
    error_path = paths["template_ico"].with_name(
        f"{paths['template_ico'].stem}_err{paths['template_ico'].suffix}"
    )
    paths["template_error"] = error_path
    return {
        name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in paths.items()
    }


def run_target(config_path, s2_summary_path, covariance_metrics, output, maxiter):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    s2_summary = json.loads(Path(s2_summary_path).read_text(encoding="utf-8"))
    target_id = config["target_id"]
    s2_row = _s2_target(s2_summary, target_id)
    s2_parameters = s2_row["best_joint"]["parameters"]
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load_metadata = load_target_vis(
        config["visibility_npz"],
        cube_path=config["fit_window_cube"],
        phase_dir_rad=phase_rad,
    )
    grid = image_grid_for_vis(data)
    template = load_sb_template(grid, Path(config["template_ico"]))
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    pa = np.radians(float(s2_parameters["pa_deg"]))
    inc = np.radians(float(s2_parameters["inclination_deg"]))
    basis = build_positive_emissivity_basis(template, grid, pa, inc)
    knot_radii, r95 = supported_knot_radii(template, grid, pa, inc, bmaj)
    vsys_seed = float(config["stage_a"]["parameter_seed"]["vsys_kms"])
    covariance = _covariance(covariance_metrics, target_id, data.n_bin)
    initial = initial_chart(
        s2_parameters,
        knot_radii,
        basis.natural_weights,
        vsys_seed_kms=vsys_seed,
        dv_kms=data.dv_kms,
        bmaj_arcsec=bmaj,
    )
    initial_logits = np.log(basis.natural_weights[1:] / basis.natural_weights[0])
    baseline_objective, _ = build_s3_objective(
        data,
        template,
        basis,
        knot_radii,
        grid,
        covariance,
        candidate="baseline_arctan",
        vsys_seed_kms=vsys_seed,
        bmaj_arcsec=bmaj,
        initial_emissivity_logits=initial_logits,
    )
    initial_value = float(baseline_objective(initial))
    initial_center_prior = (
        (float(s2_parameters["dx_arcsec"]) / 0.5) ** 2
        + (float(s2_parameters["dy_arcsec"]) / 0.5) ** 2
    )
    replay_chi2 = 2.0 * initial_value - initial_center_prior
    expected_chi2 = float(s2_row["best_joint"]["chi2"])

    record = {
        "schema_version": "kinuv-s3-joint-ablations-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256(Path(config_path)),
        "s2_summary_path": str(Path(s2_summary_path).resolve()),
        "s2_summary_sha256": sha256(Path(s2_summary_path)),
        "inputs": _template_inputs(config),
        "load": load_metadata,
        "n_row": int(data.vis.shape[0]),
        "n_channel": int(data.vis.shape[1]),
        "n_complex": int(data.vis.size),
        "maxiter": int(maxiter),
        "bmaj_arcsec": bmaj,
        "emissivity_basis": {
            "rank": 3,
            "positive": True,
            "unit_integral_components": True,
            "natural_weights": basis.natural_weights.tolist(),
            "radial_edges_arcsec": basis.radial_edges_arcsec.tolist(),
            "r95_arcsec": basis.r95_arcsec,
        },
        "velocity_support": {
            "parameterization": "projected_speed_u_equals_vc_sin_i",
            "knot_radii_arcsec": knot_radii.tolist(),
            "r95_arcsec": r95,
            "knot_count": 4,
            "inner_boundary": "solid_body_to_first_knot",
            "outer_boundary": "flat_after_last_knot",
        },
        "covariance": covariance.__dict__,
        "s2_replay": {
            "expected_chi2": expected_chi2,
            "recomputed_chi2": replay_chi2,
            "absolute_error": abs(replay_chi2 - expected_chi2),
        },
        "fits": [],
    }
    checkpoint = Path(output) / target_id / "ablations.json"
    write_json_atomic(checkpoint, record)

    z = initial
    for candidate in CANDIDATES:
        if candidate == "supported_rings":
            arctan_u = 100.0 * z[8]
            rt = z[7] * bmaj
            z[9:13] = (
                arctan_u * (2.0 / np.pi) * np.arctan(knot_radii / rt) / 100.0
            )
        if candidate == "two_zone_dispersion":
            z[15] = z[3]
        result, z = fit_s3_candidate(
            data,
            template,
            basis,
            knot_radii,
            grid,
            covariance,
            z,
            candidate=candidate,
            pa_seed_deg=float(config["geometry"]["pa_seed_deg"]),
            vsys_seed_kms=vsys_seed,
            bmaj_arcsec=bmaj,
            maxiter=maxiter,
        )
        record["fits"].append(result.to_dict())
        write_json_atomic(checkpoint, record)
        emit(
            {
                "target": target_id,
                "candidate": candidate,
                "chi2": result.chi2,
                "gradient_per_complex": result.projected_gradient_inf,
                "success": result.success,
            }
        )

    by_name = {row["candidate"]: row for row in record["fits"]}
    baseline = by_name["baseline_arctan"]
    emissivity = by_name["joint_emissivity"]
    rings = by_name["supported_rings"]
    dispersion = by_name["two_zone_dispersion"]
    emissivity_gain = baseline["chi2"] - emissivity["chi2"]
    ring_gain = emissivity["chi2"] - rings["chi2"]
    dispersion_gain = rings["chi2"] - dispersion["chi2"]
    knot_hessian = rings["hessian"]["velocity_knots"]
    knot_boundaries = [name for name in rings["boundary_parameters"] if name.startswith("u_knot")]
    emissivity_retained = emissivity_gain > 4.0
    rings_retained = (
        ring_gain >= 10.0
        and knot_hessian["rank_relative_1e-8"] == 4
        and not knot_boundaries
        and rings["projected_gradient_inf"] <= 1.0e-3
    )
    dispersion_boundaries = {
        "log_sigma_inner", "log_sigma_outer"
    }.intersection(dispersion["boundary_parameters"])
    dispersion_retained = (
        rings_retained
        and dispersion_gain > 2.0
        and not dispersion_boundaries
        and dispersion["projected_gradient_inf"] <= 1.0e-3
    )
    preferred = (
        "two_zone_dispersion"
        if dispersion_retained
        else "supported_rings" if rings_retained else "joint_emissivity" if emissivity_retained else "baseline_arctan"
    )
    gates = {
        "s2_likelihood_replay_absolute_error_le_0p1": abs(replay_chi2 - expected_chi2) <= 0.1,
        "all_candidate_objectives_finite": all(np.isfinite(row["chi2"]) for row in record["fits"]),
        "all_candidate_gradients_le_1e_3": all(
            row["projected_gradient_inf"] <= 1.0e-3 for row in record["fits"]
        ),
        "supported_ring_gain_ge_10": ring_gain >= 10.0,
        "velocity_knot_subspace_identified": knot_hessian["rank_relative_1e-8"] == 4,
        "velocity_knots_interior": not knot_boundaries,
    }
    record.update(
        {
            "selection": {
                "emissivity_delta_chi2": emissivity_gain,
                "emissivity_retained_by_aic": emissivity_retained,
                "supported_rings_delta_chi2": ring_gain,
                "supported_rings_retained": rings_retained,
                "two_zone_dispersion_delta_chi2": dispersion_gain,
                "two_zone_dispersion_retained_by_aic": dispersion_retained,
                "preferred_candidate": preferred,
            },
            "gates": gates,
            "accepted": all(gates.values()),
        }
    )
    write_json_atomic(checkpoint, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-config", action="append", type=Path, required=True)
    parser.add_argument("--s2-summary", action="append", type=Path, required=True)
    parser.add_argument("--covariance-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()
    if len(args.target_config) != len(args.s2_summary):
        raise ValueError("one S2 summary is required for each target config")
    state = git_state()
    if state["branch"] != "dev" or state["dirty"]:
        raise RuntimeError("S3 ablations require a clean exact commit on dev")
    covariance = json.loads(args.covariance_metrics.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    targets = [
        run_target(config, summary, covariance, args.output, args.maxiter)
        for config, summary in zip(args.target_config, args.s2_summary)
    ]
    summary = {
        "schema_version": "kinuv-s3-joint-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "covariance_metrics": str(args.covariance_metrics.resolve()),
        "covariance_metrics_sha256": sha256(args.covariance_metrics),
        "targets": [
            {
                "target_id": row["target_id"],
                "accepted": row["accepted"],
                "gates": row["gates"],
                "selection": row["selection"],
                "preferred": next(
                    fit for fit in row["fits"]
                    if fit["candidate"] == row["selection"]["preferred_candidate"]
                ),
            }
            for row in targets
        ],
        "accepted": all(row["accepted"] for row in targets),
    }
    write_json_atomic(args.output / "summary.json", summary)
    manifest = {
        "schema_version": "kinuv-s3-joint-manifest-v1",
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
    write_json_atomic(args.output / "MANIFEST.json", manifest)
    emit({"stage": "S3", "accepted": summary["accepted"]})
    if not summary["accepted"]:
        raise SystemExit("S3 gates did not all pass")


if __name__ == "__main__":
    main()
