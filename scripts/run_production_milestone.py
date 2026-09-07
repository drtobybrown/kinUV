#!/usr/bin/env python3
"""Run one immutable kinUV MAP milestone from a target configuration.

This entry point performs a two-start Stage A visibility MAP, a frozen-geometry
Stage B ring MAP, visibility and image diagnostics, and the downstream KinMS
comparison. Existing converged NUTS draws may be consolidated and revalidated;
the script never launches a sampler.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from kinuv.diagnostics.figures import plot_leftover_chi2
from kinuv.diagnostics.flags import leftover_velocity_structured, map_quality_flags
from kinuv.diagnostics.kinms_benchmark import write_cube_benchmark
from kinuv.diagnostics.s1 import leftover_chi2
from kinuv.forward.sb import ico_template_metadata, load_sb_template
from kinuv.infer.map import PARAM_NAMES, _lbfgs_one_start, image_grid_for_vis
from kinuv.infer.nulls import fit_nonrotating_emission
from kinuv.infer.seeds import PA_AMBIGUITY_DEG, PA_BOUND_HALF_DEG, VSYS_BOUND_HALF_KM_S, stage_a_seeds
from kinuv.infer.stage_b import (
    nuisance_from_params,
    predict_binned as predict_stage_b,
    run_stage_b_map,
    stage_b_model_adequate,
)
from kinuv.io.vis import load_target_vis, optical_to_radio_kms, radio_to_optical_kms
from kinuv.likelihood.chi2 import chi2
from kinuv.profiles.rotation import V_K_MAX_KM_S, V_K_MIN_KM_S, arctan_vc, ring_vc
from kinuv.runner.plots import write_imaging_plots, write_model_cube
from kinuv.runner.state import terminal_validation_state
from kinuv.validation import ROTATION_GATE_ID, rotation_test_metrics

REPO = Path(__file__).resolve().parents[1]
TARGET_CONFIG_SCHEMA = "kinuv-production-target-v2"
RUN_MANIFEST_SCHEMA = "kinuv-run-manifest-v2"
SUMMARY_SCHEMA = "kinuv-milestone-summary-v2"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    return {"commit": commit, "short_commit": commit[:6], "dirty": dirty}


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _require_files(config: dict) -> dict[str, dict[str, str | int]]:
    keys = ("visibility_npz", "template_ico", "fit_window_cube", "diagnostic_cube", "diagnostic_mask")
    out = {}
    for key in keys:
        path = Path(config[key])
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path}")
        out[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    for key in ("cube", "result"):
        path = Path(config["kinms"][key])
        if not path.is_file():
            raise FileNotFoundError(f"kinms.{key}: {path}")
        out[f"kinms_{key}"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
    ico_path = Path(config["template_ico"])
    error_path = ico_path.with_name(f"{ico_path.stem}_err{ico_path.suffix}")
    if not error_path.is_file():
        raise FileNotFoundError(f"template_ico_error: {error_path}")
    out["template_ico_error"] = {
        "path": str(error_path),
        "bytes": error_path.stat().st_size,
        "sha256": _sha256(error_path),
    }
    return out


def _environment() -> dict:
    names = ("kinuv", "numpy", "scipy", "jax", "jax-finufft", "astropy", "matplotlib", "numpyro")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "jax_platform": os.environ.get("JAX_PLATFORMS", "default"),
    }


def _write_checksums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            lines.append(f"{_sha256(path)}  {path.relative_to(root)}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")


def _validate_target_config(config: dict) -> None:
    if config.get("schema_version") != TARGET_CONFIG_SCHEMA:
        raise ValueError("unsupported target configuration schema")
    if config.get("acceptance", {}).get("rotation_gate_id") != ROTATION_GATE_ID:
        raise ValueError("target config does not name the frozen rotation gate")


def _write_leftover(data, model, destination: Path, params: dict) -> dict:
    b_m, per_row, vel, per_chan = leftover_chi2(data, model)
    flag_params = dict(params)
    flag_params.setdefault("delta_chi2", 1.0)
    flags = map_quality_flags(
        flag_params,
        leftover_npz={"baseline_m": b_m, "chi2_row": per_row, "chi2_chan": per_chan},
    )
    rec = {
        "chi2_sum": float(np.sum(per_row)),
        "leftover_chi2_structured": bool(leftover_velocity_structured(b_m, per_row, per_chan)),
        "leftover_uv_span": bool(flags["leftover_uv_span"]),
        "leftover_vel_span": bool(flags["leftover_vel_span"]),
        "quote_inner_slope": False,
    }
    np.savez(destination / "leftover_chi2.npz", baseline_m=b_m, chi2_row=per_row, vel_kms=vel, chi2_chan=per_chan)
    plot_leftover_chi2(b_m, per_row, vel, per_chan, destination / "leftover_chi2.png")
    (destination / "leftover_chi2.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def _write_rotation_curve(destination: Path, stage_a: dict, stage_b: dict, selected: str) -> dict:
    import matplotlib.pyplot as plt

    rmax = float(stage_b["r_knots_arcsec"][-1])
    radius = np.linspace(0.0, rmax, 301)
    va = np.asarray(arctan_vc(radius, stage_a["v0_kms"], stage_a["r_t_arcsec"]), dtype=float)
    vb = np.asarray(ring_vc(radius, stage_b["r_knots_arcsec"], stage_b["v_knots_kms"]), dtype=float)
    np.savez(destination / "rotation_curve.npz", radius_arcsec=radius, stage_a_kms=va, stage_b_kms=vb)
    np.savetxt(
        destination / "rotation_curve.csv",
        np.column_stack([radius, va, vb]),
        delimiter=",",
        header="radius_arcsec,stage_a_kms,stage_b_kms",
        comments="",
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(radius, va, label="Stage A arctan", lw=1.8)
    ax.plot(radius, vb, label="Stage B rings", lw=1.8)
    ax.scatter(stage_b["r_knots_arcsec"], stage_b["v_knots_kms"], s=22, color="black", zorder=3)
    ax.set(xlabel="Radius (arcsec)", ylabel=r"$V_c$ (km s$^{-1}$)", title=f"Selected model: {selected}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination / "rotation_curve.png", dpi=180)
    plt.close(fig)
    return {"selected": selected, "radius_max_arcsec": rmax}


def _historical_posterior_reference(source: Path | None, destination: Path) -> dict:
    """Record old draws without presenting them as samples of this likelihood."""
    if source is None:
        result = {
            "status": "no_current_posterior", "source": None,
            "current_posterior_available": False,
            "current_rhat": None, "current_ess": None,
            "note": "Historical posterior archives are documented in PRODUCTION_RECORD.md.",
        }
        destination.write_text(json.dumps(result, indent=2) + "\n")
        return result
    required = ("posterior_samples.json", "summary.json", "METRICS.md", "config.yaml")
    for name in required:
        src = source / name
        if not src.is_file():
            raise FileNotFoundError(src)
    result = {
        "status": "incompatible_historical_likelihood_reference",
        "source": str(source),
        "source_checksums": {name: _sha256(source / name) for name in required},
        "sampler": "nuts",
        "current_posterior_available": False,
        "current_rhat": None,
        "current_ess": None,
        "note": (
            "The audited Wiener template changes the forward likelihood. "
            "Historical draws are linked for provenance only and are not "
            "copied or summarized as a current posterior."
        ),
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    _validate_target_config(config)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"immutable output exists and is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir()
    benchmark_dir = output / "benchmark"
    state = {"schema_version": RUN_MANIFEST_SCHEMA, "target_id": config["target_id"], "state_transitions": []}
    state["state_transitions"].append({"state": "PREFLIGHTED", "at": _utc()})
    code = _git_state()
    if code["dirty"]:
        raise RuntimeError("production milestone requires a clean git checkout")
    inputs = _require_files(config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(_environment(), indent=2) + "\n")
    state.update({"code": code, "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)}, "inputs": inputs})
    state["state_transitions"].append({"state": "RUNNING", "at": _utc()})
    (output / "manifest.json").write_text(json.dumps(state, indent=2) + "\n")

    started = perf_counter()
    phase = config.get("phase_center_deg")
    phase_rad = np.radians(phase) if phase is not None else None
    data, load_meta = load_target_vis(
        config["visibility_npz"], cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad
    )
    grid = image_grid_for_vis(data)
    template = load_sb_template(grid, ico_path=config["template_ico"])
    template_metadata = ico_template_metadata(Path(config["template_ico"]))
    geom = config["geometry"]
    i_rad = math.radians(float(geom["inclination_deg"]))
    pa_seed = float(geom["pa_seed_deg"])
    vsys_radio = float(optical_to_radio_kms(geom["vsys_optical_seed_kms"]))
    stage_a_cfg = config["stage_a"]
    bounds = {
        "pa_deg": (pa_seed - PA_BOUND_HALF_DEG, pa_seed + PA_BOUND_HALF_DEG),
        "vsys_kms": (vsys_radio - VSYS_BOUND_HALF_KM_S, vsys_radio + VSYS_BOUND_HALF_KM_S),
    }
    starts = []
    for pa in (pa_seed, pa_seed - PA_AMBIGUITY_DEG):
        seed = stage_a_seeds(pa_deg=pa)
        seed.update({key: float(value) for key, value in stage_a_cfg["parameter_seed"].items()})
        seed["pa_deg"] = pa
        seed.setdefault("vsys_kms", vsys_radio)
        rec = _lbfgs_one_start(
            data,
            template,
            grid,
            seed,
            0.0,
            int(stage_a_cfg["maxiter"]),
            pa,
            extra_bounds=bounds,
            i_rad=i_rad,
            xla=True,
        )
        starts.append(_jsonable(rec))
        print(json.dumps({"target": config["target_id"], "stage": "A", "start_pa": pa, "chi2": rec.chi2_map, "delta_chi2": rec.delta_chi2, "success": rec.success}), flush=True)
    stage_a = dict(max(starts, key=lambda item: item["delta_chi2"]))
    stage_a["starts"] = starts
    stage_a["inclination_deg_frozen"] = float(geom["inclination_deg"])
    stage_a["chi2_blank"] = float(stage_a["chi2_zero"])
    stage_a["delta_chi2_blank"] = float(stage_a["delta_chi2"])
    (output / "stage_a_map.json").write_text(json.dumps(stage_a, indent=2) + "\n")

    nonrot_seed = {
        key: float(value)
        for key, value in stage_a_cfg["parameter_seed"].items()
        if key in {"flux", "vsys_kms", "gas_sigma_kms", "dx_arcsec", "dy_arcsec"}
    }
    nonrot = _jsonable(
        fit_nonrotating_emission(
            data,
            template,
            grid,
            seeds=nonrot_seed,
            bounds={"vsys_kms": bounds["vsys_kms"]},
            maxiter=int(stage_a_cfg["maxiter"]),
            xla=True,
        )
    )
    if not nonrot["success"]:
        raise RuntimeError(f"non-rotating emitting-disk fit failed: {nonrot}")
    rotation_metrics = _jsonable(
        rotation_test_metrics(
            chi2_blank=float(stage_a["chi2_blank"]),
            chi2_nonrot=float(nonrot["chi2_nonrot"]),
            chi2_rot=float(stage_a["chi2_map"]),
        )
    )
    (output / "nonrotating_map.json").write_text(json.dumps(nonrot, indent=2) + "\n")
    (output / "rotation_test.json").write_text(
        json.dumps(rotation_metrics, indent=2) + "\n"
    )

    stage_b_rec = run_stage_b_map(
        data,
        nuisance_from_params(stage_a),
        template,
        grid,
        lam_reg=float(config["stage_b"]["lambda_regularization"]),
        v0_init=float(stage_a["v0_kms"]),
        rt_init=float(stage_a["r_t_arcsec"]),
        n_rings=int(config["stage_b"]["n_rings"]),
        chi2_stage_a=float(stage_a["chi2_map"]),
        maxiter=int(config["stage_b"]["maxiter"]),
        i_rad=i_rad,
        r_last_arcsec=float(config["stage_b"]["r_last_arcsec"]),
        target_id=str(config["target_id"]),
    )
    stage_b = _jsonable(stage_b_rec)
    stage_b["delta_chi2_vs_stage_a"] = float(stage_a["chi2_map"] - stage_b["chi2_map"])
    v_knots = np.asarray(stage_b["v_knots_kms"], dtype=float)
    stage_b["bound_pressure"] = bool(
        np.any(np.isclose(v_knots, V_K_MIN_KM_S, atol=1.0e-6))
        or np.any(np.isclose(v_knots, V_K_MAX_KM_S, atol=1.0e-6))
    )
    stage_b["oscillation_pass"] = False
    stage_b["oscillation_gate_status"] = "blocked_pending_mock_calibration"
    (output / "stage_b_map.json").write_text(json.dumps(stage_b, indent=2) + "\n")
    print(json.dumps({"target": config["target_id"], "stage": "B", "chi2": stage_b["chi2_map"], "delta_chi2_vs_a": stage_b["delta_chi2_vs_stage_a"], "success": stage_b["success"]}), flush=True)

    stage_b_accepted = stage_b_model_adequate(stage_b, None)
    selected = "stage_b" if stage_b_accepted else "stage_a"
    params = {name: float(stage_a[name]) for name in PARAM_NAMES}
    if selected == "stage_b":
        model = predict_stage_b(
            data,
            nuisance_from_params(params),
            template,
            grid,
            r_knots_arcsec=stage_b["r_knots_arcsec"],
            v_knots_kms=stage_b["v_knots_kms"],
            i_rad=i_rad,
        )
        expected_chi2 = float(stage_b["chi2_map"])
        rings = {"r_knots_arcsec": stage_b["r_knots_arcsec"], "v_knots_kms": stage_b["v_knots_kms"]}
    else:
        from kinuv.infer.map import predict_binned

        model = np.asarray(predict_binned(data, params, template, grid, i_rad=i_rad, xla=True))
        expected_chi2 = float(stage_a["chi2_map"])
        rings = {"r_knots_arcsec": None, "v_knots_kms": None}
    identity = float(chi2(data.vis, model, data.weights, data.s))
    identity_error = abs(identity - expected_chi2)
    leftover = _write_leftover(data, model, plots, params)
    model_native = write_model_cube(
        params,
        plots / "model_native.fits",
        data=data,
        tmpl=template,
        grid=grid,
        ico_path=config["template_ico"],
        object_name=config["target_id"],
        i_rad=i_rad,
        r_knots_arcsec=rings["r_knots_arcsec"],
        v_knots_kms=rings["v_knots_kms"],
        origin=f"kinUV milestone visibility {selected} MAP",
    )
    write_imaging_plots(
        output / "stage_a_map.json",
        model_native,
        plots,
        model_label=f"kinUV {selected.replace('_', ' ').title()} visibility MAP",
        data_cube=Path(config["diagnostic_cube"]),
        mask_cube=Path(config["diagnostic_mask"]),
        target_id=config["target_id"],
        catalog_vsys_optical=float(geom["vsys_optical_seed_kms"]),
    )
    rotation = _write_rotation_curve(plots, stage_a, stage_b, selected)
    fitted_vsys_opt = float(radio_to_optical_kms(params["vsys_kms"]))
    benchmark = write_cube_benchmark(
        target_id=config["target_id"],
        data_cube=Path(config["diagnostic_cube"]),
        mask_cube=Path(config["diagnostic_mask"]),
        kinuv_cube=plots / "model_on_10kms.fits",
        kinms_cube=Path(config["kinms"]["cube"]),
        output_dir=benchmark_dir,
        pa_deg=params["pa_deg"],
        inclination_deg=float(geom["inclination_deg"]),
        vsys_kms=fitted_vsys_opt,
        dx_arcsec=params["dx_arcsec"],
        dy_arcsec=params["dy_arcsec"],
    )
    shutil.copy2(config["kinms"]["result"], benchmark_dir / "kinms_fit_result.json")
    posterior = _historical_posterior_reference(
        Path(config["retained_posterior"]) if config.get("retained_posterior") else None,
        output / "historical_posterior_reference.json"
    )

    gates = {
        "preflight": {"status": "pass", "inputs_hashed": True, "standalone_import": True},
        "analytic_closure": {"status": "pass", "evidence": f"git:{code['commit']} test suite"},
        "mock_recovery": {"status": "pass", "evidence": "docs/reviews/artifacts/2026-08-29-s1-mock", "scope": "validated production transform and optimizer"},
        "blank_comparison": {
            "status": "diagnostic_only",
            "delta_chi2_blank": stage_a["delta_chi2_blank"],
        },
        "rotation_test": rotation_metrics,
        "covariance": {"status": "pass_for_map_baseline", "weight_scale": float(data.s), "structured_residual": leftover["leftover_chi2_structured"]},
        "map_stability": {"status": "pass" if len(starts) == 2 and identity_error <= config["acceptance"]["likelihood_identity_atol"] else "fail", "starts": 2, "identity_error": identity_error},
        "stage_b_model_adequacy": {
            "status": "blocked_pending_mock_calibration",
            "aic_stage_a": stage_b["aic_stage_a"],
            "aic_stage_b": stage_b["aic_stage_b"],
            "bound_pressure": stage_b["bound_pressure"],
            "max_omega_dimensionless": stage_b["max_omega_dimensionless"],
            "criterion": None,
        },
        "posterior": {
            "status": "blocked_no_current_likelihood_draws",
            "historical_reference": posterior["source"],
        },
        "promotion": {"status": "blocked_pending_rotation_bootstrap_and_stage_b_calibration"},
    }
    terminal_status = terminal_validation_state(gates)
    blocking = [
        name
        for name, gate in gates.items()
        if str(gate.get("status", "unknown")).startswith("fail")
    ]
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "milestone": "MILESTONE-001",
        "target_id": config["target_id"],
        "status": terminal_status,
        "selected_model": selected,
        "fit_domain": "complex visibility chi2",
        "stage_a": {
            key: stage_a[key]
            for key in (
                *PARAM_NAMES,
                "chi2_map",
                "chi2_blank",
                "delta_chi2_blank",
                "chi2_zero",
                "delta_chi2",
                "success",
                "nfev",
            )
        },
        "stage_b": stage_b,
        "likelihood_identity": {"saved_chi2": expected_chi2, "recomputed_chi2": identity, "absolute_error": identity_error},
        "data_contract": {"n_row": int(data.vis.shape[0]), "n_chan": int(data.vis.shape[1]), "dv_kms": float(data.dv_kms), "n_bin": int(data.n_bin), "weight_scale": float(data.s), "load": load_meta},
        "surface_brightness_template": template_metadata,
        "posterior": posterior,
        "benchmark_metrics": benchmark["metrics"],
        "rotation_curve": rotation,
        "gates": gates,
        "limitations": ["Posterior intervals are not coverage-calibrated and must not be quoted as calibrated credible intervals.", "The real-data inner velocity slope is not a promoted claim."],
        "runtime_seconds": perf_counter() - started,
        "code": code,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "METRICS.md").write_text(
        f"# {config['target_id']} MILESTONE-001 metrics\n\n"
        f"- Selected model: `{selected}`\n"
        f"- Stage A visibility chi2: {stage_a['chi2_map']:.6f}\n"
        f"- Stage B visibility chi2: {stage_b['chi2_map']:.6f}\n"
        f"- Delta chi2 versus blank visibilities: {stage_a['delta_chi2_blank']:.6f}\n"
        f"- Delta chi2 versus fitted non-rotating disk: {rotation_metrics['delta_chi2_nonrot']:.6f}\n"
        f"- Rotation-test status: {rotation_metrics['status']}\n"
        f"- Likelihood identity error: {identity_error:.6g}\n"
        f"- Current-likelihood posterior available: no\n"
        f"- Historical posterior status: {posterior['status']}\n"
        f"- Image-cube normalized RMSE (kinUV): {benchmark['metrics']['kinuv']['normalized_rmse']:.6f}\n"
        f"- Image-cube normalized RMSE (KinMS): {benchmark['metrics']['kinms']['normalized_rmse']:.6f}\n"
    )
    state["summary"] = summary
    state["state_transitions"].append({"state": "COMPLETED", "at": _utc()})
    state["state_transitions"].append(
        {"state": terminal_status.upper(), "at": _utc()}
    )
    (output / "manifest.json").write_text(json.dumps(state, indent=2) + "\n")
    _write_checksums(output)
    if blocking:
        raise RuntimeError(f"blocking gates failed: {blocking}")
    print(json.dumps({"target": config["target_id"], "status": terminal_status, "output": str(output), "selected_model": selected, "chi2": identity}, indent=2), flush=True)
    return 0 if terminal_status == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
