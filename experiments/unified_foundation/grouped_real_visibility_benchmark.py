#!/usr/bin/env python3
"""Score fixed unified and KinMS models on grouped real-visibility folds.

This is a conditional, partitioned evaluation of already fitted full-data
models.  It is not leakage-free cross-validation: the unified morphology and
both models' parameters condition on full-data products.  Native rows are
assigned to target-neutral groups before time/uv/channel aggregation, and both
models then pass through the same kinUV PB, uv-sampling, and channel-response
operator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

from kinuv.diagnostics.comparator import load_intrinsic_kinms_cube
from kinuv.forward.model import intrinsic_sky_cube
from kinuv.forward.operators import sample_intrinsic_cube_binned
from kinuv.infer.s2 import correlated_chi2
from kinuv.infer.unified import (
    UNIFIED_PARAMETER_NAMES,
    decode_unified_chart,
    unified_profile_callables,
)
from kinuv.io.vis import load_target_vis, load_visibility_table
from kinuv.validation.groups import build_grouped_visibility_folds

import unified_map_runner


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
TARGETS = ("KGAS066", "KGAS007")
MODEL_SOURCE_PATHS = (
    Path("src/kinuv/infer/unified.py"),
    Path("src/kinuv/profiles/unified.py"),
    Path("experiments/unified_foundation/unified_map_runner.py"),
)
DEFAULT_KINMS_ROOT = (
    PROJECT
    / "results/validation/crossdomain-recovery-s4-remediation-20260907-r2/grouped"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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


def parse_winners(values: list[str]) -> dict[str, Path]:
    winners = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--winner must be TARGET=/absolute/path/result.json")
        target, raw_path = value.split("=", 1)
        if target not in TARGETS:
            raise ValueError(f"unsupported target {target!r}")
        if target in winners:
            raise ValueError(f"duplicate winner for {target}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        winners[target] = path
    if set(winners) != set(TARGETS):
        raise ValueError(f"both targets are required; received {sorted(winners)}")
    return winners


def git_record() -> dict:
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
        "benchmark_source_sha256": sha256(Path(__file__)),
    }


def require_matching_model_source(model_commit: str) -> None:
    subprocess.run(
        ["git", "cat-file", "-e", f"{model_commit}^{{commit}}"],
        cwd=REPO,
        check=True,
    )
    changed = subprocess.run(
        ["git", "diff", "--quiet", model_commit, "--", *map(str, MODEL_SOURCE_PATHS)],
        cwd=REPO,
        check=False,
    )
    if changed.returncode != 0:
        raise RuntimeError(
            f"current unified model source differs from winner commit {model_commit}"
        )


def validate_winner(path: Path, target: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "kinuv-unified-map-start-v1":
        raise ValueError(f"unsupported winner schema in {path}")
    if result.get("target_id") != target:
        raise ValueError(f"winner target mismatch: {path}")
    if result.get("gate_pass") is not True or result.get("state") != "GATE_PASS":
        raise ValueError(f"winner is not gate-passing: {path}")
    if tuple(result.get("parameter_names", ())) != UNIFIED_PARAMETER_NAMES:
        raise ValueError(f"winner parameter chart differs from current chart: {path}")
    optimum = np.asarray(result["optimum_z"], dtype=np.float64)
    if optimum.shape != (len(UNIFIED_PARAMETER_NAMES),) or not np.all(
        np.isfinite(optimum)
    ):
        raise ValueError(f"invalid optimum_z in {path}")
    require_matching_model_source(result["git"]["commit"])
    return result


def unified_intrinsic_cube(context, spec, optimum):
    physical = decode_unified_chart(optimum, spec)
    velocity_profile, dispersion_profile = unified_profile_callables(optimum, spec)
    sin_i = max(float(np.sin(physical["i_rad"])), 1.0e-6)
    cube = intrinsic_sky_cube(
        context.template,
        context.grid,
        context.data.freqs_native,
        flux=float(physical["flux"]),
        pa_rad=float(physical["pa_rad"]),
        vsys_kms=float(physical["vsys_kms"]),
        dx_arcsec=float(physical["dx_arcsec"]),
        dy_arcsec=float(physical["dy_arcsec"]),
        gas_sigma_kms=float(physical["sigma0_kms"]),
        v0_kms=float(physical["u_reference_kms"]) / sin_i,
        r_t_arcsec=float(spec.support.outer_radius_arcsec),
        i_rad=float(physical["i_rad"]),
        velocity_profile=velocity_profile,
        dispersion_profile=dispersion_profile,
    )
    cube = np.asarray(cube, dtype=np.float64)
    if cube.shape != (
        context.grid.ny,
        context.grid.nx,
        context.data.freqs_native.size,
    ):
        raise RuntimeError(f"unexpected unified intrinsic cube shape {cube.shape}")
    if np.any(~np.isfinite(cube)):
        raise RuntimeError("unified intrinsic cube is nonfinite")
    return cube


def lower_confidence(values: np.ndarray, probability: float = 0.95) -> dict:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2 or np.any(~np.isfinite(sample)):
        raise ValueError("confidence bound requires at least two finite fold values")
    mean = float(np.mean(sample))
    standard_error = float(np.std(sample, ddof=1) / np.sqrt(sample.size))
    critical = float(student_t.ppf(probability, df=sample.size - 1))
    lower = mean - critical * standard_error
    return {
        "method": "one-sided Student-t bound across paired fold scores",
        "confidence": probability,
        "n_folds": int(sample.size),
        "mean_delta_chi2_per_component": mean,
        "standard_error": standard_error,
        "critical_value": critical,
        "lower_bound_delta_chi2_per_component": float(lower),
        "positive_lower_bound_gate": bool(lower > 0.0),
    }


def score_target(
    target: str,
    winner_path: Path,
    kinms_root: Path,
    *,
    n_folds: int,
    integrations_per_group: int,
) -> dict:
    started = perf_counter()
    winner = validate_winner(winner_path, target)
    context, spec, _, initial, context_provenance = unified_map_runner.build_problem(target)
    optimum = np.asarray(winner["optimum_z"], dtype=np.float64)
    if spec.support.metadata() != winner["support"]:
        raise RuntimeError("rebuilt unified radial support differs from winner record")
    if not np.allclose(initial, winner["initial_z"], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("rebuilt unified initialization differs from winner record")

    config_path = Path(context_provenance["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    table = load_visibility_table(config["visibility_npz"])
    folds = build_grouped_visibility_folds(
        table,
        n_folds=n_folds,
        integrations_per_group=integrations_per_group,
    )
    unified_cube = unified_intrinsic_cube(context, spec, optimum)
    positive_cube_sum = float(np.sum(np.clip(unified_cube, 0.0, None)))
    negative_cube_sum = float(np.sum(np.clip(-unified_cube, 0.0, None)))
    kinms_path = kinms_root / target / "kinms_intrinsic.npz"
    kinms_cube, kinms_metadata = load_intrinsic_kinms_cube(
        kinms_path,
        grid=context.grid,
        velocity_centers_kms=context.data.vel_native,
    )

    full_unified = np.asarray(
        sample_intrinsic_cube_binned(context.data, unified_cube, context.grid)
    )
    full_kinms = np.asarray(
        sample_intrinsic_cube_binned(
            context.data,
            kinms_cube,
            context.grid,
            spatial_assignment="cubic_b_spline",
        )
    )
    full_unified_chi2 = float(
        correlated_chi2(
            context.data.vis, full_unified, context.data.weights, context.covariance
        )
    )
    full_kinms_chi2 = float(
        correlated_chi2(
            context.data.vis, full_kinms, context.data.weights, context.covariance
        )
    )
    identity_error = abs(
        full_unified_chi2 - float(winner["objective"]["visibility_chi2"])
    )
    if identity_error > 0.05:
        raise RuntimeError(
            f"unified full-data likelihood identity error {identity_error} exceeds 0.05"
        )

    fold_records = []
    for fold_id in range(folds.n_folds):
        row_mask = folds.validation_mask(fold_id)
        heldout, load_metadata = load_target_vis(
            table,
            cube_path=config["fit_window_cube"],
            phase_dir_rad=context.data.phase_dir_rad,
            row_mask=row_mask,
        )
        unified_model = np.asarray(
            sample_intrinsic_cube_binned(heldout, unified_cube, context.grid)
        )
        kinms_model = np.asarray(
            sample_intrinsic_cube_binned(
                heldout,
                kinms_cube,
                context.grid,
                spatial_assignment="cubic_b_spline",
            )
        )
        chi2_unified = float(
            correlated_chi2(
                heldout.vis, unified_model, heldout.weights, context.covariance
            )
        )
        chi2_kinms = float(
            correlated_chi2(
                heldout.vis, kinms_model, heldout.weights, context.covariance
            )
        )
        good = (
            np.isfinite(heldout.vis.real)
            & np.isfinite(heldout.vis.imag)
            & (heldout.weights > 0.0)
        )
        n_complex = int(np.sum(good))
        n_component = 2 * n_complex
        delta = chi2_kinms - chi2_unified
        fold_records.append(
            {
                "fold_id": fold_id,
                "native_rows": int(np.sum(row_mask)),
                "groups": int(sum(group.fold_id == fold_id for group in folds.groups)),
                "aggregated_rows": int(heldout.vis.shape[0]),
                "fit_channels": int(heldout.vis.shape[1]),
                "n_complex_cells": n_complex,
                "n_real_imag_components": n_component,
                "chi2_unified": chi2_unified,
                "chi2_kinms": chi2_kinms,
                "delta_chi2_kinms_minus_unified": delta,
                "delta_chi2_per_component": delta / n_component,
                "unified_relative_chi2_improvement": delta / chi2_kinms,
                "load_metadata": load_metadata,
            }
        )

    fold_values = np.asarray(
        [row["delta_chi2_per_component"] for row in fold_records], dtype=np.float64
    )
    confidence = lower_confidence(fold_values)
    return {
        "target_id": target,
        "state": "MEASURED_CONDITIONAL_PARTITIONED_EVALUATION",
        "winner": file_record(winner_path),
        "winner_start_id": int(winner["start_id"]),
        "winner_model_commit": winner["git"]["commit"],
        "winner_visibility_chi2": float(winner["objective"]["visibility_chi2"]),
        "full_data_operator_replay": {
            "unified_chi2": full_unified_chi2,
            "kinms_chi2": full_kinms_chi2,
            "unified_identity_absolute_error": identity_error,
            "identity_tolerance": 0.05,
            "identity_pass": True,
        },
        "grouping": {
            "algorithm": "build_grouped_visibility_folds",
            "n_folds": folds.n_folds,
            "integrations_per_group": integrations_per_group,
            "n_groups": len(folds.groups),
            "assignment_before_aggregation": True,
            "antenna_pair_excluded_from_group_key": True,
        },
        "covariance": {
            "model": "C1 AR(1)",
            "scale": float(context.covariance.scale),
            "rho": float(context.covariance.rho),
            "source_scale": float(context.covariance.source_scale),
            "source_rho": float(context.covariance.source_rho),
            "software_bin": int(context.covariance.software_bin),
        },
        "folds": fold_records,
        "one_sided_lower_95": confidence,
        "conditioning": {
            "fully_end_to_end_leakage_free": False,
            "unified_morphology": "empirical positive morphology prepared from the full canonical cube; fixed during this score",
            "unified_parameters": "selected MAP fitted to all visibility rows",
            "kinms_parameters": "frozen best KinMS model fitted outside these folds",
            "interpretation": "partitioned predictive adequacy of fixed full-data models, not training-fold refits",
        },
        "shared_operator": {
            "primary_beam": "kinuv.forward.operators.attenuate_intrinsic_cube",
            "uv_sampling": "kinuv.transforms.nufft.nufft2_degrid",
            "spectral_response": "native Hann then identical software binning",
            "unified_spatial_assignment_compensation": None,
            "kinms_spatial_assignment_compensation": "cardinal cubic B-spline B3",
        },
        "unified_intrinsic_cube": {
            "shape": list(unified_cube.shape),
            "minimum": float(np.min(unified_cube)),
            "maximum": float(np.max(unified_cube)),
            "negative_absolute_fraction_of_positive_sum": (
                negative_cube_sum / positive_cube_sum
            ),
            "negative_values_origin": "established Fourier subpixel shift interpolation ringing",
        },
        "kinms_intrinsic": {
            "npz": file_record(kinms_path),
            "sidecar": file_record(kinms_path.with_suffix(".json")),
            "validated_schema": kinms_metadata["schema_version"],
            "render_mode": kinms_metadata["render_mode"],
            "clean_out": kinms_metadata["clean_out"],
            "restoring_beam_applied": kinms_metadata["restoring_beam_applied"],
            "primary_beam_applied": kinms_metadata["primary_beam_applied"],
            "spectral_response_applied": kinms_metadata["spectral_response_applied"],
        },
        "inputs": {
            "config": file_record(config_path),
            "visibility_npz": file_record(Path(config["visibility_npz"])),
            "template_ico": file_record(Path(config["template_ico"])),
            "checkpoint": file_record(Path(context_provenance["checkpoint_path"])),
            "covariance": file_record(Path(context_provenance["covariance_path"])),
        },
        "elapsed_s": perf_counter() - started,
    }


def report_markdown(summary: dict) -> str:
    lines = [
        "# Unified real-visibility grouped benchmark",
        "",
        "This is a conditional partitioned evaluation of fixed full-data models. "
        "The empirical morphology and fitted parameters are not regenerated per fold, "
        "so these results are not fully end-to-end leakage-free cross-validation.",
        "",
        "| Target | mean delta chi2/component | lower 95% | Gate | Full replay error |",
        "|---|---:|---:|---|---:|",
    ]
    for row in summary["targets"]:
        bound = row["one_sided_lower_95"]
        lines.append(
            f"| {row['target_id']} | {bound['mean_delta_chi2_per_component']:.8g} "
            f"| {bound['lower_bound_delta_chi2_per_component']:.8g} "
            f"| {'PASS' if bound['positive_lower_bound_gate'] else 'FAIL'} "
            f"| {row['full_data_operator_replay']['unified_identity_absolute_error']:.8g} |"
        )
    lines.extend(
        [
            "",
            "Positive delta means the unified kinUV model has lower held-out chi-square "
            "than the frozen intrinsic KinMS comparator after the identical observational operator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--winner",
        action="append",
        required=True,
        help="TARGET=/absolute/path/to/start-N/result.json; supply both targets",
    )
    parser.add_argument("--kinms-root", type=Path, default=DEFAULT_KINMS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--integrations-per-group", type=int, default=5)
    args = parser.parse_args()
    winners = parse_winners(args.winner)
    output = args.output.resolve()
    production = (PROJECT / "results/production").resolve()
    if output == production or production in output.parents:
        raise ValueError("output must not be inside active production")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")

    started = perf_counter()
    targets = [
        score_target(
            target,
            winners[target],
            args.kinms_root.resolve(),
            n_folds=args.n_folds,
            integrations_per_group=args.integrations_per_group,
        )
        for target in TARGETS
    ]
    summary = {
        "schema_version": "kinuv-unified-real-grouped-visibility-v1",
        "created_utc": utc_now(),
        "state": "MEASURED_CONDITIONAL_PARTITIONED_EVALUATION",
        "git": git_record(),
        "targets": targets,
        "gate": {
            "rule": "one-sided 95% lower bound of paired fold delta chi2 per real/imag component is positive for every target",
            "all_targets_pass": all(
                row["one_sided_lower_95"]["positive_lower_bound_gate"]
                for row in targets
            ),
        },
        "production_figure_rerender_inventory": {
            "required_posterior_contract": [
                "same 14D unified chart and parameter_names as the selected MAP",
                "NUTS chain/draw samples with unit weights and exact source/prior provenance",
                "primary geometry Rhat <= 1.05, ESS >= 400, divergence and tree-depth records",
                "samplewise u(R), Vrot(R), sigma(R), geometry, support, and R50 status",
                "q16/q50/q84 bands labelled by stage, sampler, and conditionality",
            ],
            "figures_to_regenerate": [
                "plots/pv_diagrams.pdf and .png: major-axis projected-speed band with every native radio/TOPO endpoint converted to optical-LSRK; minor axis keeps systemic/geometry annotation",
                "plots/rotation_curve.pdf and .png: unified u, Vrot, and sigma q16/q50/q84 bands plus missingness/support flags",
                "plots/spectral_profiles.pdf and .png only if posterior-predictive aperture spectra are supplied; otherwise retain explicit MAP label",
                "plots/posterior_corner.pdf and .png from the matching unified posterior only",
                "benchmarks counterparts when KinMS overlays are shown",
            ],
            "representative_cube_rule": (
                "moments and fixed-model spectra need rerendering only if posterior promotion "
                "changes the representative point model; never mix a posterior from another checkpoint"
            ),
            "promotion_bookkeeping": [
                "archive the superseded target tree",
                "update selection.json, posterior status, provenance, and both manifests atomically",
                "preserve frozen data/mask/KinMS hashes and label the full-data morphology conditioning",
            ],
        },
        "elapsed_s": perf_counter() - started,
    }
    output.mkdir(parents=True)
    write_json_atomic(output / "result.json", summary)
    (output / "REPORT.md").write_text(report_markdown(summary), encoding="ascii")
    manifest = {
        "schema_version": "kinuv-unified-real-grouped-visibility-manifest-v1",
        "created_utc": summary["created_utc"],
        "files": {
            path.name: file_record(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
