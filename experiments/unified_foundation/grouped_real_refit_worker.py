#!/usr/bin/env python3
"""Refit unified MAP on one grouped training split and score its held-out fold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from kinuv.diagnostics.comparator import load_intrinsic_kinms_cube
from kinuv.forward.operators import sample_intrinsic_cube_binned
from kinuv.infer.s2 import correlated_chi2
from kinuv.infer.unified import (
    UNIFIED_PARAMETER_NAMES,
    build_unified_log_density,
    decode_unified_chart,
)
from kinuv.io.vis import load_target_vis, load_visibility_table
from kinuv.validation.groups import build_grouped_visibility_folds

import grouped_real_visibility_benchmark as fixed_benchmark
import unified_map_runner


TARGETS = ("KGAS066", "KGAS007")


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


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_winner(path: Path, target: str) -> dict:
    """Validate the immutable MAP payload without requiring a scratch Git checkout."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "kinuv-unified-map-start-v1":
        raise ValueError(f"unsupported winner schema in {path}")
    if result.get("target_id") != target:
        raise ValueError(f"winner target mismatch: {path}")
    if result.get("gate_pass") is not True or result.get("state") != "GATE_PASS":
        raise ValueError(f"winner is not gate-passing: {path}")
    if tuple(result.get("parameter_names", ())) != UNIFIED_PARAMETER_NAMES:
        raise ValueError(f"winner parameter chart differs from worker chart: {path}")
    optimum = np.asarray(result["optimum_z"], dtype=np.float64)
    if optimum.shape != (len(UNIFIED_PARAMETER_NAMES),) or not np.all(np.isfinite(optimum)):
        raise ValueError(f"invalid optimum_z in {path}")
    return result


def run(args) -> dict:
    started = perf_counter()
    winner = validate_winner(args.winner, args.target)
    context, spec, _, initial, provenance = unified_map_runner.build_problem(args.target)
    full_map = np.asarray(winner["optimum_z"], dtype=np.float64)
    if spec.support.metadata() != winner["support"]:
        raise RuntimeError("rebuilt support differs from winner")
    if not np.allclose(initial, winner["initial_z"], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("rebuilt initialization differs from winner")

    config_path = Path(provenance["config_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    table = load_visibility_table(config["visibility_npz"])
    folds = build_grouped_visibility_folds(
        table,
        n_folds=args.n_folds,
        integrations_per_group=args.integrations_per_group,
    )
    train_mask = folds.training_mask(args.fold_id, embargo_s=0.0)
    heldout_mask = folds.validation_mask(args.fold_id)
    if np.any(train_mask & heldout_mask):
        raise RuntimeError("training and held-out masks overlap")
    excluded_boundary_mask = ~(train_mask | heldout_mask)
    train, train_load = load_target_vis(
        table,
        cube_path=config["fit_window_cube"],
        phase_dir_rad=context.data.phase_dir_rad,
        row_mask=train_mask,
    )
    heldout, heldout_load = load_target_vis(
        table,
        cube_path=config["fit_window_cube"],
        phase_dir_rad=context.data.phase_dir_rad,
        row_mask=heldout_mask,
    )

    density = build_unified_log_density(
        train, context.template, context.grid, context.covariance, spec
    )
    n_train_complex = int(np.sum(train.weights > 0.0))
    objective = jax.jit(jax.value_and_grad(lambda z: -density.log_posterior(z)))
    compile_started = perf_counter()
    compiled_value, compiled_gradient = objective(jnp.asarray(full_map))
    jax.block_until_ready(compiled_gradient)
    compile_s = perf_counter() - compile_started
    calls = 0

    def normalized_objective(z):
        nonlocal calls
        value, gradient = objective(jnp.asarray(z))
        calls += 1
        return (
            float(value) / n_train_complex,
            np.asarray(gradient, dtype=np.float64) / n_train_complex,
        )

    optimize_started = perf_counter()
    fit = minimize(
        normalized_objective,
        full_map,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": args.maxiter,
            "maxls": 40,
            "ftol": 1.0e-11,
            "gtol": args.gradient_gate,
            "maxcor": 15,
        },
    )
    optimize_s = perf_counter() - optimize_started
    optimum = np.asarray(fit.x, dtype=np.float64)
    final_value, final_gradient = objective(jnp.asarray(optimum))
    final_gradient = np.asarray(final_gradient, dtype=np.float64)
    gradient = float(np.max(np.abs(final_gradient)) / n_train_complex)
    train_chi2 = float(-2.0 * density.log_likelihood(jnp.asarray(optimum)))
    initial_train_chi2 = float(-2.0 * density.log_likelihood(jnp.asarray(full_map)))
    physical = unified_map_runner.physical_json(decode_unified_chart(optimum, spec))
    cos_fraction = (
        float(physical["cos_inclination"]) - spec.cos_inclination_min
    ) / (spec.cos_inclination_max - spec.cos_inclination_min)
    gates = {
        "optimizer_success": bool(fit.success),
        "finite_objective_gradient": bool(
            np.isfinite(final_value) and np.all(np.isfinite(final_gradient))
        ),
        "gradient_inf_per_complex_le_gate": gradient <= args.gradient_gate,
        "inclination_not_at_transform_edge": 0.01 <= cos_fraction <= 0.99,
        "rotation_coefficients_not_runaway": float(np.max(np.abs(optimum[7:11])))
        <= 5.0,
        "dispersion_coefficients_not_runaway": float(np.max(np.abs(optimum[12:14])))
        <= 5.0,
    }

    unified_cube = fixed_benchmark.unified_intrinsic_cube(context, spec, optimum)
    unified_model = np.asarray(
        sample_intrinsic_cube_binned(heldout, unified_cube, context.grid)
    )
    kinms_path = args.kinms_root / args.target / "kinms_intrinsic.npz"
    kinms_cube, kinms_metadata = load_intrinsic_kinms_cube(
        kinms_path,
        grid=context.grid,
        velocity_centers_kms=heldout.vel_native,
    )
    kinms_model = np.asarray(
        sample_intrinsic_cube_binned(
            heldout,
            kinms_cube,
            context.grid,
            spatial_assignment="cubic_b_spline",
        )
    )
    heldout_unified_chi2 = float(
        correlated_chi2(
            heldout.vis, unified_model, heldout.weights, context.covariance
        )
    )
    heldout_kinms_chi2 = float(
        correlated_chi2(
            heldout.vis, kinms_model, heldout.weights, context.covariance
        )
    )
    good = (
        np.isfinite(heldout.vis.real)
        & np.isfinite(heldout.vis.imag)
        & (heldout.weights > 0.0)
    )
    n_heldout_complex = int(np.sum(good))
    n_component = 2 * n_heldout_complex
    delta = heldout_kinms_chi2 - heldout_unified_chi2
    return {
        "schema_version": "kinuv-unified-real-grouped-refit-fold-v1",
        "created_utc": utc_now(),
        "state": "MEASURED" if all(gates.values()) else "OPTIMIZER_GATE_FAILED",
        "target_id": args.target,
        "fold_id": args.fold_id,
        "git": {
            "code_commit": args.code_commit,
            "worker_source_sha256": args.source_sha256,
            "winner_model_commit": winner["git"]["commit"],
        },
        "winner": file_record(args.winner),
        "grouping": {
            "algorithm": "build_grouped_visibility_folds",
            "n_folds": folds.n_folds,
            "integrations_per_group": args.integrations_per_group,
            "n_groups": len(folds.groups),
            "training_native_rows": int(np.sum(train_mask)),
            "heldout_native_rows": int(np.sum(heldout_mask)),
            "boundary_embargo_native_rows": int(np.sum(excluded_boundary_mask)),
            "heldout_groups": int(
                sum(group.fold_id == args.fold_id for group in folds.groups)
            ),
            "assignment_before_aggregation": True,
        },
        "training": {
            "start": "selected full-data unified MAP",
            "initial_visibility_chi2": initial_train_chi2,
            "final_visibility_chi2": train_chi2,
            "delta_chi2": initial_train_chi2 - train_chi2,
            "negative_log_posterior": float(final_value),
            "gradient_inf_per_complex": gradient,
            "gradient_gate": args.gradient_gate,
            "optimizer_success": bool(fit.success),
            "optimizer_message": str(fit.message),
            "iterations": int(fit.nit),
            "function_calls": int(fit.nfev),
            "recorded_calls": calls,
            "maxiter": args.maxiter,
            "gates": gates,
            "all_gates_pass": all(gates.values()),
            "optimum_z": optimum.tolist(),
            "physical": physical,
            "aggregated_rows": int(train.vis.shape[0]),
            "n_complex_cells": n_train_complex,
            "load_metadata": train_load,
        },
        "heldout": {
            "unified_chi2": heldout_unified_chi2,
            "kinms_chi2": heldout_kinms_chi2,
            "delta_chi2_kinms_minus_unified": delta,
            "delta_chi2_per_component": delta / n_component,
            "unified_relative_chi2_improvement": delta / heldout_kinms_chi2,
            "aggregated_rows": int(heldout.vis.shape[0]),
            "n_complex_cells": n_heldout_complex,
            "n_real_imag_components": n_component,
            "load_metadata": heldout_load,
        },
        "conditioning": {
            "unified_kinematics_refitted_on_training_rows": True,
            "unified_morphology": "fixed empirical morphology prepared from the full canonical cube",
            "kinms_model": "frozen full-data best fit; deliberately not refitted per fold",
            "fully_end_to_end_leakage_free": False,
            "interpretation": "held-out kinematic refit score conditioned on all-data morphology; KinMS full-data information makes its comparison conservative",
        },
        "shared_operator": {
            "primary_beam": True,
            "uv_sampling": True,
            "native_hann_then_software_bin": True,
            "kinms_contract_schema": kinms_metadata["schema_version"],
        },
        "inputs": {
            "config": file_record(config_path),
            "visibility_npz": file_record(Path(config["visibility_npz"])),
            "kinms_intrinsic": file_record(kinms_path),
        },
        "timing": {
            "compile_s": compile_s,
            "optimize_s": optimize_s,
            "total_s": perf_counter() - started,
        },
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--fold-id", required=True, type=int, choices=range(5))
    parser.add_argument("--winner", required=True, type=Path)
    parser.add_argument("--kinms-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--integrations-per-group", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--gradient-gate", type=float, default=1.0e-3)
    args = parser.parse_args()
    if sha256(Path(__file__)) != args.source_sha256:
        raise RuntimeError("worker source checksum mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    atomic_json(args.output / "result.json", result)
    print(
        json.dumps(
            {
                "target": args.target,
                "fold": args.fold_id,
                "state": result["state"],
                "delta_chi2_per_component": result["heldout"][
                    "delta_chi2_per_component"
                ],
                "gradient": result["training"]["gradient_inf_per_complex"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
