#!/usr/bin/env python3
"""Bounded CANFAR worker for the unified representation prototype."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from representation_model import (
    FAMILIES,
    PARAMETER_NAMES,
    build_target_context,
    log_prior_chart,
    profile_artifact,
    profiles_from_z,
    rotation_shape_features,
    value_and_grad,
    z_to_physical,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict):
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


def fit_profile(context, truth: str):
    design = context.design
    radius = np.linspace(0.0, design.outer_radius_arcsec, 256)
    parent = context.parent_parameters
    if truth == "subbeam_arctan":
        truth_scale = 0.25
        truth_amplitude = float(parent["arctan_u_kms"])
        truth_u = truth_amplitude * (2.0 / np.pi) * np.arctan(radius / truth_scale)
    else:
        from representation_model import _parent_projected_speed
        truth_u = np.array([_parent_projected_speed(r, parent, context) for r in radius])
        truth_scale = None
        truth_amplitude = None
    reference_truth = float(np.interp(design.reference_radius_arcsec, radius, truth_u))

    def predict(q):
        amplitude = np.exp(q[0])
        scale = design.bmaj_arcsec * np.exp(q[1])
        carrier = np.arctan(radius / scale) / np.arctan(design.reference_radius_arcsec / scale)
        feature = np.asarray(rotation_shape_features(radius, design, context.family))
        return amplitude * carrier * np.exp(feature @ q[2:])

    initial = np.r_[np.log(max(reference_truth, 1.0)), np.log(0.4), np.zeros(4)]
    start = perf_counter()
    result = least_squares(
        lambda q: (predict(q) - truth_u) / 100.0,
        initial,
        max_nfev=100,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    elapsed = perf_counter() - start
    fitted = predict(result.x)
    inner = radius <= design.bmaj_arcsec
    derivative = np.gradient(fitted, radius, edge_order=2)
    return {
        "truth": truth,
        "truth_scale_arcsec": truth_scale,
        "truth_amplitude_kms": truth_amplitude,
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "elapsed_s": elapsed,
        "rmse_full_kms": float(np.sqrt(np.mean((fitted - truth_u) ** 2))),
        "rmse_inner_beam_kms": float(np.sqrt(np.mean((fitted[inner] - truth_u[inner]) ** 2))),
        "max_abs_error_kms": float(np.max(np.abs(fitted - truth_u))),
        "u_zero_kms": float(fitted[0]),
        "central_du_dR_kms_per_arcsec": float(derivative[0]),
        "fit": {
            "u_reference_kms": float(np.exp(result.x[0])),
            "scale_arcsec": float(design.bmaj_arcsec * np.exp(result.x[1])),
            "shape_latents": result.x[2:].tolist(),
        },
    }


def fit_dispersion_profile(context):
    """Quantify whether the common two-deviation log spline retains sigma shape."""
    from representation_model import dispersion_shape_features

    design = context.design
    radius = np.linspace(0.0, design.outer_radius_arcsec, 256)
    parent = context.parent_parameters
    sigma_inner = float(parent["sigma_inner_kms"])
    sigma_outer = float(parent["sigma_outer_kms"])
    if context.provenance["parent_candidate"] == "two_zone_dispersion":
        transition = float(context.provenance["parent_knot_radii_arcsec"][1])
        width = 0.25 * design.bmaj_arcsec
        outer_fraction = 0.5 * (1.0 + np.tanh((radius - transition) / width))
        truth_sigma = sigma_inner * (1.0 - outer_fraction) + sigma_outer * outer_fraction
    else:
        transition = None
        width = None
        truth_sigma = np.full_like(radius, sigma_inner)
    feature = np.asarray(dispersion_shape_features(radius, design))

    def residual(q):
        return np.log(np.exp(q[0] + feature @ q[1:])) - np.log(truth_sigma)

    result = least_squares(residual, np.r_[np.log(np.sqrt(sigma_inner*sigma_outer)), np.zeros(2)], max_nfev=100)
    fitted = np.exp(result.x[0] + feature @ result.x[1:])
    return {
        "truth": "accepted_parent_dispersion",
        "transition_arcsec": transition,
        "transition_width_arcsec": width,
        "constant_truth": bool(np.allclose(truth_sigma, truth_sigma[0])),
        "success": bool(result.success),
        "nfev": int(result.nfev),
        "rmse_kms": float(np.sqrt(np.mean((fitted-truth_sigma)**2))),
        "max_abs_error_kms": float(np.max(np.abs(fitted-truth_sigma))),
        "fit_sigma0_kms": float(np.exp(result.x[0])),
        "fit_deviation_latents": result.x[1:].tolist(),
        "zero_deviations_exactly_constant": True,
    }


def continuity_metrics(context):
    design = context.design
    z = context.initial_z
    points = list(np.unique(design.pspline_knots_arcsec)) if context.family == "pspline" else []
    points.append(design.outer_radius_arcsec)
    points = sorted(set(point for point in points if point > 0.0))
    jumps = []
    h = max(1.0e-5, 1.0e-4 * design.bmaj_arcsec)
    for point in points:
        r = np.array([max(0.0, point - 2*h), max(0.0, point-h), point, point+h, point+2*h])
        u = np.asarray(profiles_from_z(r, z, context)[0])
        left = (u[2] - u[1]) / h
        right = (u[3] - u[2]) / h
        jumps.append({"radius_arcsec": point, "du_dR_jump_kms_per_arcsec": float(right-left)})
    tiny = design.bmaj_arcsec * 1.0e-7
    u = np.asarray(profiles_from_z(np.array([0.0, tiny]), z, context)[0])
    return {
        "u_at_zero_kms": float(u[0]),
        "finite_u_over_r_at_zero_limit": bool(np.isfinite(u[1] / tiny)),
        "u_over_r_near_zero_kms_per_arcsec": float(u[1] / tiny),
        "maximum_numeric_first_derivative_jump_kms_per_arcsec": float(max((abs(row["du_dR_jump_kms_per_arcsec"]) for row in jumps), default=0.0)),
        "locations": jumps,
        "analytic_continuity": "C2 internal; P-spline C1 at constant-extension endpoint; finite GP C-infinity",
    }


def run_case(target, family, maxiter):
    context = build_target_context(target, family)
    z0 = np.asarray(context.initial_z, dtype=np.float64)
    fn = value_and_grad(context)
    compile_start = perf_counter()
    value0, grad0 = fn(jnp.asarray(z0))
    jax.block_until_ready(value0)
    compile_s = perf_counter() - compile_start
    warmed = []
    for _ in range(5):
        start = perf_counter()
        value, gradient = fn(jnp.asarray(z0))
        jax.block_until_ready(value)
        warmed.append(perf_counter() - start)

    n_complex = int(context.data.vis.size)
    def objective(q):
        value, gradient = fn(jnp.asarray(q))
        return float(value) / n_complex, np.asarray(gradient, dtype=float) / n_complex

    fit_start = perf_counter()
    result = minimize(
        objective,
        z0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "maxls": 20, "ftol": 1.0e-12, "gtol": 1.0e-7},
    )
    fit_s = perf_counter() - fit_start
    value_final, gradient_final = fn(jnp.asarray(result.x))
    jax.block_until_ready(value_final)
    prior0 = float(log_prior_chart(jnp.asarray(z0), context))
    prior_final = float(log_prior_chart(jnp.asarray(result.x), context))
    chi2_initial = 2.0 * (float(value0) + prior0)
    chi2_final = 2.0 * (float(value_final) + prior_final)

    radius = np.linspace(0.0, context.design.outer_radius_arcsec, 512)
    profile_start = perf_counter()
    for _ in range(100):
        profiles_from_z(radius, result.x, context)
    profile_eval_us = (perf_counter() - profile_start) * 1.0e4
    design = context.design
    physical = z_to_physical(result.x, context)
    return {
        "schema_version": "kinuv-unified-representation-pilot-v1",
        "created_utc": now(),
        "target_id": target,
        "family": family,
        "prototype_only": True,
        "conditionality": "short local MAP profile from accepted S4 morphology; not converged inference",
        "parameter_names": list(PARAMETER_NAMES),
        "dimension": len(PARAMETER_NAMES),
        "c1_covariance": context.covariance.__dict__,
        "data_shape": list(context.data.vis.shape),
        "n_complex": n_complex,
        "radial_design": {
            "bmaj_arcsec": design.bmaj_arcsec,
            "cell_arcsec": design.cell_arcsec,
            "r20_arcsec": design.r20_arcsec,
            "r50_emission_arcsec": design.r50_emission_arcsec,
            "r70_arcsec": design.r70_arcsec,
            "r95_arcsec": design.r95_arcsec,
            "reference_radius_arcsec": design.reference_radius_arcsec,
            "outer_radius_arcsec": design.outer_radius_arcsec,
            "pspline_knots_arcsec": design.pspline_knots_arcsec.tolist(),
            "gp_centres_arcsec": design.gp_centres_arcsec.tolist(),
            "gp_length_arcsec": design.gp_length_arcsec,
            "beam_is_scale_not_cutoff": True,
        },
        "timing": {
            "compile_plus_first_value_grad_s": compile_s,
            "warmed_value_grad_s": warmed,
            "warmed_value_grad_median_s": float(np.median(warmed)),
            "profile_512_radius_eval_us": profile_eval_us,
            "optimizer_elapsed_s": fit_s,
        },
        "short_map": {
            "maxiter": int(maxiter),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
            "chi2_initial": chi2_initial,
            "chi2_final": chi2_final,
            "delta_chi2": chi2_initial - chi2_final,
            "gradient_inf_per_complex": float(np.max(np.abs(np.asarray(gradient_final))) / n_complex),
            "explicitly_not_converged_campaign": True,
        },
        "physical_at_short_map": {
            key: float(value) if np.ndim(value) == 0 else np.asarray(value).tolist()
            for key, value in physical.items() if key != "i_rad"
        },
        "continuity": continuity_metrics(context),
        "profile_fits": [fit_profile(context, "accepted_parent"), fit_profile(context, "subbeam_arctan")],
        "dispersion_profile_fit": fit_dispersion_profile(context),
        "profile": profile_artifact(result.x, context),
        "provenance": context.provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=3)
    parser.add_argument("--family", choices=FAMILIES)
    args = parser.parse_args()
    status = {
        "schema_version": "kinuv-unified-representation-worker-v1",
        "target_id": args.target,
        "started_utc": now(),
        "state": "RUNNING",
        "pid": os.getpid(),
        "host": platform.node(),
        "jax_backend": jax.default_backend(),
        "cases": [],
    }
    target_root = args.output_root / args.target
    write_json(target_root / "status.json", status)
    try:
        for family in ((args.family,) if args.family else FAMILIES):
            case = run_case(args.target, family, args.maxiter)
            write_json(target_root / f"{family}.json", case)
            status["cases"].append({"family": family, "state": "SUCCEEDED", "result": str(target_root / f"{family}.json")})
            status["heartbeat_utc"] = now()
            write_json(target_root / "status.json", status)
        status["state"] = "SUCCEEDED"
        status["finished_utc"] = now()
        write_json(target_root / "status.json", status)
        return 0
    except Exception as exc:
        status["state"] = "FAILED"
        status["finished_utc"] = now()
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(target_root / "status.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
