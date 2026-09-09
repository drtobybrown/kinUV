#!/usr/bin/env python3
"""Run one MAP start with the common 14D visibility-space model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
from time import perf_counter, monotonic

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import representation_model as legacy_context
from kinuv.infer.unified import (
    UNIFIED_PARAMETER_NAMES,
    UnifiedChartSpec,
    build_support_from_template,
    build_unified_log_density,
    decode_unified_chart,
    initial_unified_chart,
)
from kinuv.profiles.unified import projected_velocity, velocity_dispersion


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
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


def build_problem(target: str):
    """Adapt the accepted checkpoint only as a target-neutral initial condition."""
    workspace = Path(
        os.environ.get(
            "KINUV_WORKSPACE", "/arc/projects/KILOGAS/analysis/toby_sandbox"
        )
    )
    # Executable code is the frozen /scratch archive; immutable scientific
    # inputs remain under the durable workspace.
    legacy_context.REPO = workspace / "kinUV"
    legacy_context.PROJECT = workspace
    legacy_context.RECOVERY = (
        workspace
        / "results/validation/crossdomain-recovery-s4-remediation-20260907-r1"
    )
    legacy_context.COVARIANCE_RECORD = (
        workspace / "results/validation/crossdomain-recovery-s2-20260907-r1/metrics.json"
    )
    parent = legacy_context.build_target_context(target, "pspline")
    parameters = parent.parent_parameters
    pa_rad = np.radians(float(parameters["pa_deg"]))
    inclination_deg = float(parameters["inclination_deg"])
    support = build_support_from_template(
        parent.template,
        parent.grid,
        pa_rad=pa_rad,
        i_rad=np.radians(inclination_deg),
        bmaj_arcsec=parent.design.bmaj_arcsec,
    )
    u_reference = max(
        legacy_context._parent_projected_speed(
            support.reference_radius_arcsec, parameters, parent
        ),
        1.0,
    )
    sigma_reference = np.sqrt(
        float(parameters["sigma_inner_kms"])
        * float(parameters["sigma_outer_kms"])
    )
    spec = UnifiedChartSpec(
        support=support,
        pa_reference_rad=pa_rad,
        vsys_reference_kms=float(parameters["vsys_kms"]),
        dv_kms=float(parent.data.dv_kms),
        flux_reference=float(parameters["flux"]),
        u_reference_kms=float(u_reference),
        sigma_reference_kms=float(sigma_reference),
    )
    initial = initial_unified_chart(spec, inclination_deg=inclination_deg)
    density = build_unified_log_density(
        parent.data,
        parent.template,
        parent.grid,
        parent.covariance,
        spec,
    )
    provenance = {
        **parent.provenance,
        "legacy_checkpoint_role": "initialization_only",
        "model_schema": spec.metadata(),
    }
    return parent, spec, density, np.asarray(initial, dtype=np.float64), provenance


def start_vector(initial: np.ndarray, start_id: int, dv_kms: float) -> np.ndarray:
    """Four deterministic starts applied identically to every target."""
    value = np.asarray(initial, dtype=np.float64).copy()
    if start_id == 1:
        return value
    if start_id == 2:
        value[6] += np.log(1.15)
        value[7:11] += np.array([0.15, -0.10, 0.05, 0.0])
        return value
    if start_id == 3:
        value[5] += 0.55
        value[7:11] += np.array([-0.10, 0.18, -0.12, 0.06])
        value[12:14] += np.array([0.20, -0.20])
        return value
    if start_id == 4:
        value[2] -= 10.0 / float(dv_kms)
        value[3:5] += np.array([0.10, -0.10])
        return value
    raise ValueError("start_id must be in 1..4")


def physical_json(physical) -> dict:
    output = {}
    for key, value in physical.items():
        if key == "i_rad":
            output["inclination_deg"] = float(np.degrees(value))
        else:
            array = np.asarray(value)
            output[key] = float(array) if array.ndim == 0 else array.tolist()
    return output


def profile_record(optimum, spec: UnifiedChartSpec) -> dict:
    physical = decode_unified_chart(optimum, spec)
    radius = np.linspace(0.0, spec.support.outer_radius_arcsec, 256)
    u = np.asarray(
        projected_velocity(
            radius,
            optimum[6],
            optimum[7:11],
            spec.support,
        ),
        dtype=np.float64,
    )
    sin_i = max(float(np.sin(physical["i_rad"])), 1.0e-8)
    sigma = np.asarray(
        velocity_dispersion(radius, optimum[11], optimum[12:14], spec.support),
        dtype=np.float64,
    )
    return {
        "schema_version": "kinuv-unified-map-profile-v1",
        "radius_arcsec": radius.tolist(),
        "u_projected_kms": u.tolist(),
        "v_rot_kms": (u / sin_i).tolist(),
        "sigma_kms": sigma.tolist(),
        "R50": None,
        "flags": {
            "R50_status": "UNRESOLVED_PLATEAU",
            "R50_reason": (
                "MAP-only profile does not independently establish a constrained outer plateau"
            ),
            "outer_constant_extension_supplies_evidence": False,
        },
    }


def run(args) -> int:
    # The headless wrapper extracts this exact commit with ``git archive``.
    # An archive intentionally has no .git directory, so provenance is the
    # immutable commit supplied by the clean-checkout launcher.
    state = {"commit": args.code_commit, "archive_snapshot": True, "dirty": False}
    started = perf_counter()
    heartbeat = {
        "schema_version": "kinuv-unified-map-heartbeat-v1",
        "state": "STARTING",
        "target_id": args.target,
        "start_id": args.start_id,
        "session_id": os.environ.get(
            "SKAHA_SESSION_ID", os.environ.get("skaha_sessionid", platform.node())
        ),
        "pid": os.getpid(),
        "git": state,
        "updated_utc": utc_now(),
    }
    atomic_json(args.heartbeat, heartbeat)
    try:
        setup_started = perf_counter()
        context, spec, density, initial, provenance = build_problem(args.target)
        setup_s = perf_counter() - setup_started
        start = start_vector(initial, args.start_id, spec.dv_kms)
        n_complex = int(context.data.vis.size)

        objective = jax.jit(jax.value_and_grad(lambda z: -density.log_posterior(z)))
        compile_started = perf_counter()
        compiled_value, compiled_gradient = objective(jnp.asarray(start))
        jax.block_until_ready(compiled_gradient)
        compile_s = perf_counter() - compile_started

        last = {
            "value": float(compiled_value) / n_complex,
            "gradient": float(np.max(np.abs(np.asarray(compiled_gradient)))) / n_complex,
        }
        calls = 0
        iterations = 0
        last_heartbeat = monotonic()

        def normalized_objective(z):
            nonlocal calls
            value, gradient = objective(jnp.asarray(z))
            calls += 1
            last["value"] = float(value) / n_complex
            last["gradient"] = float(np.max(np.abs(np.asarray(gradient)))) / n_complex
            return last["value"], np.asarray(gradient, dtype=np.float64) / n_complex

        def callback(_):
            nonlocal iterations, last_heartbeat
            iterations += 1
            if monotonic() - last_heartbeat >= 55.0:
                heartbeat.update(
                    {
                        "state": "OPTIMIZING",
                        "iteration": iterations,
                        "objective_per_complex": last["value"],
                        "gradient_inf_per_complex": last["gradient"],
                        "updated_utc": utc_now(),
                    }
                )
                atomic_json(args.heartbeat, heartbeat)
                last_heartbeat = monotonic()

        optimize_started = perf_counter()
        fit = minimize(
            normalized_objective,
            start,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
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
        final_value = float(final_value)
        final_gradient = np.asarray(final_gradient, dtype=np.float64)
        gradient = float(np.max(np.abs(final_gradient)) / n_complex)
        chi2 = float(-2.0 * density.log_likelihood(jnp.asarray(optimum)))
        log_prior = float(density.log_prior(jnp.asarray(optimum)))
        physical = physical_json(decode_unified_chart(optimum, spec))
        cos_fraction = (
            float(physical["cos_inclination"]) - spec.cos_inclination_min
        ) / (spec.cos_inclination_max - spec.cos_inclination_min)
        rotation_pressure = float(np.max(np.abs(optimum[7:11])))
        dispersion_pressure = float(np.max(np.abs(optimum[12:14])))
        gates = {
            "optimizer_success": bool(fit.success),
            "finite_objective_gradient": bool(
                np.isfinite(final_value) and np.all(np.isfinite(final_gradient))
            ),
            "gradient_inf_per_complex_le_1e-3": gradient <= args.gradient_gate,
            "inclination_not_at_transform_edge": 0.01 <= cos_fraction <= 0.99,
            "rotation_coefficients_not_runaway": rotation_pressure <= 5.0,
            "dispersion_coefficients_not_runaway": dispersion_pressure <= 5.0,
        }
        gate_pass = all(gates.values())
        result = {
            "schema_version": "kinuv-unified-map-start-v1",
            "state": "GATE_PASS" if gate_pass else "GATE_FAILED",
            "created_utc": utc_now(),
            "target_id": args.target,
            "start_id": args.start_id,
            "git": state,
            "parameter_names": list(UNIFIED_PARAMETER_NAMES),
            "initial_z": initial.tolist(),
            "start_z": start.tolist(),
            "optimum_z": optimum.tolist(),
            "physical": physical,
            "profile": profile_record(optimum, spec),
            "support": spec.support.metadata(),
            "objective": {
                "negative_log_posterior": final_value,
                "visibility_chi2": chi2,
                "log_prior": log_prior,
                "gradient_inf_per_complex": gradient,
                "n_complex": n_complex,
            },
            "optimizer": {
                "success": bool(fit.success),
                "message": str(fit.message),
                "iterations": int(fit.nit),
                "function_calls": int(fit.nfev),
                "recorded_calls": calls,
                "maxiter": args.maxiter,
            },
            "conditioning": {
                "cos_inclination_fraction": cos_fraction,
                "max_abs_rotation_coefficient": rotation_pressure,
                "max_abs_dispersion_coefficient": dispersion_pressure,
            },
            "gates": gates,
            "gate_pass": gate_pass,
            "timing": {
                "setup_s": setup_s,
                "compile_s": compile_s,
                "optimize_s": optimize_s,
                "total_s": perf_counter() - started,
            },
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
            "provenance": provenance,
        }
        atomic_json(args.output / "result.json", result)
        heartbeat.update(
            {
                "state": result["state"],
                "iteration": int(fit.nit),
                "objective_per_complex": final_value / n_complex,
                "gradient_inf_per_complex": gradient,
                "updated_utc": utc_now(),
            }
        )
        atomic_json(args.heartbeat, heartbeat)
        print(
            json.dumps(
                {
                    "target": args.target,
                    "start": args.start_id,
                    "state": result["state"],
                    "chi2": chi2,
                    "gradient": gradient,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        heartbeat.update(
            {
                "state": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "updated_utc": utc_now(),
            }
        )
        atomic_json(args.heartbeat, heartbeat)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--start-id", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--gradient-gate", type=float, default=1.0e-3)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
