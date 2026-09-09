#!/usr/bin/env python3
"""Bounded sampler spike on the frozen collaborator visibility posterior.

This is an experiment, not a production runner.  It keeps the visibility
likelihood, C1 covariance, bounds, and prior measure used by the collaborator
campaign, while exposing likelihood-only and posterior-gradient timings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm


REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
MAP_ROOT = WORKSPACE / "results/incoming/collaborator-delivery-20260908/map"


def now():
    return datetime.now(timezone.utc).isoformat()


def rss_mib():
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measured(callable_, repeats):
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        try:
            result.block_until_ready()
        except AttributeError:
            if isinstance(result, tuple):
                for item in result:
                    try:
                        item.block_until_ready()
                    except AttributeError:
                        pass
        values.append(time.perf_counter() - start)
    return {
        "repeats": repeats,
        "seconds": values,
        "median_s": float(np.median(values)),
        "mean_s": float(np.mean(values)),
    }


def finite_difference_curvature(value_gradient, point, relative_step=1.0e-2):
    """Symmetric finite difference of the already-compiled gradient."""
    point = np.asarray(point, dtype=np.float64)
    hessian = np.empty((point.size, point.size), dtype=np.float64)
    for index in range(point.size):
        step = relative_step * max(1.0, abs(float(point[index])))
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        _, gp = value_gradient(plus)
        _, gm = value_gradient(minus)
        hessian[:, index] = (np.asarray(gp) - np.asarray(gm)) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


def regularized_covariance(hessian, eigen_floor=1.0e-6):
    """Return C ~= H^-1, flagging rather than hiding negative curvature."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(hessian))
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).eps)
    floor = scale * eigen_floor
    regularized = np.maximum(eigenvalues, floor)
    covariance = (eigenvectors * (1.0 / regularized)) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {
        "raw_eigenvalues": eigenvalues.tolist(),
        "negative_eigenvalues": int(np.sum(eigenvalues < -floor)),
        "nonpositive_eigenvalues": int(np.sum(eigenvalues <= 0.0)),
        "eigen_floor": floor,
        "regularized_condition": float(np.max(regularized) / np.min(regularized)),
        "policy": "flag negative curvature; floor all eigenvalues below max_abs*1e-6; never abs-fold",
    }


def build_problem(target, *, require_separable_prior=False):
    # Reuse the production forward model/objective.  No duplicate Fourier path.
    from run_collaborator_nuts_chain import setup_problem

    selected_path = MAP_ROOT / target / "selected_map.json"
    selected_doc, transform, fixed_full, potential = setup_problem(selected_path)
    selected = selected_doc["selected"]
    if selected["emissivity_weights"] is None:
        raise RuntimeError("baseline spike requires frozen emissivity weights")
    uses_rings = selected["selected_parent"] == "supported_rings" or (
        selected["selected_parent"] == "two_zone_dispersion"
        and selected["two_zone_uses_rings"]
    )
    if uses_rings and require_separable_prior:
        raise RuntimeError("exact prior transform for coupled ring regularization is not implemented")
    config = json.loads((REPO / "configs/targets" / f"{target}.json").read_text())
    bmaj = float(config["diagnostic_beam"]["bmaj_arcsec"])
    y_map = transform.active_initial_unconstrained(fixed_full)
    bounds = transform.solver_bounds()

    def prior_transform(unit):
        unit = np.clip(np.asarray(unit, dtype=float), 1e-12, 1.0 - 1e-12)
        solver = np.asarray(transform.full_to_bounded(fixed_full), dtype=float)
        for position, index in enumerate(transform.active):
            lo, hi = map(float, bounds[index])
            if index in (4, 5):
                sigma = 0.5 / bmaj
                solver[index] = truncnorm.ppf(unit[position], lo / sigma, hi / sigma, scale=sigma)
            elif index in transform.physical_log_indices:
                physical = np.exp(lo) + unit[position] * (np.exp(hi) - np.exp(lo))
                solver[index] = np.log(physical)
            else:
                solver[index] = lo + unit[position] * (hi - lo)
        return np.asarray(transform.bounded_to_full_numpy(solver))[list(transform.active)]

    def active_to_full(active):
        full = np.asarray(fixed_full, dtype=float).copy()
        full[list(transform.active)] = np.asarray(active, dtype=float)
        return full

    return {
        "selected_path": selected_path,
        "selected": selected,
        "transform": transform,
        "fixed_full": fixed_full,
        "potential": potential,
        "y_map": y_map,
        "prior_transform": prior_transform,
        "active_to_full": active_to_full,
        "bmaj": bmaj,
        "uses_rings": uses_rings,
    }


def run(args):
    started = now()
    wall0 = time.perf_counter()
    rss0 = rss_mib()
    setup0 = time.perf_counter()
    problem = build_problem(
        args.target, require_separable_prior=args.sampler == "dynesty"
    )
    setup_s = time.perf_counter() - setup0

    import jax
    import jax.numpy as jnp

    potential = jax.jit(problem["potential"])
    value_gradient = jax.jit(jax.value_and_grad(problem["potential"]))
    y_map = jnp.asarray(problem["y_map"])
    compile_value = measured(lambda: potential(y_map), 1)
    compile_value_grad = measured(lambda: value_gradient(y_map), 1)
    warmed_value = measured(lambda: potential(y_map), args.profile_repeats)
    warmed_value_grad = measured(lambda: value_gradient(y_map), args.profile_repeats)

    curvature0 = time.perf_counter()
    hessian = finite_difference_curvature(
        lambda value: value_gradient(jnp.asarray(value)), np.asarray(y_map)
    )
    covariance, metric = regularized_covariance(hessian)
    curvature_s = time.perf_counter() - curvature0
    chol = np.linalg.cholesky(covariance)
    logdet_chol = float(np.sum(np.log(np.diag(chol))))

    result = {
        "schema_version": "kinuv-unified-sampler-spike-v1",
        "state": "BENCHMARK_INCOMPLETE",
        "posterior_fidelity": "unresolved",
        "created_utc": started,
        "target_id": args.target,
        "representation": "collaborator-10/11D-baseline; not unified representation",
        "sampler": args.sampler,
        "parameter_names": list(problem["transform"].names),
        "ndim": len(problem["y_map"]),
        "prior": (
            "same bounded physical measure; independent truncated N(0,0.5 arcsec) center prior"
            + ("; coupled bounded ring smoothness prior" if problem["uses_rings"] else "")
        ),
        "likelihood": "sole fixed-C1 correlated complex-visibility log likelihood",
        "affine_chart": "y = y_MAP + L z, L L^T = regularized H_y^-1; constant log|L| kept separate",
        "nonlinear_chart": "CampaignTransform bounded logit/log maps and their nonconstant Jacobian",
        "metric_api": "NumPyro inverse_mass_matrix is M^-1 in K=p^T M^-1 p/2; C=H^-1 is correct in y, identity after whitening",
        "inputs": {
            "selected_map": str(problem["selected_path"]),
            "selected_map_sha256": sha256(problem["selected_path"]),
            "visibility": problem["selected"]["inputs"]["visibility"],
            "covariance": problem["selected"]["inputs"]["covariance"],
            "model_checkpoint": problem["selected"]["inputs"]["checkpoint"],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        },
        "timing": {
            "queue_s": None,
            "setup_load_s": setup_s,
            "compile_value_s": compile_value["seconds"][0],
            "compile_value_and_grad_s": compile_value_grad["seconds"][0],
            "warmed_value": warmed_value,
            "warmed_value_and_grad": warmed_value_grad,
            "curvature_s": curvature_s,
        },
        "metric": metric,
        "constant_affine_log_abs_det_jacobian": logdet_chol,
        "memory": {"initial_max_rss_mib": rss0, "profile_max_rss_mib": rss_mib()},
    }

    if args.sampler == "profile":
        pass
    elif args.sampler == "nuts":
        from numpyro.infer import NUTS

        lmat = jnp.asarray(chol)
        y0 = y_map

        def whitened_potential(z):
            return potential(y0 + lmat @ z) - logdet_chol

        kernel = NUTS(
            potential_fn=jax.jit(whitened_potential),
            inverse_mass_matrix=jnp.eye(len(y0)),
            dense_mass=True,
            adapt_mass_matrix=args.adapt_mass,
            target_accept_prob=args.target_accept,
            max_tree_depth=(args.max_tree_depth, args.max_tree_depth),
            find_heuristic_step_size=True,
        )
        init0 = time.perf_counter()
        state = kernel.init(jax.random.PRNGKey(args.seed), args.warmup, init_params=jnp.zeros_like(y0))
        sample_once = jax.jit(lambda current: kernel.sample(current, (), {}))
        init_compile_s = time.perf_counter() - init0
        steps, divergences, energies, samples = [], [], [], []
        run0 = time.perf_counter()
        for index in range(args.warmup + args.samples):
            state = sample_once(state)
            if index >= args.warmup:
                samples.append(np.asarray(state.z))
                steps.append(int(np.asarray(state.num_steps)))
                divergences.append(bool(np.asarray(state.diverging)))
                energies.append(float(np.asarray(state.energy)))
        run_s = time.perf_counter() - run0
        result["timing"].update({"sampler_init_compile_s": init_compile_s, "sampler_transition_s": run_s})
        result["nuts"] = {
            "warmup": args.warmup,
            "draws": args.samples,
            "num_steps": steps,
            "mean_num_steps": float(np.mean(steps)) if steps else None,
            "max_num_steps": max(steps, default=None),
            "tree_depth_saturations": int(np.sum(np.asarray(steps) >= 2**args.max_tree_depth - 1)),
            "divergences": int(np.sum(divergences)),
            "finite_energy": bool(np.all(np.isfinite(energies))),
            "samples_whitened": np.asarray(samples).tolist(),
        }
    elif args.sampler == "dynesty":
        import dynesty

        calls = 0
        call_seconds = []

        def loglike_counted(theta):
            nonlocal calls
            start = time.perf_counter()
            full = problem["active_to_full"](theta)
            y = problem["transform"].active_initial_unconstrained(full)
            _, log_jacobian = problem["transform"].unconstrained_to_full_jax(
                y, problem["fixed_full"]
            )
            objective = potential(jnp.asarray(y)) + log_jacobian
            bmaj = problem["bmaj"]
            center_penalty = 0.5 * (
                (full[4] * bmaj / 0.5) ** 2 + (full[5] * bmaj / 0.5) ** 2
            )
            value = float(-objective + center_penalty)
            call_seconds.append(time.perf_counter() - start)
            calls += 1
            return value

        sampler0 = time.perf_counter()
        sampler = dynesty.DynamicNestedSampler(
            loglike_counted,
            problem["prior_transform"],
            len(problem["y_map"]),
            nlive=args.nlive,
            sample="rslice",
            slices=args.slices,
            first_update={"min_ncall": 0, "min_eff": 101.0},
            rstate=np.random.default_rng(args.seed),
        )
        construct_s = time.perf_counter() - sampler0
        run0 = time.perf_counter()
        sampler.run_nested(
            nlive_init=args.nlive,
            maxiter_init=args.maxiter,
            maxcall_init=args.maxcall,
            maxiter=args.maxiter,
            maxcall=args.maxcall,
            maxbatch=0,
            use_stop=False,
            print_progress=False,
        )
        run_s = time.perf_counter() - run0
        nested = sampler.results
        weights = np.exp(nested.logwt - logsumexp(nested.logwt))
        result["timing"].update({"sampler_construct_s": construct_s, "sampler_run_s": run_s})
        result["dynesty"] = {
            "version": dynesty.__version__,
            "sample": "rslice",
            "slices": args.slices,
            "nlive_init": args.nlive,
            "first_update": {"min_ncall": 0, "min_eff": 101.0},
            "maxiter": args.maxiter,
            "maxcall": args.maxcall,
            "actual_loglikelihood_calls": calls,
            "reported_ncall": int(np.sum(nested.ncall)),
            "likelihood_call_seconds": call_seconds,
            "likelihood_call_median_s": float(np.median(call_seconds)),
            "iterations": int(len(nested.samples)),
            "effective_weighted_samples": float(1.0 / np.sum(weights**2)),
            "logz_last": float(nested.logz[-1]),
            "logzerr_last": float(nested.logzerr[-1]),
            "samples": np.asarray(nested.samples).tolist(),
            "log_weights": np.asarray(nested.logwt).tolist(),
            "normalized_weights": weights.tolist(),
        }
    else:
        raise ValueError(args.sampler)

    result["timing"]["total_elapsed_s"] = time.perf_counter() - wall0
    result["memory"]["final_max_rss_mib"] = rss_mib()
    result["completed_utc"] = now()
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "sampler": args.sampler, "target": args.target, "elapsed_s": result["timing"]["total_elapsed_s"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--sampler", choices=("profile", "nuts", "dynesty"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9609)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--max-tree-depth", type=int, default=6)
    parser.add_argument("--adapt-mass", action="store_true")
    parser.add_argument("--nlive", type=int, default=20)
    parser.add_argument("--slices", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--maxcall", type=int, default=80)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
