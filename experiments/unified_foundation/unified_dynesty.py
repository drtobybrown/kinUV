#!/usr/bin/env python3
"""Resumable DynamicNestedSampler fallback for the unified 14D posterior."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import tempfile
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp, ndtri

from unified_map_runner import atomic_json, build_problem


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scientific_input_hashes(map_doc):
    provenance = map_doc["provenance"]
    config_path = Path(provenance["config_path"])
    checkpoint_path = Path(provenance["checkpoint_path"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    paths = {
        "target_config": config_path,
        "unified_parent_checkpoint": checkpoint_path,
        "c1_covariance": Path(provenance["covariance_path"]),
        "s2_emissivity": Path(checkpoint["s2_summary_path"]),
        "visibility": Path(config["visibility_npz"]),
        "template_ico": Path(config["template_ico"]),
        "fit_window_cube": Path(config["fit_window_cube"]),
    }
    return {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in paths.items()}


def atomic_npz(path, **arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


class UnifiedPriorTransform:
    """Exact normalized transform for ``unified_prior_components``."""

    def __init__(self, spec):
        self.normal_mean = np.array([
            np.log(spec.flux_reference), 0.0, 0.0, 0.0,
            np.log(spec.u_reference_kms), np.log(spec.sigma_reference_kms), 0.0, 0.0,
        ])
        self.normal_scale = np.array([
            spec.log_flux_scale,
            spec.vsys_scale_kms / spec.dv_kms,
            spec.center_scale_arcsec / spec.support.bmaj_arcsec,
            spec.center_scale_arcsec / spec.support.bmaj_arcsec,
            spec.log_u_scale, spec.log_sigma_scale, 1.0, 1.0,
        ])
        from kinuv.profiles.unified import rotation_prior_precision
        self.rotation_precision_chol = np.linalg.cholesky(rotation_prior_precision(spec.support))

    def __call__(self, unit):
        unit = np.clip(np.asarray(unit, dtype=np.float64), 1e-14, 1.0 - 1e-14)
        normal = self.normal_mean + self.normal_scale * ndtri(unit[[0, 2, 3, 4, 6, 11, 12, 13]])
        q = np.empty(14, dtype=np.float64)
        q[[0, 2, 3, 4, 6, 11, 12, 13]] = normal
        q[1] = np.log(unit[1]) - np.log1p(-unit[1])
        q[5] = np.log(unit[5]) - np.log1p(-unit[5])
        q[7:11] = np.linalg.solve(self.rotation_precision_chol.T, ndtri(unit[7:11]))
        return q


class UnifiedLikelihood:
    """Pickle-safe, lazily reconstructed value-only C1 visibility likelihood."""

    def __init__(self, target):
        self.target = target
        self.calls = 0
        self.seconds = 0.0
        self._compiled = None

    def _setup(self):
        _, _, density, _, _ = build_problem(self.target)
        self._compiled = jax.jit(density.log_likelihood)

    def __call__(self, q):
        if self._compiled is None:
            self._setup()
        started = time.perf_counter()
        value = float(self._compiled(jnp.asarray(q)))
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return value if np.isfinite(value) else -np.inf

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled"] = None
        return state


def write_heartbeat(path, state):
    text = (
        f"utc={utc_now()} target={state['target_id']} replicate={state['replicate']} "
        f"session={state['session_id']} state={state['state']} iteration={state.get('iteration', 'NA')} "
        f"calls={state.get('calls', 'NA')}\n"
    )
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="ascii")
    os.replace(temporary, path)


def run(args):
    import dynesty

    scratch = args.scratch / args.target / f"replicate-{args.replicate}"
    durable = args.durable / args.target / f"replicate-{args.replicate}"
    scratch.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    scratch_checkpoint = scratch / "dynesty.save"
    durable_checkpoint = durable / "dynesty.save"
    status_path, heartbeat_path = durable / "status.json", durable / "heartbeat.txt"
    map_doc = json.loads(args.map_result.read_text(encoding="utf-8"))
    if map_doc.get("target_id") != args.target or not map_doc.get("gate_pass"):
        raise RuntimeError("MAP result must match target and pass its gate")
    if map_doc.get("git", {}).get("commit") != args.map_commit:
        raise RuntimeError("MAP result commit mismatch")
    context, spec, _, _, provenance = build_problem(args.target)
    contract = {
        "code_commit": args.code_commit, "map_commit": args.map_commit,
        "map_sha256": sha256(args.map_result), "scientific_inputs": scientific_input_hashes(map_doc),
        "dynesty_version": dynesty.__version__,
        "dynesty_module_sha256": sha256(Path(dynesty.__file__)),
        "ndim": 14, "nlive": args.nlive, "sample": "rslice", "n_effective": args.n_effective,
        "prior": "exact normalized unified chart prior", "likelihood": "sole fixed-C1 visibility likelihood",
    }
    prior_status = json.loads(status_path.read_text(encoding="ascii")) if status_path.is_file() else None
    compatible = bool(prior_status and prior_status.get("contract") == contract
                      and prior_status.get("seed") == args.seed and durable_checkpoint.is_file())
    if prior_status and prior_status.get("state") == "SUCCEEDED":
        return 0
    if prior_status and not compatible:
        raise RuntimeError("existing nested run is not resume compatible")
    if compatible:
        publish(durable_checkpoint, scratch_checkpoint)
        sampler = dynesty.DynamicNestedSampler.restore(str(scratch_checkpoint))
        likelihood = sampler.loglikelihood.loglikelihood
    else:
        likelihood = UnifiedLikelihood(args.target)
        sampler = dynesty.DynamicNestedSampler(
            likelihood, UnifiedPriorTransform(spec), 14, nlive=args.nlive,
            sample="rslice", bound="multi", rstate=np.random.default_rng(args.seed),
        )
    state = {
        "schema_version": "kinuv-unified-dynesty-status-v1", "state": "RUNNING",
        "target_id": args.target, "replicate": args.replicate, "seed": args.seed,
        "session_id": os.environ.get("SKAHA_SESSION_ID", os.environ.get("skaha_sessionid", platform.node())),
        "pid": os.getpid(), "started_utc": prior_status.get("started_utc", utc_now()) if compatible else utc_now(),
        "resumed": compatible, "contract": contract, "provenance": provenance,
    }
    atomic_json(status_path, state)
    write_heartbeat(heartbeat_path, state)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(55):
            state.update({"calls": int(getattr(likelihood, "calls", 0)), "updated_utc": utc_now()})
            if scratch_checkpoint.is_file():
                publish(scratch_checkpoint, durable_checkpoint)
            atomic_json(status_path, state)
            write_heartbeat(heartbeat_path, state)

    threading.Thread(target=heartbeat, daemon=True).start()

    def terminate(signum, _frame):
        state.update({"state": "INTERRUPTED_RESUMABLE", "signal": int(signum), "updated_utc": utc_now()})
        if scratch_checkpoint.is_file():
            publish(scratch_checkpoint, durable_checkpoint)
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    started = time.perf_counter()

    def progress(_results, niter, ncall, **_kwargs):
        state.update({"iteration": int(niter), "calls": int(ncall)})

    try:
        sampler.run_nested(
            nlive_init=args.nlive, nlive_batch=args.nlive, n_effective=args.n_effective,
            resume=compatible, checkpoint_file=str(scratch_checkpoint), checkpoint_every=60,
            print_progress=True, print_func=progress,
        )
        if scratch_checkpoint.is_file():
            publish(scratch_checkpoint, durable_checkpoint)
        nested = sampler.results
        weights = np.exp(nested.logwt - logsumexp(nested.logwt))
        weighted_ess = float(1.0 / np.sum(weights**2))
        atomic_npz(
            durable / "posterior_weighted.npz", q_unified=np.asarray(nested.samples),
            log_likelihood=np.asarray(nested.logl), log_weight=np.asarray(nested.logwt),
            weight=weights, log_evidence=np.asarray(nested.logz), log_evidence_error=np.asarray(nested.logzerr),
        )
        result = {
            "schema_version": "kinuv-unified-dynesty-replicate-v1", "state": "SUCCEEDED",
            "created_utc": utc_now(), "target_id": args.target, "replicate": args.replicate,
            "contract": contract, "n_samples": int(len(nested.samples)), "n_calls": int(np.sum(nested.ncall)),
            "weighted_ess": weighted_ess, "log_evidence": float(nested.logz[-1]),
            "log_evidence_error": float(nested.logzerr[-1]),
            "sampling_efficiency_percent": float(nested.eff),
            "elapsed_s": time.perf_counter() - started,
            "posterior_fidelity": "PENDING_INDEPENDENT_REPLICATE_AGREEMENT",
        }
        atomic_json(durable / "result.json", result)
        state.update({"state": "SUCCEEDED", "completed_utc": utc_now(), "result": result})
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        return 0
    except SystemExit:
        raise
    except BaseException as error:
        state.update({"state": "FAILED_RESUMABLE", "error": f"{type(error).__name__}: {error}",
                      "completed_utc": utc_now()})
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        raise
    finally:
        stop.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("KGAS066", "KGAS007"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--map-result", type=Path, required=True)
    parser.add_argument("--map-commit", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--n-effective", type=int, default=2000)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
