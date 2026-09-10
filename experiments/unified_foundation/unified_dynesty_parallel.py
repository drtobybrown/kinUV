#!/usr/bin/env python3
"""Shared-JIT thread-pool DynamicNestedSampler worker for the unified posterior."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import platform
import signal
import threading
import time

import numpy as np
from scipy.special import logsumexp

from unified_dynesty import (
    UnifiedLikelihood,
    UnifiedPriorTransform,
    atomic_npz,
    publish,
    scientific_input_hashes,
    sha256,
    utc_now,
    write_heartbeat,
)
from unified_map_runner import atomic_json, build_problem


USE_POOL = {
    "prior_transform": False,
    "loglikelihood": True,
    "propose_point": True,
    "update_bound": False,
}


class DynestyThreadPool:
    """Minimal pool interface backed by threads sharing one warmed JAX runtime."""

    def __init__(self, workers):
        self.workers = workers
        self._executor = None

    @property
    def size(self):
        return self.workers

    def __enter__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="kinuv-loglike"
        )
        return self

    def map(self, function, values):
        return self._executor.map(function, values)

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None


def rebind_restored_sampler(sampler, pool, likelihood, prior, workers):
    """Attach live functions and executor to every sampler held by a checkpoint."""
    samplers = [sampler]
    if getattr(sampler, "sampler", None) is not None:
        samplers.append(sampler.sampler)
    if getattr(sampler, "batch_sampler", None) is not None:
        samplers.append(sampler.batch_sampler)
    for current in samplers:
        current.M = pool.map
        current.pool = pool
        current.queue_size = workers
        current.prior_transform = prior
        current.loglikelihood.pool = pool
        current.loglikelihood.loglikelihood = likelihood


def run(args):
    import dynesty

    scratch = args.scratch / args.target / f"replicate-{args.replicate}"
    durable = args.durable / args.target / f"replicate-{args.replicate}"
    scratch.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    scratch_checkpoint = scratch / "dynesty.save"
    durable_checkpoint = durable / "dynesty.save"
    status_path = durable / "status.json"
    heartbeat_path = durable / "heartbeat.txt"

    map_doc = json.loads(args.map_result.read_text(encoding="utf-8"))
    if map_doc.get("target_id") != args.target or not map_doc.get("gate_pass"):
        raise RuntimeError("MAP result must match target and pass its gate")
    if map_doc.get("git", {}).get("commit") != args.map_commit:
        raise RuntimeError("MAP result commit mismatch")

    context, spec, density, initial, provenance = build_problem(args.target)
    del context, density
    gc.collect()
    prior = UnifiedPriorTransform(spec)
    likelihood = UnifiedLikelihood(args.target)
    state_q = np.asarray(map_doc.get("optimum_z", initial), dtype=np.float64)
    if state_q.shape != (14,):
        raise RuntimeError("MAP optimum_z must have shape (14,)")
    del initial
    contract = {
        "code_commit": args.code_commit,
        "map_commit": args.map_commit,
        "map_sha256": sha256(args.map_result),
        "scientific_inputs": scientific_input_hashes(map_doc),
        "dynesty_version": dynesty.__version__,
        "dynesty_module_sha256": sha256(Path(dynesty.__file__)),
        "ndim": 14,
        "nlive": args.nlive,
        "sample": "rslice",
        "slices": args.slices,
        "n_effective": args.n_effective,
        "dlogz_init": args.dlogz_init,
        "prior": "exact normalized unified chart prior",
        "likelihood": "sole fixed-C1 visibility likelihood",
        "parallel": {
            "executor": "ThreadPoolExecutor",
            "workers": args.workers,
            "queue_size": args.workers,
            "use_pool": USE_POOL,
            "shared_prewarmed_jit": True,
            "jax_threads_per_call": 1,
        },
    }
    prior_status = json.loads(status_path.read_text(encoding="ascii")) if status_path.is_file() else None
    if prior_status and prior_status.get("contract", {}).get("resume_lineage"):
        contract["resume_lineage"] = prior_status["contract"]["resume_lineage"]
    migrated = False
    if prior_status is None and args.resume_root is not None:
        source = args.resume_root / args.target / f"replicate-{args.replicate}"
        source_status_path = source / "status.json"
        source_checkpoint = source / "dynesty.save"
        if not source_status_path.is_file() or not source_checkpoint.is_file():
            raise FileNotFoundError(f"incomplete resume source: {source}")
        source_status = json.loads(source_status_path.read_text(encoding="ascii"))
        source_contract = source_status.get("contract", {})
        required_equal = (
            "map_commit", "map_sha256", "scientific_inputs", "ndim", "nlive",
            "sample", "slices", "n_effective", "prior", "likelihood",
        )
        if (
            source_status.get("target_id") != args.target
            or source_status.get("replicate") != args.replicate
            or source_status.get("seed") != args.seed
            or any(source_contract.get(key) != contract.get(key) for key in required_equal)
        ):
            raise RuntimeError("resume source scientific contract mismatch")
        publish(source_checkpoint, durable_checkpoint)
        contract["resume_lineage"] = {
            "source_root": str(args.resume_root.resolve()),
            "source_code_commit": source_contract.get("code_commit"),
            "source_checkpoint_sha256": sha256(durable_checkpoint),
            "source_status_sha256": sha256(source_status_path),
            "source_iteration": source_status.get("iteration"),
            "reason": "continue identical sampler state under right-sized evidence stopping tolerance",
        }
        migrated = True
    compatible = bool(
        (migrated or (
            prior_status
            and prior_status.get("contract") == contract
            and prior_status.get("seed") == args.seed
        ))
        and durable_checkpoint.is_file()
    )
    if prior_status and prior_status.get("state") == "SUCCEEDED":
        return 0
    if prior_status and not compatible:
        raise RuntimeError("existing parallel nested run is not resume compatible")
    if compatible:
        publish(durable_checkpoint, scratch_checkpoint)

    state = {
        "schema_version": "kinuv-unified-dynesty-parallel-status-v1",
        "state": "PREWARMING",
        "target_id": args.target,
        "replicate": args.replicate,
        "seed": args.seed,
        "session_id": os.environ.get(
            "SKAHA_SESSION_ID", os.environ.get("skaha_sessionid", platform.node())
        ),
        "pid": os.getpid(),
        "started_utc": prior_status.get("started_utc", utc_now()) if compatible else utc_now(),
        "resumed": compatible,
        "contract": contract,
        "provenance": provenance,
    }
    atomic_json(status_path, state)
    write_heartbeat(heartbeat_path, state)
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(55):
            state["updated_utc"] = utc_now()
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
        state.update({"state": "RUNNING", "iteration": int(niter), "calls": int(ncall)})

    try:
        # Compile and materialize the sole likelihood once. All executor threads
        # then call the same immutable data and compiled executable.
        warm_value = likelihood(state_q)
        if not np.isfinite(warm_value):
            raise RuntimeError("nonfinite likelihood at the accepted MAP during prewarm")
        state.update({"state": "STARTING_POOL", "prewarm_log_likelihood": warm_value})
        atomic_json(status_path, state)
        write_heartbeat(heartbeat_path, state)
        with DynestyThreadPool(args.workers) as pool:
            if compatible:
                sampler = dynesty.DynamicNestedSampler.restore(str(scratch_checkpoint), pool=pool)
                if sampler.queue_size != args.workers:
                    raise RuntimeError("restored checkpoint queue_size mismatch")
                rebind_restored_sampler(sampler, pool, likelihood, prior, args.workers)
            else:
                sampler = dynesty.DynamicNestedSampler(
                    likelihood,
                    prior,
                    14,
                    nlive=args.nlive,
                    sample="rslice",
                    slices=args.slices,
                    bound="multi",
                    rstate=np.random.default_rng(args.seed),
                    pool=pool,
                    queue_size=args.workers,
                    use_pool=USE_POOL,
                )
            state["state"] = "RUNNING"
            atomic_json(status_path, state)
            write_heartbeat(heartbeat_path, state)
            sampler.run_nested(
                nlive_init=args.nlive,
                nlive_batch=args.nlive,
                n_effective=args.n_effective,
                dlogz_init=args.dlogz_init,
                resume=compatible,
                checkpoint_file=str(scratch_checkpoint),
                checkpoint_every=60,
                print_progress=True,
                print_func=progress,
            )
            if scratch_checkpoint.is_file():
                publish(scratch_checkpoint, durable_checkpoint)
            nested = sampler.results

        weights = np.exp(nested.logwt - logsumexp(nested.logwt))
        weighted_ess = float(1.0 / np.sum(weights**2))
        atomic_npz(
            durable / "posterior_weighted.npz",
            q_unified=np.asarray(nested.samples),
            log_likelihood=np.asarray(nested.logl),
            log_weight=np.asarray(nested.logwt),
            weight=weights,
            log_evidence=np.asarray(nested.logz),
            log_evidence_error=np.asarray(nested.logzerr),
        )
        result = {
            "schema_version": "kinuv-unified-dynesty-parallel-replicate-v1",
            "state": "SUCCEEDED",
            "created_utc": utc_now(),
            "target_id": args.target,
            "replicate": args.replicate,
            "contract": contract,
            "n_samples": int(len(nested.samples)),
            "n_calls": int(np.sum(nested.ncall)),
            "weighted_ess": weighted_ess,
            "log_evidence": float(nested.logz[-1]),
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
        state.update(
            {
                "state": "FAILED_RESUMABLE",
                "error": f"{type(error).__name__}: {error}",
                "completed_utc": utc_now(),
            }
        )
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
    parser.add_argument("--workers", type=int, default=4, choices=(4, 8, 16, 32))
    parser.add_argument("--slices", type=int, default=17)
    parser.add_argument("--dlogz-init", type=float, default=0.01)
    parser.add_argument("--resume-root", type=Path)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
