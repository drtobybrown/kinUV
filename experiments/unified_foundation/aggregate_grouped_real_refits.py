#!/usr/bin/env python3
"""Aggregate immutable grouped real-visibility training-refit evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from grouped_real_visibility_benchmark import lower_confidence


TARGETS = ("KGAS066", "KGAS007")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    outputs = (root / "result.json", root / "REPORT.md", root / "manifest.json")
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to replace aggregate output")
    dispatch_path = root / "dispatch.json"
    dispatch = json.loads(dispatch_path.read_text())
    targets = {}
    source_hashes = set()
    commits = set()
    for target in TARGETS:
        folds = []
        for fold_id in range(5):
            directory = root / target / f"fold-{fold_id}"
            exit_record = json.loads((directory / "headless_exit.json").read_text())
            if exit_record["exit_code"] != 0:
                raise RuntimeError(f"worker failed: {target} fold {fold_id}")
            path = directory / "result.json"
            fold = json.loads(path.read_text())
            if fold["target_id"] != target or fold["fold_id"] != fold_id:
                raise RuntimeError(f"fold identity mismatch: {path}")
            source_hashes.add(fold["git"]["worker_source_sha256"])
            commits.add(fold["git"]["code_commit"])
            folds.append({
                "fold_id": fold_id,
                "state": fold["state"],
                "training_all_gates_pass": fold["training"]["all_gates_pass"],
                "optimizer_success": fold["training"]["optimizer_success"],
                "optimizer_message": fold["training"]["optimizer_message"],
                "iterations": fold["training"]["iterations"],
                "gradient_inf_per_complex": fold["training"]["gradient_inf_per_complex"],
                "heldout_delta_chi2": fold["heldout"]["delta_chi2_kinms_minus_unified"],
                "heldout_delta_chi2_per_component": fold["heldout"]["delta_chi2_per_component"],
                "heldout_components": fold["heldout"]["n_real_imag_components"],
                "timing_s": fold["timing"],
                "result": {"path": str(path), "sha256": sha256(path)},
            })
        confidence = lower_confidence([fold["heldout_delta_chi2_per_component"] for fold in folds])
        all_optimizers = all(fold["training_all_gates_pass"] for fold in folds)
        targets[target] = {
            "folds": folds,
            "confidence": confidence,
            "all_optimizer_gates_pass": all_optimizers,
            "positive_lower95_and_optimizer_gate": all_optimizers and confidence["positive_lower_bound_gate"],
        }
    if source_hashes != {dispatch["source"]["worker"]["sha256"]} or commits != {dispatch["code_commit"]}:
        raise RuntimeError("worker source or code commit differs across folds")
    gate = all(item["positive_lower95_and_optimizer_gate"] for item in targets.values())
    result = {
        "schema_version": "kinuv-unified-real-grouped-refit-aggregate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "MEASURED_GATE_PASS" if gate else "MEASURED_GATE_FAIL",
        "gate_pass": gate,
        "code_commit": dispatch["code_commit"],
        "worker_source_sha256": next(iter(source_hashes)),
        "comparison": "per-fold unified training refit versus frozen full-data KinMS through identical PB, uv, and spectral operators",
        "conditioning": dispatch["conditioning"],
        "fixed_model_r2_preserved": "/arc/projects/KILOGAS/analysis/toby_sandbox/results/incoming/unified-foundation-phase4-real-20260910-r2",
        "targets": targets,
    }
    atomic_text(root / "result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Phase 4 grouped real-visibility training-refit benchmark",
        "",
        f"State: `{result['state']}`. Unified 14D parameters were refit on each training partition from the selected full-data MAP. KinMS remained frozen at its full-data best fit, which is conservative for KinMS.",
        "",
        "The unified empirical morphology remains conditioned on the full canonical cube, so this is genuine held-out kinematic prediction but not a fully end-to-end leakage-free workflow.",
        "",
        "| Target | Fold deltas chi2/component | Mean | One-sided lower 95% | Optimizers | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target, item in targets.items():
        values = ", ".join(f"{fold['heldout_delta_chi2_per_component']:.8f}" for fold in item["folds"])
        conf = item["confidence"]
        passed = sum(fold["training_all_gates_pass"] for fold in item["folds"])
        lines.append(f"| {target} | {values} | {conf['mean_delta_chi2_per_component']:.8f} | {conf['lower_bound_delta_chi2_per_component']:.8f} | {passed}/5 | {item['positive_lower95_and_optimizer_gate']} |")
    lines.extend(["", "Positive values favor unified kinUV. Exact fold convergence, scores, timings, inputs, and hashes are in each fold result and the aggregate JSON.", ""])
    atomic_text(root / "REPORT.md", "\n".join(lines))
    manifest = {
        "schema_version": "kinuv-unified-real-grouped-refit-manifest-v1",
        "files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    atomic_text(root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"state": result["state"], "gate_pass": gate, "targets": {key: value["confidence"] for key, value in targets.items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
