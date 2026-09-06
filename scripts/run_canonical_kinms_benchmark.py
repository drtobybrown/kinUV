#!/usr/bin/env python3
"""Run downstream kinUV-versus-KinMS diagnostics for canonical targets.

KGAS066 and KGAS007 are selected by default. Existing verified KinMS cubes are
reused; otherwise the external KinMS worker is invoked. This script never
calls a kinUV optimizer or sampler and never contributes to the visibility
likelihood.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from kinuv.diagnostics.kinms_benchmark import write_cube_benchmark

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/benchmarks/canonical-kinms.json"
DEFAULT_TARGETS = ("KGAS066", "KGAS007")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(directory: Path) -> None:
    entries = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            entries.append(f"{_sha256(path)}  {path.relative_to(directory)}")
    (directory / "CHECKSUMS.sha256").write_text("\n".join(entries) + "\n")


def _git_state() -> dict[str, str | bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _require_file(path, label: str) -> Path:
    item = Path(path)
    if not item.is_file():
        raise FileNotFoundError(f"{label} does not exist: {item}")
    return item


def _run_kinms_fit(target: dict, destination: Path, python: str) -> tuple[Path, Path]:
    fit = target.get("fit")
    if not fit:
        raise FileNotFoundError(
            f"{target['target_id']} has no reusable KinMS cube and no fit configuration"
        )
    work = destination / "kinms"
    work.mkdir(parents=True, exist_ok=True)
    config = {
        "cube": target["data_cube"],
        "mask": target["mask_cube"],
        "work": str(work),
        **fit,
    }
    config_path = work / "fit_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    command = [python, str(REPO / "external/_kinms_best_worker.py"), str(config_path)]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (work / "worker.log").write_text(proc.stdout)
    if proc.returncode:
        raise RuntimeError(
            f"KinMS fit failed for {target['target_id']}; see {work / 'worker.log'}"
        )
    return _require_file(work / "model_cube.fits", "generated KinMS cube"), _require_file(
        work / "kinms_fit_result.json", "generated KinMS result"
    )


def _resolve_kinms(target: dict, destination: Path, python: str, no_fit: bool):
    cube = target.get("kinms_cube")
    result = target.get("kinms_result")
    if cube and Path(cube).is_file():
        return Path(cube), Path(result) if result and Path(result).is_file() else None
    generated = destination / "kinms/model_cube.fits"
    generated_result = destination / "kinms/kinms_fit_result.json"
    if generated.is_file():
        return generated, generated_result if generated_result.is_file() else None
    if no_fit:
        raise FileNotFoundError(f"no KinMS cube available for {target['target_id']}")
    return _run_kinms_fit(target, destination, python)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", action="append", choices=DEFAULT_TARGETS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--kinms-python",
        default=os.environ.get("KINUV_KINMS_PYTHON", sys.executable),
        help="Python executable containing KinMS, NumPy, SciPy, and Astropy",
    )
    parser.add_argument("--no-fit", action="store_true", help="require reusable KinMS cubes")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    selected = set(args.target or DEFAULT_TARGETS)
    output_root = args.output_root or Path(config["output_root"])
    code = _git_state()
    receipts = []
    indexed = {item["target_id"]: item for item in config["targets"]}
    missing = selected - indexed.keys()
    if missing:
        raise KeyError(f"targets absent from benchmark config: {sorted(missing)}")
    for target_id in DEFAULT_TARGETS:
        if target_id not in selected:
            continue
        target = indexed[target_id]
        destination = output_root / target_id
        kinms_cube, kinms_result = _resolve_kinms(
            target, destination, args.kinms_python, args.no_fit
        )
        receipt = write_cube_benchmark(
            target_id=target_id,
            data_cube=_require_file(target["data_cube"], "official data cube"),
            mask_cube=_require_file(target["mask_cube"], "official mask cube"),
            kinuv_cube=_require_file(target["kinuv_cube"], "kinUV model cube"),
            kinms_cube=kinms_cube,
            output_dir=destination,
            pa_deg=float(target["pa_deg"]),
            inclination_deg=float(target["inclination_deg"]),
            vsys_kms=float(target["vsys_optical_kms"]),
            dx_arcsec=float(target.get("dx_arcsec", 0.0)),
            dy_arcsec=float(target.get("dy_arcsec", 0.0)),
        )
        kinuv_summary = _require_file(target["kinuv_summary"], "kinUV summary")
        receipt["kinuv_summary"] = {
            "path": str(kinuv_summary),
            "sha256": _sha256(kinuv_summary),
        }
        receipt["kinms_result"] = (
            {"path": str(kinms_result), "sha256": _sha256(kinms_result)}
            if kinms_result
            else None
        )
        receipt["kinuv_code"] = code
        (destination / "benchmark.json").write_text(json.dumps(receipt, indent=2) + "\n")
        _write_checksums(destination)
        receipts.append(receipt)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "kinuv-canonical-benchmark-index-v1",
        "targets_requested": [name for name in DEFAULT_TARGETS if name in selected],
        "targets_completed": [item["target_id"] for item in receipts],
        "likelihood_untouched": True,
        "config": {"path": str(args.config), "sha256": _sha256(args.config)},
        "kinuv_code": code,
        "products": {item["target_id"]: str(output_root / item["target_id"]) for item in receipts},
    }
    (output_root / "benchmark_index.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_checksums(output_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
