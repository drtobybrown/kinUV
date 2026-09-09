#!/usr/bin/env python3
"""Build a versioned posterior-bearing candidate after NUTS finalization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import gaussian_filter

from kinuv.diagnostics.style import COLOUR, apply_style, save_publication


TARGETS = ("KGAS066", "KGAS007")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="ascii")


def levels(histogram):
    ordered = np.sort(np.asarray(histogram).ravel())[::-1]
    cumulative = np.cumsum(ordered)
    cumulative /= cumulative[-1]
    return sorted({float(ordered[min(np.searchsorted(cumulative, p), ordered.size - 1)]) for p in (0.95, 0.68)})


def corner(target: str, samples: dict[str, np.ndarray], output: Path, accepted: bool) -> None:
    preferred = (
        ("v_flat_kms", "r_turn_arcsec", "inclination_deg", "pa_deg", "vsys_native_kms", "sigma_inner_kms", "sigma_outer_kms")
        if "v_flat_kms" in samples
        else ("u_knot_1_kms", "u_knot_2_kms", "u_knot_3_kms", "u_knot_4_kms", "inclination_deg", "pa_deg", "vsys_native_kms", "sigma_inner_kms")
    )
    names = [name for name in preferred if name in samples]
    labels = {
        "v_flat_kms": r"$V_{\rm flat}$", "r_turn_arcsec": r"$R_{\rm turn}$",
        "inclination_deg": r"$i$", "pa_deg": r"$\mathrm{PA}$",
        "vsys_native_kms": r"$V_{\rm sys}$", "sigma_inner_kms": r"$\sigma_{\rm in}$",
        "sigma_outer_kms": r"$\sigma_{\rm out}$", "u_knot_1_kms": r"$u_1$",
        "u_knot_2_kms": r"$u_2$", "u_knot_3_kms": r"$u_3$", "u_knot_4_kms": r"$u_4$",
    }
    values = {name: np.asarray(samples[name], dtype=float).reshape(-1) for name in names}
    limits = {}
    for name, value in values.items():
        lo, hi = np.quantile(value, (0.002, 0.998))
        pad = max(0.08 * (hi - lo), np.finfo(float).eps)
        limits[name] = (lo - pad, hi + pad)
    size = len(names)
    apply_style(columns=2, aspect_ratio=1.0)
    figure, axes = plt.subplots(size, size, figsize=(9.0, 9.0), squeeze=False)
    for row, y_name in enumerate(names):
        for column, x_name in enumerate(names):
            axis = axes[row, column]
            if column > row:
                axis.set_visible(False)
                continue
            x = values[x_name]
            if row == column:
                axis.hist(x, bins=35, density=True, color=COLOUR["model"], alpha=0.45)
                for quantile in np.quantile(x, (0.16, 0.50, 0.84)):
                    axis.axvline(quantile, color=COLOUR["data"], linewidth=0.7)
                axis.set_yticks([])
            else:
                histogram, x_edges, y_edges = np.histogram2d(x, values[y_name], bins=36, range=(limits[x_name], limits[y_name]))
                smooth = gaussian_filter(histogram.T, 1.0)
                contour_levels = levels(smooth)
                if contour_levels:
                    axis.contour(
                        0.5 * (x_edges[:-1] + x_edges[1:]),
                        0.5 * (y_edges[:-1] + y_edges[1:]),
                        smooth,
                        levels=contour_levels,
                        colors=("#7CB7D3", COLOUR["model"]),
                        linewidths=(0.8, 1.1),
                    )
                axis.set_ylim(*limits[y_name])
            axis.set_xlim(*limits[x_name])
            axis.xaxis.set_major_locator(MaxNLocator(3))
            axis.yaxis.set_major_locator(MaxNLocator(3))
            axis.tick_params(labelsize=8)
            if row < size - 1:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(labels[x_name], fontsize=10)
                axis.tick_params(axis="x", labelrotation=35)
            if column > 0:
                axis.tick_params(labelleft=False)
            elif row > 0:
                axis.set_ylabel(labels[y_name], fontsize=10)
    state = "accepted" if accepted else "unaccepted"
    figure.suptitle(f"{target}: conditional NUTS posterior ({state})", fontsize=13, y=0.995)
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.11, top=0.95, wspace=0.08, hspace=0.08)
    save_publication(figure, output / "posterior_corner")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-candidate-root", type=Path, required=True)
    parser.add_argument("--posterior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    for target in args.targets:
        source = args.map_candidate_root / target
        destination = args.output_root / target
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        posterior_source = args.posterior_root / target
        posterior_destination = destination / "best_model" / "posterior"
        shutil.copytree(posterior_source, posterior_destination, copy_function=shutil.copy2)
        summary = json.loads((posterior_source / "summary.json").read_text(encoding="utf-8"))
        with np.load(posterior_source / "posterior_samples.npz", allow_pickle=False) as archive:
            samples = {name: np.asarray(archive[name]) for name in archive.files if name not in {"chain", "draw"}}
        corner(target, samples, destination / "plots", bool(summary["gates"]["accepted"]))
        with (destination / "README.md").open("a", encoding="ascii") as stream:
            stream.write("\n## Conditional posterior\n\n")
            stream.write(
                f"Four {summary['draws_per_chain']}-draw NUTS chains completed after "
                f"{summary['warmup_per_chain']} warm-up steps. Posterior gate accepted: "
                f"`{str(summary['gates']['accepted']).lower()}`. See "
                "`best_model/posterior/summary.json` for R-hat, ESS, divergences, "
                "BFMI, and tree-depth diagnostics.\n"
            )
        files = {
            path.relative_to(destination).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "POSTERIOR_CANDIDATE_MANIFEST.json"
        }
        write_json(destination / "POSTERIOR_CANDIDATE_MANIFEST.json", {
            "schema_version": "kinuv-collaborator-posterior-candidate-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "target_id": target,
            "status": "POSTERIOR_CANDIDATE" if summary["gates"]["accepted"] else "UNACCEPTED_POSTERIOR_CANDIDATE",
            "posterior_gates": summary["gates"],
            "files": files,
        })
        print(json.dumps({"target": target, "accepted": summary["gates"]["accepted"], "output": str(destination)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
