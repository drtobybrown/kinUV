"""Headless NUTS product plots. PNGs on /arc; FITS stay in the run dir.

Corner, leftover chi2, and Data|Model|Residual at the NUTS mean. Not a
likelihood. 16/50/84 on the corner are not calibrated.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from kinuv.diagnostics.figures import plot_leftover_chi2, plot_posterior_corner
from kinuv.diagnostics.flags import leftover_velocity_structured
from kinuv.infer.posterior import PARAM_NAMES
from kinuv.runner.canfar import PROJECT_ROOT, REPO, utc_now, write_json

os.environ.setdefault("MPLBACKEND", "Agg")

ARTIFACT_G3 = REPO / "docs" / "reviews" / "artifacts" / "2026-08-30-g3-nuts"
MAP_DIR = PROJECT_ROOT / "results" / "KILOGAS066" / "kinuv-KGAS066-uvsign-map"
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
CUBE_30 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
CORNER_TITLE = (
    "066 NUTS PA 199.73; 6 sampled; not calibrated; not S2 Laplace; "
    "do not quote inner dV/dr"
)
PNG_NAMES = (
    "corner.png",
    "leftover_chi2.png",
    "moments.png",
    "spectra.png",
    "pv_major.png",
    "pv_minor.png",
)


def mean_params(rec: dict) -> dict[str, float]:
    draws = np.asarray(rec["draws"], dtype=np.float64)
    if draws.ndim == 3:
        draws = draws.reshape(-1, draws.shape[-1])
    mean = draws.mean(axis=0)
    return {n: float(v) for n, v in zip(PARAM_NAMES, mean)}


def _copy_pngs(src: Path, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name in PNG_NAMES:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dest / name)
            out.append(dest / name)
    return out


def write_corner(rec: dict, dest: Path, *, title: str = CORNER_TITLE) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return plot_posterior_corner(rec, dest, title=title)


def write_leftover_at_params(params: dict, dest: Path, *, data, tmpl, grid) -> dict:
    from kinuv.diagnostics.s1 import leftover_chi2
    from kinuv.infer.map import predict_binned

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    model = predict_binned(data, params, tmpl, grid, xla=True)
    b_m, per_row, vel, per_chan = leftover_chi2(data, model)
    total = float(np.sum(per_row))
    structured = leftover_velocity_structured(b_m, per_row, per_chan)
    summary = {
        "chi2_sum": total,
        "leftover_chi2_structured": structured,
        "n_row": int(np.asarray(data.vis).shape[0]),
        "n_chan": int(np.asarray(data.vis).shape[1]),
        "s": float(data.s),
        "params": {k: float(params[k]) for k in PARAM_NAMES},
        "note": (
            "NUTS-mean leftover; not MAP. Intervals not calibrated. "
            "Do not quote inner dV/dr."
        ),
        "updated_at": utc_now(),
    }
    (dest / "leftover_chi2.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(
        dest / "leftover_chi2.npz",
        baseline_m=b_m,
        chi2_row=per_row,
        vel_kms=vel,
        chi2_chan=per_chan,
    )
    plot_leftover_chi2(b_m, per_row, vel, per_chan, dest / "leftover_chi2.png")
    return summary


def write_stage_a_cube(params: dict, dest: Path, *, data, tmpl, grid) -> Path:
    """Native-channel Stage A cube at NUTS mean. Never the official MAP tree."""
    from astropy.io import fits
    from astropy.wcs import WCS

    from kinuv.constants import F_REST_CO21_HZ, freq_to_velocity_kms
    from kinuv.forward.model import sky_cube

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if MAP_DIR.resolve() in dest.resolve().parents or dest.resolve().parent == MAP_DIR.resolve():
        raise ValueError(f"refusing to write a cube under the official MAP tree: {dest}")
    freqs = data.freqs_native
    vel = freq_to_velocity_kms(freqs)
    cube = sky_cube(
        template=tmpl,
        grid=grid,
        freqs_hz=freqs,
        flux=params["flux"],
        pa_rad=np.deg2rad(params["pa_deg"]),
        vsys_kms=params["vsys_kms"],
        dx_arcsec=params["dx_arcsec"],
        dy_arcsec=params["dy_arcsec"],
        gas_sigma_kms=params["gas_sigma_kms"],
        v0_kms=params["v0_kms"],
        r_t_arcsec=params["r_t_arcsec"],
    )
    ico_hdr = fits.getheader(ICO) if ICO.is_file() else {}
    w = WCS(naxis=3)
    w.wcs.crpix = [grid.nx // 2 + 1, grid.ny // 2 + 1, 1.0]
    cdelt = float(grid.cell_arcsec) / 3600.0
    dv = float(np.median(np.diff(vel))) if vel.size > 1 else 1.0
    w.wcs.cdelt = np.array([-cdelt, cdelt, dv])
    w.wcs.crval = [
        float(ico_hdr.get("CRVAL1", 0.0)),
        float(ico_hdr.get("CRVAL2", 0.0)),
        float(vel[0]),
    ]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "VRAD"]
    w.wcs.cunit = ["deg", "deg", "km/s"]
    w.wcs.specsys = "LSRK"
    arr = np.flip(np.moveaxis(np.asarray(cube, dtype=np.float32), 2, 0), axis=2)
    hdr = w.to_header()
    hdr["BUNIT"] = "Jy/pixel"
    hdr["RESTFRQ"] = (float(F_REST_CO21_HZ), "CO(2-1) rest frequency [Hz]")
    hdr["OBJECT"] = "KGAS066"
    hdr["ORIGIN"] = "kinUV Stage A NUTS-mean sky_cube"
    hdr["V0"] = (float(params["v0_kms"]), "km/s")
    hdr["RT"] = (float(params["r_t_arcsec"]), "arcsec")
    fits.PrimaryHDU(data=arr, header=hdr).writeto(dest, overwrite=True)
    return dest


def write_imaging_plots(geom_json: Path, model_cube: Path, out_dir: Path) -> None:
    scripts = REPO / "scripts"
    sys.path.insert(0, str(scripts))
    from plot_stage_b_vs_imaging import main as imaging_main

    imaging_main(
        [
            "--stage-a",
            str(geom_json),
            "--model-cube",
            str(model_cube),
            "--out-dir",
            str(out_dir),
            "--matched-fits",
            str(out_dir / "model_on_10kms.fits"),
        ]
    )


def write_nuts_product_plots(
    rec: dict,
    run_dir,
    *,
    artifact_dir=ARTIFACT_G3,
    data=None,
    tmpl=None,
    grid=None,
    imaging: bool = True,
    leftover: bool = True,
):
    """Corner + leftover + imaging at NUTS mean. FITS stay in run_dir/plots."""
    run_dir = Path(run_dir)
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    written = {"corner": str(write_corner(rec, plots / "corner.png"))}
    params = mean_params(rec)
    geom_path = plots / "nuts_mean_params.json"
    geom_path.write_text(json.dumps(params, indent=2) + "\n")
    written["mean_params"] = str(geom_path)

    if leftover or imaging:
        if data is None or tmpl is None or grid is None:
            from kinuv.forward.sb import load_sb_template
            from kinuv.infer.map import image_grid_for_vis
            from kinuv.io.vis import load_kgas066

            data = load_kgas066(NPZ, cube_path=CUBE_30 if CUBE_30.is_file() else None)
            grid = image_grid_for_vis(data)
            tmpl = load_sb_template(grid, ico_path=ICO if ICO.is_file() else None)

    if leftover:
        leftover_rec = write_leftover_at_params(params, plots, data=data, tmpl=tmpl, grid=grid)
        written["leftover"] = leftover_rec
        rec["leftover_chi2_structured"] = bool(leftover_rec["leftover_chi2_structured"])
        rec["chi2_nuts_mean"] = float(leftover_rec["chi2_sum"])
        post = run_dir / "posteriors"
        if post.is_dir():
            write_json(post / "kgas066_nuts.json", rec)
            write_json(
                post / "summary.json",
                {k: rec[k] for k in rec if k != "draws"},
            )

    if imaging:
        cube = write_stage_a_cube(
            params, plots / "stage_a_nuts_mean.fits", data=data, tmpl=tmpl, grid=grid
        )
        write_imaging_plots(geom_path, cube, plots)
        written["cube"] = str(cube)

    post = run_dir / "posteriors"
    if post.is_dir() and artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for name in ("kgas066_nuts.json", "summary.json"):
            src = post / name
            if src.is_file():
                shutil.copy2(src, artifact_dir / name)
        for extra in ("leftover_chi2.json", "leftover_chi2.npz", "nuts_mean_params.json"):
            src = plots / extra
            if src.is_file():
                shutil.copy2(src, artifact_dir / extra)
        written["artifact_pngs"] = [str(p) for p in _copy_pngs(plots, artifact_dir)]
    return written
