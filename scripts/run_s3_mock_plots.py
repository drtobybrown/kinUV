#!/usr/bin/env python3
"""Mock galaxy benchmark: 5-column PV + rotation curves with kinUV visibility recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib.gridspec import GridSpec

from kinuv.diagnostics.imaging import masked_moments, pv_diagram, spectral_axis_kms
from kinuv.diagnostics.s1 import (
    CANFAR_CUBE_10,
    CANFAR_ICO,
    CANFAR_NPZ,
    dirty_cube_from_truth,
    inner_slope_arctan,
    r_eval_arcsec,
)
from kinuv.diagnostics.style import (
    COLOUR,
    apply_style,
    cbar,
    imshow_masked,
    intensity_cmap,
    panel_letter,
    residual_cmap,
    save_fig,
    sequential_clim,
    sky_extent_arcsec,
    symmetric_clim,
    vsys_line,
)
from run_s3_live_plots import _five_col_moments
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import sky_to_galaxy
from kinuv.infer.map import image_grid_for_vis
from kinuv.io.vis import load_kgas066, radio_to_optical_kms
from kinuv.template.wiener import k_to_jy_per_beam
from kinuv.constants import C_LIGHT_KM_S, F_REST_CO21_HZ

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
MOCK = DEST / "mock_controlled"
LENGTH_ARCSEC = 16.0
V_PV_LO, V_PV_HI = 8050.0, 8550.0


def _kinms_jy_beam_to_k(jy_cube, hdr, vel):
    rest = float(hdr.get("RESTFRQ", F_REST_CO21_HZ))
    nu = rest / (1.0 + float(np.median(vel)) / C_LIGHT_KM_S)
    bmaj = float(hdr["BMAJ"]) * 3600.0
    bmin = float(hdr["BMIN"]) * 3600.0
    scale = float(k_to_jy_per_beam(1.0, nu, bmaj, bmin))
    return np.asarray(jy_cube, dtype=np.float64) / scale


def _major_axis_gradient(m1, m0, hdr, pa_deg, i_deg, vsys_opt):
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    x = (np.arange(nx) + 1.0 - float(hdr["CRPIX1"])) * float(hdr["CDELT1"]) * 3600.0
    y = (np.arange(ny) + 1.0 - float(hdr["CRPIX2"])) * float(hdr["CDELT2"]) * 3600.0
    east = -x if float(hdr["CDELT1"]) < 0.0 else x
    xe, yn = np.meshgrid(east, y, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, np.radians(pa_deg), np.radians(i_deg))
    r = np.hypot(xg, yg)
    sini = np.sin(np.radians(i_deg))
    cos_th = np.divide(xg, r, out=np.zeros_like(r), where=r > 0)
    vc = np.where(np.abs(cos_th) > 0.2, (m1 - vsys_opt) / (sini * cos_th), np.nan)
    disk = np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0)) & (np.abs(yg) < 0.5)
    bins = np.linspace(0.1, 4.0, 40)
    prof_r, prof_v = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = disk & (r >= lo) & (r < hi)
        if np.sum(m) >= 3:
            prof_r.append(0.5 * (lo + hi))
            prof_v.append(float(np.nanmedian(vc[m])))
    return np.asarray(prof_r), np.asarray(prof_v)


def _kinuv_cube_from_fit(fitted, truth):
    data = load_kgas066(CANFAR_NPZ, cube_path=CANFAR_CUBE_10)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=CANFAR_ICO)
    ico_hdr = fits.getheader(CANFAR_ICO)
    img_hdr = fits.getheader(CANFAR_CUBE_10)
    p = dict(truth)
    p.update(fitted)
    p["i_deg"] = float(truth["i_deg"])
    matched, _, _, _ = dirty_cube_from_truth(p, tmpl, grid, data.freqs_native, ico_hdr, img_hdr)
    return matched


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock-dir", type=Path, default=MOCK)
    args = p.parse_args(argv)
    mock = Path(args.mock_dir)
    summary = json.loads((mock / "summary.json").read_text())
    truth = summary["truth"]
    pa = float(truth["pa_deg"])
    vsys_opt = float(radio_to_optical_kms(float(truth["vsys_kms"])))
    i_deg = float(truth["i_deg"])

    cube_path = Path(summary["mock_cube"])
    hdr = fits.getheader(cube_path)
    data = np.asarray(fits.getdata(cube_path), dtype=np.float64)
    vel = spectral_axis_kms(hdr)
    dv = abs(float(hdr["CDELT3"]))
    ra = float(hdr.get("CRVAL1", 0.0))
    dec = float(hdr.get("CRVAL2", 0.0))

    kinuv = summary["kinuv_mock"]["fitted"]
    kinms = summary["kinms_mock"]["fitted"]
    kinuv_k = _kinuv_cube_from_fit(kinuv, truth)

    kinms_cube = mock / "kinms_mock" / "model_cube.fits"
    kinms_k = _kinms_jy_beam_to_k(fits.getdata(kinms_cube), hdr, vel) if kinms_cube.is_file() else data * 0

    m0_d, m1_d, m2_d = masked_moments(data, vel, data > 0, dv)
    m0_kuv, m1_kuv, m2_kuv = masked_moments(kinuv_k, vel, data > 0, dv)
    m0_k, m1_k, m2_k = masked_moments(kinms_k, vel, data > 0, dv)

    pv_d, off = pv_diagram(data, hdr, ra, dec, pa, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0)
    pv_kuv, _ = pv_diagram(kinuv_k, hdr, ra, dec, pa, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0)
    pv_k, _ = pv_diagram(kinms_k, hdr, ra, dec, pa, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0)

    r_eval = r_eval_arcsec()
    slope_truth = inner_slope_arctan(truth["v0_kms"], truth["r_t_arcsec"], r_eval)
    slope_kinuv = inner_slope_arctan(kinuv["v0_kms"], kinuv["r_t_arcsec"], r_eval)
    slope_kinms_param = inner_slope_arctan(kinms["v0_kms"], kinms["r_t_arcsec"], r_eval)
    r_m1, vc_m1 = _major_axis_gradient(m1_k, m0_k, hdr, pa, i_deg, vsys_opt)
    inner = (r_m1 > 0.15) & (r_m1 < 1.2) & np.isfinite(vc_m1)
    if np.sum(inner) >= 3:
        slope_kinms_m1 = float(np.polyfit(r_m1[inner], vc_m1[inner], 1)[0])
    else:
        slope_kinms_m1 = float("nan")

    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    fig = plt.figure(figsize=(16.8, 10.8))
    gs = GridSpec(
        3,
        7,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.055, 1.0, 1.0, 0.055],
        height_ratios=[3.15, 2.15, 0.72],
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.05,
        hspace=0.38,
        wspace=0.18,
    )
    extent = [off[0], off[-1], max(V_PV_LO, vel[0]), min(V_PV_HI, vel[-1])]
    titles = (
        "Synthetic data",
        "kinUV vis recovery",
        "KinMS cube fit",
        "Data − kinUV",
        "Data − KinMS",
    )
    col_of = (0, 1, 2, 4, 5)
    images = (pv_d, pv_kuv, pv_k, pv_d - pv_kuv, pv_d - pv_k)
    vmin_i, vmax_i = sequential_clim(pv_d, pv_kuv, pv_k)
    ims_dm = []
    ims_r = []
    for j, (title, img, col) in enumerate(zip(titles, images, col_of)):
        ax = fig.add_subplot(gs[0, col])
        if j < 3:
            im = imshow_masked(ax, img, extent, vmin_i, vmax_i, intensity_cmap(), aspect="auto")
            ims_dm.append(im)
        else:
            rv0, rv1 = symmetric_clim(img)
            im = imshow_masked(ax, img, extent, rv0, rv1, residual_cmap(), aspect="auto")
            ims_r.append(im)
        ax.set_title(title)
        ax.set_xlabel("Offset (arcsec)")
        if j == 0:
            ax.set_ylabel("Optical velocity (km/s, LSRK)")
        else:
            ax.tick_params(labelleft=False)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", direction="in", top=True, right=True)
        vsys_line(ax, vsys_opt, orientation="h")
        panel_letter(ax, "abcde"[j])

    cbar(fig, ims_dm[0], "K", cax=fig.add_subplot(gs[0, 3]))
    cbar(fig, ims_r[0], "K", cax=fig.add_subplot(gs[0, 6]))

    ax = fig.add_subplot(gs[1, :])
    r_d, vc_d = _major_axis_gradient(m1_d, m0_d, hdr, pa, i_deg, vsys_opt)
    r_kuv, vc_kuv = _major_axis_gradient(m1_kuv, m0_kuv, hdr, pa, i_deg, vsys_opt)
    r_k, vc_k = _major_axis_gradient(m1_k, m0_k, hdr, pa, i_deg, vsys_opt)
    r_line = np.linspace(0.05, 3.5, 100)
    rt, v0 = float(truth["r_t_arcsec"]), float(truth["v0_kms"])
    vc_t = v0 * (2.0 / np.pi) * np.arctan(r_line / rt)
    ax.plot(r_line, vc_t, color=COLOUR["data"], lw=2, label="Truth V_c(R)")
    ax.plot(
        r_line,
        float(kinuv["v0_kms"]) * (2.0 / np.pi) * np.arctan(r_line / float(kinuv["r_t_arcsec"])),
        "--",
        color=COLOUR["model"],
        lw=1.5,
        label="kinUV V_c(R)",
    )
    ax.plot(
        r_line,
        float(kinms["v0_kms"]) * (2.0 / np.pi) * np.arctan(r_line / float(kinms["r_t_arcsec"])),
        "--",
        color="#C44E52",
        lw=1.5,
        label="KinMS parametric V_c(R)",
    )
    ax.plot(r_d, vc_d, "o", ms=3.5, color=COLOUR["data"], label="Data M1 → V_c")
    ax.plot(r_kuv, vc_kuv, ".", ms=5, color=COLOUR["model"], alpha=0.75, label="kinUV M1 → V_c")
    ax.plot(r_k, vc_k, "s", ms=3.5, color="#C44E52", label="KinMS M1 → V_c")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("V_c (km/s)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="lower right", ncol=2)
    panel_letter(ax, "f")

    rt_inf = float(kinms["r_t_arcsec"]) / float(truth["r_t_arcsec"])
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    ax_txt.text(
        0.5,
        0.55,
        (
            f"Truth:  r_t = {truth['r_t_arcsec']:.3f}\"   V_0 = {truth['v0_kms']:.1f} km/s   "
            f"(dV/dr)_0 = {slope_truth:.1f} km/s/arcsec\n"
            f"kinUV (vis):  r_t = {kinuv['r_t_arcsec']:.3f}\"   V_0 = {kinuv['v0_kms']:.1f} km/s   "
            f"(dV/dr)_0 = {slope_kinuv:.1f} km/s/arcsec\n"
            f"KinMS (cube):  r_t = {kinms['r_t_arcsec']:.3f}\"   V_0 = {kinms['v0_kms']:.1f} km/s   "
            f"r_t inflation = {rt_inf:.2f}x"
        ),
        ha="center",
        va="center",
        fontsize=9,
        family="DejaVu Sans",
        color=COLOUR["text"],
        transform=ax_txt.transAxes,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": COLOUR["zero"],
            "linewidth": 0.6,
        },
    )
    fig.suptitle("Controlled mock benchmark  ·  truth r_t = 0.25 arcsec, V_0 = 250 km/s")
    out = mock / "mock_benchmark.png"
    save_fig(fig, out, dpi=300)

    sky_extent = sky_extent_arcsec(hdr)
    beam = (
        float(hdr["BMAJ"]) * 3600.0,
        float(hdr["BMIN"]) * 3600.0,
        float(hdr["BPA"]),
    )
    centre = (float(truth.get("dx_arcsec", 0.0)), float(truth.get("dy_arcsec", 0.0)))
    mom_out = mock / "mock_moments_comparison.png"
    _five_col_moments(
        {"mom0": m0_d, "mom1": m1_d, "mom2": m2_d},
        {"mom0": m0_kuv, "mom1": m1_kuv, "mom2": m2_kuv},
        {"mom0": m0_k, "mom1": m1_k, "mom2": m2_k},
        sky_extent,
        vsys_opt,
        beam,
        centre,
        mom_out,
        "KinMS (sbProf)",
        headers=(
            "Mock Synthetic Data",
            "kinUV Recovery",
            "KinMS (sbProf) Recovery",
            "Data − kinUV Residual",
            "Data − KinMS Residual",
        ),
        title="Controlled mock  ·  Moment 0 / 1 / 2  ·  kinUV vis vs KinMS (sbProf)",
        footer="east left, north up  ·  optical LSRK  ·  identical clim per row  ·  restoring beam on M0 data",
    )

    rec = {
        "status": "wrote_mock_figure",
        "figure": str(out),
        "moments": str(mom_out),
        "truth_inner_slope": slope_truth,
        "kinuv_inner_slope": slope_kinuv,
        "kinms_parametric_slope": slope_kinms_param,
        "kinms_m1_slope": slope_kinms_m1,
    }
    (mock / "plots.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
