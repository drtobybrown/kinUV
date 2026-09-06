#!/usr/bin/env python3
"""Publication PV/moment panels: kinUV (vis MAP) vs KinMS (cube fit).

Follows docs/diagnostics/plotting.md and stage-b-vs-imaging.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib.gridspec import GridSpec

from kinuv.constants import C_LIGHT_KM_S, F_REST_CO21_HZ
from kinuv.diagnostics.imaging import (
    masked_moments,
    match_model_to_imaging,
    pv_diagram,
    spectral_axis_kms,
)
from kinuv.diagnostics.style import (
    COLOUR,
    CROP_ARCSEC,
    apply_style,
    beam_ellipse,
    cbar,
    format_sky_ax,
    imshow_masked,
    intensity_cmap,
    panel_letter,
    residual_cmap,
    save_fig,
    sequential_clim,
    sky_extent_arcsec,
    symmetric_clim,
    velocity_cmap,
    vsys_line,
)
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.io.vis import load_kgas066, radio_to_optical_kms
from kinuv.geometry import inclination_deg, sky_to_galaxy
from kinuv.diagnostics.s1 import inner_slope_arctan, r_eval_arcsec
from kinuv.template.wiener import k_to_jy_per_beam
from kinuv.runner.plots import write_stage_a_cube

REPO = Path(__file__).resolve().parents[1]
ROOT_10KMS = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms"
)
MAP_DIR = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/production/KGAS066/"
    "kinuv-KGAS066-uvsign-map"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
LENGTH_ARCSEC = 16.0


def _major_axis_vc(m1, m0, hdr, pa_deg, i_deg, vsys_opt, dx=0.0, dy=0.0):
    nx = int(hdr["NAXIS1"])
    ny = int(hdr["NAXIS2"])
    x = (np.arange(nx) + 1.0 - float(hdr["CRPIX1"])) * float(hdr["CDELT1"]) * 3600.0
    y = (np.arange(ny) + 1.0 - float(hdr["CRPIX2"])) * float(hdr["CDELT2"]) * 3600.0
    east = -x if float(hdr["CDELT1"]) < 0.0 else x
    xe, yn = np.meshgrid(east - dx, y - dy, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, np.radians(pa_deg), np.radians(i_deg))
    r = np.hypot(xg, yg)
    sini = np.sin(np.radians(i_deg))
    cos_th = np.divide(xg, r, out=np.zeros_like(r), where=r > 0)
    vc = np.where(np.abs(cos_th) > 0.2, (m1 - vsys_opt) / (sini * cos_th), np.nan)
    disk = np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0)) & (np.abs(yg) < 0.5)
    bins = np.linspace(0.1, 6.0, 50)
    prof_r, prof_v = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = disk & (r >= lo) & (r < hi)
        if np.sum(m) >= 3:
            prof_r.append(0.5 * (lo + hi))
            prof_v.append(float(np.nanmedian(vc[m])))
    return np.asarray(prof_r), np.asarray(prof_v)


def _arctan_vc(r, v0, rt):
    r = np.asarray(r, dtype=np.float64)
    return float(v0) * (2.0 / np.pi) * np.arctan(r / max(float(rt), 1e-3))


def _ticks(ax):
    from matplotlib.ticker import AutoMinorLocator

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _param_box(ax, kinuv_ann, kinms_ann):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        (
            "kinUV (official MAP): "
            f"r_t = {kinuv_ann['r_t_arcsec']:.3f}\"   V_0 = {kinuv_ann['v0_kms']:.0f} km/s   "
            f"PA = {kinuv_ann['pa_deg']:.1f} deg   i = {kinuv_ann['i_deg']:.1f} deg\n"
            "KinMS (sbProf): "
            f"r_t = {kinms_ann.get('r_t_arcsec', float('nan')):.3f}\"   "
            f"V_0 = {kinms_ann.get('v0_kms', float('nan')):.0f} km/s   "
            f"PA = {kinms_ann.get('pa_deg', float('nan')):.1f} deg   "
            f"i = {kinms_ann.get('i_deg', float('nan')):.1f} deg"
        ),
        ha="center",
        va="center",
        fontsize=9,
        family="DejaVu Sans",
        color=COLOUR["text"],
        transform=ax.transAxes,
    )


def _kinms_jy_beam_to_k(jy_cube, data_hdr, vel_kms):
    """KinMS Jy/beam cube (nv, ny, nx) → Kelvin on the imaging axis."""
    rest = float(data_hdr.get("RESTFRQ", F_REST_CO21_HZ))
    v_mid = float(np.median(vel_kms))
    nu = rest / (1.0 + v_mid / C_LIGHT_KM_S)
    bmaj = float(data_hdr["BMAJ"]) * 3600.0
    bmin = float(data_hdr["BMIN"]) * 3600.0
    scale = float(k_to_jy_per_beam(1.0, nu, bmaj, bmin))
    return np.asarray(jy_cube, dtype=np.float64) / scale


def _five_col_pv_row(
    gs_row,
    v,
    offsets,
    data_pv,
    kinuv_pv,
    kinms_pv,
    row_title,
    vsys,
    letters,
    kinms_label="KinMS",
    *,
    titles=True,
    xlabel=True,
):
    panels = (
        ("Data", data_pv),
        ("kinUV", kinuv_pv),
        (kinms_label, kinms_pv),
        ("Data − kinUV", data_pv - kinuv_pv),
        (f"Data − {kinms_label}", data_pv - kinms_pv),
    )
    vmin, vmax = sequential_clim(data_pv, kinuv_pv, kinms_pv)
    extent = [offsets[0], offsets[-1], v[0], v[-1]]
    ims_dm = []
    ims_r = []
    axes = []
    for j, (name, img) in enumerate(panels):
        ax = gs_row[j]
        axes.append(ax)
        if j < 3:
            im = imshow_masked(ax, img, extent, vmin, vmax, intensity_cmap(), aspect="auto")
            ims_dm.append(im)
        else:
            rv0, rv1 = symmetric_clim(img)
            im = imshow_masked(ax, img, extent, rv0, rv1, residual_cmap(), aspect="auto")
            ims_r.append(im)
        if titles:
            ax.set_title(name)
        if j == 0:
            ax.set_ylabel(f"{row_title}\nOptical velocity (km/s, LSRK)")
        else:
            ax.tick_params(labelleft=False)
        if xlabel:
            ax.set_xlabel("Offset (arcsec; receding +)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        _ticks(ax)
        vsys_line(ax, vsys, orientation="h")
        panel_letter(ax, letters[j])
    return ims_dm, ims_r, axes


def _pv_major_minor_figure(major, minor, vsys, out, kinms_label="KinMS", kinuv_ann=None, kinms_ann=None):
    apply_style()
    import matplotlib.pyplot as plt

    (v_m, off_m, d_m, k_m, s_m, t_m) = major
    (v_n, off_n, d_n, k_n, s_n, t_n) = minor
    fig = plt.figure(figsize=(16.4, 11.4))
    gs = GridSpec(
        5,
        5,
        figure=fig,
        height_ratios=[0.48, 3.0, 3.0, 0.18, 0.32],
        left=0.08,
        right=0.98,
        top=0.94,
        bottom=0.04,
        wspace=0.20,
        hspace=0.28,
    )
    banner = fig.add_subplot(gs[0, :])
    if kinuv_ann and kinms_ann:
        _param_box(banner, kinuv_ann, kinms_ann)
    else:
        banner.axis("off")
    row_major = [fig.add_subplot(gs[1, j]) for j in range(5)]
    row_minor = [fig.add_subplot(gs[2, j]) for j in range(5)]
    ims_dm, ims_r, _ = _five_col_pv_row(
        row_major, v_m, off_m, d_m, k_m, s_m, t_m, vsys, "abcde", kinms_label, titles=True, xlabel=False
    )
    dm2, rr2, _ = _five_col_pv_row(
        row_minor, v_n, off_n, d_n, k_n, s_n, t_n, vsys, "fghij", kinms_label, titles=False, xlabel=True
    )
    ims_dm.extend(dm2)
    ims_r.extend(rr2)
    cbar(fig, ims_dm[0], "K", cax=fig.add_subplot(gs[3, 0:3]), orientation="horizontal")
    cbar(fig, ims_r[0], "K", cax=fig.add_subplot(gs[3, 3:5]), orientation="horizontal")
    foot = fig.add_subplot(gs[4, :])
    foot.axis("off")
    foot.text(
        0.5,
        0.4,
        f"v_sys = {vsys:.1f} km/s (optical LSRK)  ·  east left, north up",
        ha="center",
        va="center",
        color=COLOUR["muted"],
        fontsize=8,
        transform=foot.transAxes,
    )
    fig.suptitle(f"KGAS066 position-velocity  ·  vis MAP kinUV vs {kinms_label} cube fit")
    save_fig(fig, out, dpi=300)


def _rotation_curves_figure(
    data_m1,
    data_m0,
    kinuv_m1,
    kinuv_m0,
    kinms_m1,
    kinms_m0,
    hdr,
    vsys,
    kinuv_fit,
    kinms_fit,
    pa,
    i_kinuv,
    i_kinms,
    dx,
    dy,
    out,
):
    apply_style()
    import matplotlib.pyplot as plt

    r_d, vc_d = _major_axis_vc(data_m1, data_m0, hdr, pa, i_kinuv, vsys, dx, dy)
    r_k, vc_k = _major_axis_vc(kinuv_m1, kinuv_m0, hdr, pa, i_kinuv, vsys, dx, dy)
    r_s, vc_s = _major_axis_vc(kinms_m1, kinms_m0, hdr, pa, i_kinms, vsys, dx, dy)
    r_line = np.linspace(0.05, 6.0, 120)
    sini_k = np.sin(np.radians(i_kinuv))
    sini_s = np.sin(np.radians(i_kinms))
    vc_kinuv = _arctan_vc(r_line, kinuv_fit["v0_kms"], kinuv_fit["r_t_arcsec"])
    vc_kinms = _arctan_vc(r_line, kinms_fit["v0_kms"], kinms_fit["r_t_arcsec"])
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(r_line, vc_kinuv, "--", color=COLOUR["model"], lw=1.6, label="kinUV intrinsic V_c(R)")
    ax.plot(r_line, vc_kinuv * sini_k, "-", color=COLOUR["model"], lw=1.2, label=f"kinUV V_c sin i (i={i_kinuv:.1f} deg)")
    ax.plot(r_line, vc_kinms, "--", color="#C44E52", lw=1.6, label="KinMS intrinsic V_c(R)")
    ax.plot(r_line, vc_kinms * sini_s, "-", color="#C44E52", lw=1.2, label=f"KinMS V_c sin i (i={i_kinms:.1f} deg)")
    ax.plot(r_d, vc_d, "o", ms=3.5, color=COLOUR["data"], label="Data M1 → V_c (deprojected)")
    ax.plot(r_k, vc_k, ".", ms=5, color=COLOUR["model"], alpha=0.75, label="kinUV M1 → V_c")
    ax.plot(r_s, vc_s, ".", ms=5, color="#C44E52", alpha=0.75, label="KinMS M1 → V_c")
    ax.set_xlabel("Galactocentric radius (arcsec)")
    ax.set_ylabel("Velocity (km/s)")
    _ticks(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_title("KGAS066 rotation curves: dashed = intrinsic V_c(R); solid = V_c sin i")
    fig.subplots_adjust(right=0.72, left=0.10, top=0.90, bottom=0.14)
    save_fig(fig, out, dpi=300)


def _five_col_moments(
    data,
    kinuv,
    kinms,
    extent,
    vsys,
    beam,
    centre,
    out,
    kinms_label="KinMS",
    *,
    headers=None,
    title=None,
    footer=None,
):
    apply_style()
    import matplotlib.pyplot as plt

    rows = (
        ("mom0", "Moment 0", "K km/s", intensity_cmap(), False),
        ("mom1", "Moment 1", "v − vsys (km/s)", velocity_cmap(), True),
        ("mom2", "Moment 2", "km/s", intensity_cmap(), False),
    )
    fig = plt.figure(figsize=(17.2, 10.6))
    gs = GridSpec(
        3,
        7,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.075, 0.075],
        left=0.11,
        right=0.97,
        top=0.90,
        bottom=0.07,
        wspace=0.22,
        hspace=0.20,
    )
    bmaj, bmin, bpa = beam
    cx, cy = centre
    crop = CROP_ARCSEC
    letters = "abcdefghijklmno"
    li = 0
    headers = headers or (
        "Data",
        "kinUV",
        kinms_label,
        "Data − kinUV",
        f"Data − {kinms_label}",
    )
    first_data_ax = None
    for i, (key, row_name, unit, cmap, is_vel) in enumerate(rows):
        d = data[key]
        kuv = kinuv[key]
        kms = kinms[key]
        if is_vel:
            d, kuv, kms = d - vsys, kuv - vsys, kms - vsys
            vmin, vmax = symmetric_clim(d, kuv, kms)
        else:
            vmin, vmax = sequential_clim(d, kuv, kms)
        trip = ((d, vmin, vmax), (kuv, vmin, vmax), (kms, vmin, vmax))
        r0, r1 = d - kuv, d - kms
        resid = ((r0, *symmetric_clim(r0)), (r1, *symmetric_clim(r1)))
        im_pair = None
        im_res = None
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            if first_data_ax is None:
                first_data_ax = ax
            img, v0, v1 = trip[j]
            im_pair = imshow_masked(ax, img, extent, v0, v1, cmap)
            format_sky_ax(ax, crop, centre, xlabel=(i == 2), ylabel=(j == 0))
            if i == 0:
                ax.set_title(headers[j])
            if j == 0:
                ax.text(
                    -0.38,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                )
            panel_letter(ax, letters[li])
            li += 1
        for j, (img, rv0, rv1) in enumerate(resid):
            ax = fig.add_subplot(gs[i, 3 + j])
            im_res = imshow_masked(ax, img, extent, rv0, rv1, residual_cmap())
            format_sky_ax(ax, crop, centre, xlabel=(i == 2), ylabel=False)
            if i == 0:
                ax.set_title(headers[3 + j])
            panel_letter(ax, letters[li])
            li += 1
        cbar(fig, im_pair, unit, cax=fig.add_subplot(gs[i, 5]))
        cbar(fig, im_res, unit, cax=fig.add_subplot(gs[i, 6]))
    beam_ellipse(first_data_ax, bmaj, bmin, bpa, (cx + crop - 2.1, cy - crop + 2.1))
    fig.suptitle(title or f"KGAS066  ·  vis MAP kinUV vs {kinms_label}")
    fig.text(
        0.50,
        0.015,
        footer
        or "east left, north up  ·  optical LSRK  ·  KinMS SB = elliptical-annulus sbProf",
        ha="center",
        fontsize=8,
        color=COLOUR["muted"],
    )
    save_fig(fig, out, dpi=300)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live-dir", type=Path, required=True)
    p.add_argument("--kinms-cube", type=Path, required=True)
    p.add_argument("--data-cube", type=Path, default=ROOT_10KMS / "KGAS66_clipped_cube.fits")
    p.add_argument("--mask-cube", type=Path, default=ROOT_10KMS / "KGAS66_mask_cube.fits")
    p.add_argument("--kinms-label", type=str, default="KinMS")
    p.add_argument("--kinms-json", type=Path, default=None)
    args = p.parse_args(argv)
    kinms_label = str(args.kinms_label)
    live = Path(args.live_dir)
    live.mkdir(parents=True, exist_ok=True)

    params = json.loads((MAP_DIR / "stage_a_map.json").read_text())
    pa = float(params["pa_deg"])
    vsys_opt = float(radio_to_optical_kms(float(params["vsys_kms"])))
    dx, dy = float(params["dx_arcsec"]), float(params["dy_arcsec"])
    ra = float(fits.getheader(args.data_cube).get("CRVAL1", 0.0))
    dec = float(fits.getheader(args.data_cube).get("CRVAL2", 0.0))

    data_hdu = fits.open(args.data_cube)[0]
    mask_hdu = fits.open(args.mask_cube)[0]
    data = np.asarray(data_hdu.data, dtype=np.float64)
    mask3d = np.asarray(mask_hdu.data, dtype=np.float64) > 0.5
    hdr = data_hdu.header
    vel = spectral_axis_kms(hdr)
    dv = abs(float(hdr["CDELT3"]))
    extent = sky_extent_arcsec(hdr)
    beam = (
        float(hdr["BMAJ"]) * 3600.0,
        float(hdr["BMIN"]) * 3600.0,
        float(hdr["BPA"]),
    )
    centre = (dx, dy)

    vis = load_kgas066(NPZ, cube_path=args.data_cube)
    grid = image_grid_for_vis(vis)
    tmpl = load_sb_template(grid, ico_path=ROOT_10KMS / "KGAS66_Ico_K_kms-1.fits")
    kinuv_fits = live / "kinuv_stage_a_model.fits"
    write_stage_a_cube(params, kinuv_fits, data=vis, tmpl=tmpl, grid=grid)
    kinuv_k, _, _ = match_model_to_imaging(
        np.asarray(fits.getdata(kinuv_fits), dtype=np.float64),
        fits.getheader(kinuv_fits),
        hdr,
    )

    kinms_jy = np.asarray(fits.getdata(args.kinms_cube), dtype=np.float64)
    kinms_k = _kinms_jy_beam_to_k(kinms_jy, hdr, vel)

    m0, m1, m2 = masked_moments(data, vel, mask3d, dv)
    k0, k1, k2 = masked_moments(kinuv_k, vel, mask3d, dv)
    s0, s1, s2 = masked_moments(kinms_k, vel, mask3d, dv)

    kinms_fit = {}
    if args.kinms_json and Path(args.kinms_json).is_file():
        kinms_fit = json.loads(Path(args.kinms_json).read_text()).get("fitted") or {}
    elif (live / "kinms_best" / "kinms_fit_result.json").is_file():
        kinms_fit = json.loads((live / "kinms_best" / "kinms_fit_result.json").read_text()).get("fitted") or {}

    kinuv_ann = {
        "r_t_arcsec": float(params["r_t_arcsec"]),
        "v0_kms": float(params["v0_kms"]),
        "pa_deg": float(params["pa_deg"]),
        "i_deg": float(inclination_deg()),
    }

    pv_out = live / "pv_comparison_real.png"
    mom_out = live / "moments_comparison_real.png"
    rot_out = live / "rotation_curves_real.png"

    pv_data_maj, off_maj = pv_diagram(
        data, hdr, ra, dec, pa, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    )
    pv_kinuv_maj, _ = pv_diagram(
        kinuv_k, hdr, ra, dec, pa, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    )
    pv_kinms_maj, _ = pv_diagram(
        kinms_k, hdr, ra, dec, pa, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    )
    pa_min = pa + 90.0
    pv_data_min, off_min = pv_diagram(
        data, hdr, ra, dec, pa_min, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    )
    pv_kinuv_min, _ = pv_diagram(
        kinuv_k, hdr, ra, dec, pa_min, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    )
    pv_kinms_min, _ = pv_diagram(
        kinms_k, hdr, ra, dec, pa_min, LENGTH_ARCSEC, 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    )

    _pv_major_minor_figure(
        (vel, off_maj, pv_data_maj, pv_kinuv_maj, pv_kinms_maj, f"Major axis (PA {pa:.1f}°)"),
        (vel, off_min, pv_data_min, pv_kinuv_min, pv_kinms_min, f"Minor axis (PA {pa_min:.1f}°)"),
        vsys_opt,
        pv_out,
        kinms_label,
        kinuv_ann=kinuv_ann,
        kinms_ann=kinms_fit,
    )
    _five_col_moments(
        {"mom0": m0, "mom1": m1, "mom2": m2},
        {"mom0": k0, "mom1": k1, "mom2": k2},
        {"mom0": s0, "mom1": s1, "mom2": s2},
        extent, vsys_opt, beam, centre, mom_out, kinms_label,
    )
    if kinms_fit:
        _rotation_curves_figure(
            m1, m0, k1, k0, s1, s0, hdr, vsys_opt, kinuv_ann, kinms_fit,
            pa, float(kinuv_ann["i_deg"]), float(kinms_fit.get("i_deg", kinuv_ann["i_deg"])),
            dx, dy, rot_out,
        )

    rec = {
        "status": "wrote_publication_panels",
        "pv": str(pv_out),
        "moments": str(mom_out),
        "rotation_curves": str(rot_out) if kinms_fit else None,
        "vsys_optical_kms": vsys_opt,
        "kinuv_map": str(MAP_DIR),
        "kinms_cube": str(args.kinms_cube),
        "quote_inner_slope": False,
    }
    (live / "overlays.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
