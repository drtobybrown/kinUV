#!/usr/bin/env python3
"""Advanced mock diagnostics: vis adjoint residuals vs KinMS cube residuals.

Figure A images ``F^{-1}[ΔV]`` with a script-local type-1 DFT adjoint of the
881×95 residual visibilities (natural weights, no PB divide). That is not a
production ``nufft1`` and is not ``numpy.fft`` of the vis array. Figure D
contours are vis/cube χ² slices (Laplace); they are not MCMC. S2 SBC failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

from kinuv.constants import ARCSEC_TO_RAD, C_LIGHT_KM_S, Distance, F_REST_CO21_HZ
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
    vsys_line,
)
from kinuv.forward.sb import load_sb_template
from kinuv.geometry import inclination_rad, sky_to_galaxy
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.io.vis import load_kgas066, radio_to_optical_kms
from kinuv.likelihood.chi2 import chi2
from kinuv.template.wiener import k_to_jy_per_beam
from kinuv.transforms.dft import vis_uv_wavelengths

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
MOCK = DEST / "mock_controlled"
OUT = DEST / "advanced_diagnostics"
FITTERS_PY = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters/venv/bin/python")
MASK_CUBE = CANFAR_CUBE_10.parent / "KGAS66_mask_cube.fits"
KINMS_RED = "#C44E52"
CROP_CHAN = 8.0
LENGTH_ARCSEC = 16.0
V_PV_LO, V_PV_HI = 8050.0, 8550.0
CHI2_68, CHI2_95 = 2.30, 5.99


def _kinms_jy_beam_to_k(jy_cube, hdr, vel):
    rest = float(hdr.get("RESTFRQ", F_REST_CO21_HZ))
    nu = rest / (1.0 + float(np.median(vel)) / C_LIGHT_KM_S)
    bmaj = float(hdr["BMAJ"]) * 3600.0
    bmin = float(hdr["BMIN"]) * 3600.0
    scale = float(k_to_jy_per_beam(1.0, nu, bmaj, bmin))
    return np.asarray(jy_cube, dtype=np.float64) / scale


def _kinuv_cube(fitted, truth):
    data = load_kgas066(CANFAR_NPZ, cube_path=CANFAR_CUBE_10)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=CANFAR_ICO)
    ico_hdr = fits.getheader(CANFAR_ICO)
    img_hdr = fits.getheader(CANFAR_CUBE_10)
    p = dict(truth)
    p.update(fitted)
    matched, _, _, _ = dirty_cube_from_truth(
        p, tmpl, grid, data.freqs_native, ico_hdr, img_hdr
    )
    return matched


def _metric_box_text(truth, kinuv, kinms):
    r_e = r_eval_arcsec()
    s_t = inner_slope_arctan(truth["v0_kms"], truth["r_t_arcsec"], r_e)
    s_u = inner_slope_arctan(kinuv["v0_kms"], kinuv["r_t_arcsec"], r_e)
    s_k = inner_slope_arctan(kinms["v0_kms"], kinms["r_t_arcsec"], r_e)
    infl = float(kinms["r_t_arcsec"]) / float(truth["r_t_arcsec"])
    return (
        f"True r_t = {truth['r_t_arcsec']:.3f}\"   "
        f"kinUV r_t = {kinuv['r_t_arcsec']:.3f}\"   "
        f"KinMS r_t = {kinms['r_t_arcsec']:.3f}\"\n"
        f"Inner dV/dr:  {s_t:.1f}   vs   {s_u:.1f}   vs   {s_k:.1f}  km/s/arcsec\n"
        f"KinMS r_t inflation = {infl:.2f}x"
    )


def _inset(ax, text, *, loc="upper right"):
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    x = 0.98 if ha == "right" else 0.02
    y = 0.96 if va == "top" else 0.04
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=7.5,
        family="DejaVu Sans",
        color=COLOUR["text"],
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": COLOUR["zero"],
            "linewidth": 0.6,
        },
        zorder=12,
    )


def _sky_east_north(hdr):
    nx, ny = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    x = (np.arange(nx) + 1.0 - float(hdr["CRPIX1"])) * float(hdr["CDELT1"]) * 3600.0
    y = (np.arange(ny) + 1.0 - float(hdr["CRPIX2"])) * float(hdr["CDELT2"]) * 3600.0
    east = -x if float(hdr["CDELT1"]) < 0.0 else x
    return east, y


def _cutout(east, north, crop):
    ix = np.where(np.abs(east) <= crop + 0.15)[0]
    iy = np.where(np.abs(north) <= crop + 0.15)[0]
    return ix, iy


def dirty_adjoint_channel(dvis, weights, u_m, v_m, freq_hz, east, north):
    """Type-1 DFT adjoint of one residual-vis channel. Natural weights.

    Forward kernel is ``exp(-2πi (u_λ l + v_λ m))`` with ``NPZ_UV_SIGN``.
    The adjoint is ``Re Σ_k w_k ΔV_k exp(+2πi (u_λ l + v_λ m))``.
    """
    u_lam, v_lam = vis_uv_wavelengths(u_m, v_m, np.asarray([freq_hz], dtype=np.float64))
    u = np.asarray(u_lam[:, 0], dtype=np.float64)
    v = np.asarray(v_lam[:, 0], dtype=np.float64)
    wdv = np.asarray(weights, dtype=np.float64) * np.asarray(dvis, dtype=np.complex128)
    ee, nn = np.meshgrid(east, north, indexing="xy")
    l_rad = ee.ravel() * ARCSEC_TO_RAD
    m_rad = nn.ravel() * ARCSEC_TO_RAD
    phase = 2.0 * np.pi * (np.outer(u, l_rad) + np.outer(v, m_rad))
    dirty = np.real(np.exp(1j * phase).T @ wdv)
    return dirty.reshape(ee.shape)


def azimuthal_pk(img, cell_arcsec, n_bin=16):
    a = np.asarray(img, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    ny, nx = a.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    f = np.fft.fftshift(np.fft.fft2(a * win))
    p2 = (f.real ** 2 + f.imag ** 2)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=float(cell_arcsec)))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=float(cell_arcsec)))
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    kk = np.hypot(kkx, kky)
    kmax = 0.45 / float(cell_arcsec)
    edges = np.linspace(0.0, kmax, n_bin + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    pk = np.full(n_bin, np.nan)
    for i in range(n_bin):
        m = (kk >= edges[i]) & (kk < edges[i + 1])
        if np.any(m):
            pk[i] = float(np.median(p2[m]))
    hi = np.isfinite(pk) & (centres > 0.55 * kmax)
    floor = float(np.nanmedian(pk[hi])) if np.any(hi) else float(np.nanmedian(pk))
    if not np.isfinite(floor) or floor <= 0.0:
        floor = 1.0
    return centres, pk / floor


def _select_channels(vel_opt, vsys_opt, v_proj):
    targets = (
        ("approaching horn", vsys_opt - v_proj),
        ("inner approaching", vsys_opt - 0.35 * v_proj),
        ("systemic", vsys_opt),
        ("inner receding", vsys_opt + 0.35 * v_proj),
        ("receding horn", vsys_opt + v_proj),
        ("outer disk", vsys_opt + 0.82 * v_proj),
    )
    used = set()
    out = []
    for name, v0 in targets:
        idx = int(np.argmin(np.abs(vel_opt - v0)))
        if idx in used:
            for step in (1, -1, 2, -2):
                j = idx + step
                if 0 <= j < vel_opt.size and j not in used:
                    idx = j
                    break
        used.add(idx)
        out.append((name, idx, float(vel_opt[idx])))
    return out


def _mad_sigma(img):
    v = np.asarray(img, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return 1.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _scale_bar(ax, x0, y0, length, label):
    ax.plot([x0, x0 - length], [y0, y0], color=COLOUR["data"], lw=1.5, zorder=6)
    ax.text(
        x0 - 0.5 * length,
        y0 + 0.35,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color=COLOUR["data"],
        zorder=6,
    )


def _annular_profile(val, m0, hdr, pa_deg, i_deg, r_max=3.0, nbin=18):
    east, north = _sky_east_north(hdr)
    xe, yn = np.meshgrid(east, north, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, np.radians(pa_deg), np.radians(i_deg))
    r = np.hypot(xg, yg)
    disk = np.isfinite(val) & np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0))
    edges = np.linspace(0.05, r_max, nbin + 1)
    rr, yy = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = disk & (r >= lo) & (r < hi)
        if np.sum(m) >= 4:
            rr.append(0.5 * (lo + hi))
            yy.append(float(np.nanmedian(val[m])))
    return np.asarray(rr), np.asarray(yy)


def _m1_gradient(m1, m0, hdr, pa_deg, i_deg, vsys_opt):
    east, north = _sky_east_north(hdr)
    xe, yn = np.meshgrid(east, north, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, np.radians(pa_deg), np.radians(i_deg))
    r = np.hypot(xg, yg)
    sini = np.sin(np.radians(i_deg))
    cos_th = np.divide(xg, r, out=np.zeros_like(r), where=r > 0)
    vc = np.where(np.abs(cos_th) > 0.25, (m1 - vsys_opt) / (sini * cos_th), np.nan)
    disk = np.isfinite(m0) & (m0 > 0.05 * np.nanmax(m0)) & (np.abs(yg) < 0.55)
    edges = np.linspace(0.08, 3.0, 22)
    rr, vv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = disk & (r >= lo) & (r < hi) & np.isfinite(vc)
        if np.sum(m) >= 4:
            rr.append(0.5 * (lo + hi))
            vv.append(float(np.nanmedian(vc[m])))
    r_a = np.asarray(rr)
    v_a = np.asarray(vv)
    if r_a.size < 4:
        return r_a, v_a
    return r_a, np.gradient(v_a, r_a)


def _arctan_slope(v0, rt, r):
    return float(v0) * (2.0 / np.pi) * float(rt) / (float(rt) ** 2 + r ** 2)


def _chi2_slice_2d(data, tmpl, grid, base, name_x, xs, name_y, ys, *, i_scan=False):
    z = np.empty((len(ys), len(xs)), dtype=np.float64)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            p = dict(base)
            p[name_x] = float(xv)
            p[name_y] = float(yv)
            kw = {}
            if i_scan or name_x == "i_deg" or name_y == "i_deg":
                ideg = p.get("i_deg")
                if ideg is not None:
                    kw["i_rad"] = np.radians(float(ideg))
            model = predict_binned(data, p, tmpl, grid, **kw)
            z[i, j] = chi2(data.vis, model, data.weights, data.s)
    return z


def _ensure_kinms_slices(out_dir: Path) -> dict:
    dest = out_dir / "kinms_chi2_slices.json"
    if dest.is_file():
        return json.loads(dest.read_text())
    cfg = out_dir / "kinms_chi2_grid.json"
    proc = subprocess.run(
        [str(FITTERS_PY), str(REPO / "external" / "_kinms_chi2_grid_worker.py"), str(cfg)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO / "external"),
    )
    if not dest.is_file():
        raise RuntimeError(
            f"KinMS chi2 grid failed rc={proc.returncode}: {(proc.stderr or '')[-800:]}"
        )
    return json.loads(dest.read_text())


def fig_dirty_channels(ctx, path):
    apply_style()
    import matplotlib.pyplot as plt

    hdr = ctx["hdr"]
    data = ctx["data_cube"]
    kinms = ctx["kinms_k"]
    dvis = ctx["dvis"]
    vis = ctx["vis"]
    vel_opt = ctx["vel_opt"]
    vsys_opt = ctx["vsys_opt"]
    v_proj = ctx["v_proj"]
    extent = sky_extent_arcsec(hdr)
    cell = abs(float(hdr["CDELT1"])) * 3600.0
    east, north = _sky_east_north(hdr)
    ix, iy = _cutout(east, north, CROP_CHAN + 1.0)
    e_c, n_c = east[ix], north[iy]
    vis_opt = radio_to_optical_kms(vis.vel)
    chans = _select_channels(vel_opt, vsys_opt, v_proj)
    beam = (
        float(hdr["BMAJ"]) * 3600.0,
        float(hdr["BMIN"]) * 3600.0,
        float(hdr["BPA"]),
    )
    z = ctx["z"]
    bar_1kpc = ctx["dist"].kpc_to_arcsec(1.0)
    kpc_per = ctx["dist"].kpc_per_arcsec

    fig = plt.figure(figsize=(13.6, 16.4))
    gs = GridSpec(
        6,
        7,
        figure=fig,
        width_ratios=[1.0, 0.045, 1.0, 0.045, 1.0, 0.045, 1.20],
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.045,
        wspace=0.16,
        hspace=0.22,
    )
    letters = "abcdefghijklmnopqrstuvwxyz"
    li = 0
    headers = (
        "Mock channel (beam-convolved)",
        "kinUV dirty residual  F^{-1}[ΔV]",
        "KinMS residual  Data − model",
        "P(k) / P_noise",
    )
    for i, (name, ic, vch) in enumerate(chans):
        iv = int(np.argmin(np.abs(vis_opt - vch)))
        dat = np.asarray(data[ic], dtype=np.float64)
        kms = np.asarray(kinms[ic], dtype=np.float64)
        resid_k = dat - kms
        dirty = dirty_adjoint_channel(
            dvis[:, iv],
            vis.weights[:, iv],
            vis.u_m,
            vis.v_m,
            float(vis.freqs[iv]),
            e_c,
            n_c,
        )
        dat_c = dat[np.ix_(iy, ix)]
        kres_c = resid_k[np.ix_(iy, ix)]
        dx = float(hdr["CDELT1"]) * 3600.0
        dy = float(hdr["CDELT2"]) * 3600.0
        crp1, crp2 = float(hdr["CRPIX1"]), float(hdr["CRPIX2"])
        x_fits = (ix + 1.0 - crp1) * dx
        y_fits = (iy + 1.0 - crp2) * dy
        ext_c = (
            (ix[0] + 0.5 - crp1) * dx,
            (ix[-1] + 1.5 - crp1) * dx,
            (iy[0] + 0.5 - crp2) * dy,
            (iy[-1] + 1.5 - crp2) * dy,
        )
        xx, yy = np.meshgrid(x_fits, y_fits, indexing="xy")

        ax0 = fig.add_subplot(gs[i, 0])
        vmin, vmax = sequential_clim(dat_c)
        im0 = imshow_masked(ax0, dat_c, ext_c, vmin, vmax, intensity_cmap())
        format_sky_ax(ax0, CROP_CHAN, (0.0, 0.0), xlabel=(i == 5), ylabel=True)
        ax0.set_ylabel(f"{name}\n{vch:.0f} km/s")
        if i == 0:
            ax0.set_title(headers[0])
            beam_ellipse(
                ax0, beam[0], beam[1], beam[2], (CROP_CHAN - 1.8, -CROP_CHAN + 1.6)
            )
            _scale_bar(
                ax0, -(CROP_CHAN - 0.3), -CROP_CHAN + 1.15, 1.0, f'1" = {kpc_per:.2f} kpc'
            )
            _scale_bar(ax0, -(CROP_CHAN - 0.3), -CROP_CHAN + 2.55, bar_1kpc, "1 kpc")
        cbar(fig, im0, "K", cax=fig.add_subplot(gs[i, 1]))
        panel_letter(ax0, letters[li])
        li += 1

        ax1 = fig.add_subplot(gs[i, 2])
        rv0, rv1 = symmetric_clim(dirty)
        im1 = imshow_masked(ax1, dirty, ext_c, rv0, rv1, residual_cmap())
        format_sky_ax(ax1, CROP_CHAN, (0.0, 0.0), xlabel=(i == 5), ylabel=False)
        sig_d = _mad_sigma(dirty)
        ax1.contour(
            xx,
            yy,
            dirty,
            levels=[-4 * sig_d, -2 * sig_d, 2 * sig_d, 4 * sig_d],
            colors=COLOUR["data"],
            linewidths=0.4,
            alpha=0.55,
        )
        if i == 0:
            ax1.set_title(headers[1])
        cbar(fig, im1, "adjoint (arb.)", cax=fig.add_subplot(gs[i, 3]))
        panel_letter(ax1, letters[li])
        li += 1

        ax2 = fig.add_subplot(gs[i, 4])
        kv0, kv1 = symmetric_clim(kres_c)
        im2 = imshow_masked(ax2, kres_c, ext_c, kv0, kv1, residual_cmap())
        format_sky_ax(ax2, CROP_CHAN, (0.0, 0.0), xlabel=(i == 5), ylabel=False)
        sig_k = _mad_sigma(kres_c)
        ax2.contour(
            xx,
            yy,
            kres_c,
            levels=[-4 * sig_k, -2 * sig_k, 2 * sig_k, 4 * sig_k],
            colors=COLOUR["data"],
            linewidths=0.4,
            alpha=0.55,
        )
        if i == 0:
            ax2.set_title(headers[2])
        cbar(fig, im2, "K", cax=fig.add_subplot(gs[i, 5]))
        panel_letter(ax2, letters[li])
        li += 1

        ax3 = fig.add_subplot(gs[i, 6])
        k1, p1 = azimuthal_pk(dirty, cell)
        k2, p2 = azimuthal_pk(kres_c, cell)
        ax3.plot(k1, p1, color=COLOUR["model"], lw=1.5, label="kinUV")
        ax3.plot(k2, p2, color=KINMS_RED, lw=1.5, label="KinMS")
        ax3.axhline(1.0, color=COLOUR["zero"], lw=0.7)
        ax3.set_yscale("log")
        ax3.set_xlim(0.0, float(np.nanmax(k1)))
        ax3.set_ylim(3.0e-2, 3.0e3)
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        if i == 0:
            ax3.set_title(headers[3])
            ax3.legend(loc="upper right")
        if i == 5:
            ax3.set_xlabel("k (arcsec$^{-1}$)")
        else:
            ax3.tick_params(labelbottom=False)
        ax3.set_ylabel("P / P_noise")
        panel_letter(ax3, letters[li])
        li += 1

    fig.suptitle(
        "Controlled mock  ·  channel residuals  ·  kinUV type-1 adjoint vs KinMS image residual"
    )
    fig.text(
        0.50,
        0.012,
        "east left, north up  ·  ±2σ/±4σ MAD contours  ·  "
        "kinUV is F^{-1}[V_data−V_model], not a CLEAN residual  ·  "
        f"z={z:.4f} (from vsys)",
        ha="center",
        fontsize=8,
        color=COLOUR["muted"],
    )
    save_fig(fig, path, dpi=300)


def fig_radial(ctx, path):
    apply_style()
    import matplotlib.pyplot as plt

    truth, kinuv, kinms = ctx["truth"], ctx["kinuv"], ctx["kinms"]
    hdr = ctx["hdr"]
    vsys_opt = ctx["vsys_opt"]
    pa = float(truth["pa_deg"])
    i_deg = float(truth["i_deg"])
    m0_d, m1_d, m2_d = ctx["mom_data"]
    m0_u, m1_u, m2_u = ctx["mom_kinuv"]
    m0_k, m1_k, m2_k = ctx["mom_kinms"]
    bmaj = float(hdr["BMAJ"]) * 3600.0
    r_beam = 0.5 * bmaj
    r = np.linspace(0.02, 3.0, 240)
    s_t = np.array([_arctan_slope(truth["v0_kms"], truth["r_t_arcsec"], x) for x in r])
    s_u = np.array([_arctan_slope(kinuv["v0_kms"], kinuv["r_t_arcsec"], x) for x in r])
    s_k = np.array([_arctan_slope(kinms["v0_kms"], kinms["r_t_arcsec"], x) for x in r])
    grid = ctx.get("kinuv_slices", {}).get("rt_v0")
    lo = s_u.copy()
    hi = s_u.copy()
    if grid is not None:
        xs = np.asarray(grid["x"])
        ys = np.asarray(grid["y"])
        z = np.asarray(grid["chi2"])
        z0 = float(np.min(z))
        ok = z - z0 <= CHI2_68
        if np.any(ok):
            band = []
            for x in r:
                vals = [
                    _arctan_slope(yv, xv, x)
                    for iy, yv in enumerate(ys)
                    for ix, xv in enumerate(xs)
                    if ok[iy, ix]
                ]
                band.append((min(vals), max(vals)))
            lo = np.array([b[0] for b in band])
            hi = np.array([b[1] for b in band])
    r_m1, g_m1 = _m1_gradient(m1_d, m0_d, hdr, pa, i_deg, vsys_opt)

    fig = plt.figure(figsize=(7.6, 8.4))
    gs = GridSpec(
        2, 1, figure=fig, left=0.12, right=0.97, top=0.90, bottom=0.08, hspace=0.28
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(r, lo, hi, color=COLOUR["model"], alpha=0.22, linewidth=0)
    ax.plot(r, s_t, color=COLOUR["data"], lw=2.0, label="Truth arctan")
    ax.plot(r, s_u, color=COLOUR["model"], lw=1.7, label="kinUV MAP")
    ax.plot(r, s_k, "--", color=KINMS_RED, lw=1.6, label="KinMS best fit")
    if r_m1.size:
        ax.plot(
            r_m1,
            g_m1,
            ":",
            color=COLOUR["vsys"],
            lw=1.5,
            label="CLEAN M1 numerical dV/dR",
        )
    ax.axvline(r_beam, color=COLOUR["vsys"], ls="--", lw=0.9)
    ax.text(r_beam + 0.06, 0.92, r"$R_{\rm beam}=\theta_{\rm FWHM}/2$", transform=ax.get_xaxis_transform(), fontsize=8, color=COLOUR["muted"])
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, None)
    ax.set_ylabel("dV_c / dR  (km/s / arcsec)")
    ax.set_xlabel("R (arcsec)")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    panel_letter(ax, "a")
    _inset(ax, _metric_box_text(truth, kinuv, kinms), loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    r_d, s_d = _annular_profile(m2_d, m0_d, hdr, pa, i_deg)
    r_u, s_um = _annular_profile(m2_u, m0_u, hdr, pa, i_deg)
    r_k, s_km = _annular_profile(m2_k, m0_k, hdr, pa, i_deg)
    ax2.axhline(float(truth["gas_sigma_kms"]), color=COLOUR["data"], lw=2.0, label="Truth σ = 10 km/s")
    ax2.axhline(float(kinuv["gas_sigma_kms"]), color=COLOUR["model"], lw=1.4, label=f"kinUV σ = {kinuv['gas_sigma_kms']:.2f} km/s")
    ax2.axhline(float(kinms["gas_sigma_kms"]), color=KINMS_RED, ls="--", lw=1.4, label=f"KinMS σ = {kinms['gas_sigma_kms']:.2f} km/s")
    ax2.plot(r_d, s_d, "o", ms=3.5, color=COLOUR["data"], label="Data M2(R)")
    ax2.plot(r_u, s_um, ".", ms=6, color=COLOUR["model"], alpha=0.8, label="kinUV M2(R)")
    ax2.plot(r_k, s_km, "s", ms=3.5, color=KINMS_RED, label="KinMS M2(R)")
    ax2.axvline(r_beam, color=COLOUR["vsys"], ls="--", lw=0.9)
    ax2.set_xlim(0.0, 3.0)
    ax2.set_xlabel("R (arcsec)")
    ax2.set_ylabel("σ or M2 (km/s)")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.legend(loc="upper right", ncol=2)
    panel_letter(ax2, "b")
    fig.suptitle("Controlled mock  ·  inner gradient and dispersion vs radius")
    fig.text(
        0.50,
        0.015,
        "Blue band = vis χ² 68% envelope on (V0, r_t); not MCMC  ·  "
        "horizontal lines are parametric gas_sigma; points are beam-smeared M2",
        ha="center",
        fontsize=8,
        color=COLOUR["muted"],
    )
    save_fig(fig, path, dpi=300)


def fig_pv_fan(ctx, path):
    apply_style()
    import matplotlib.pyplot as plt

    hdr = ctx["hdr"]
    data, kinuv, kinms = ctx["data_cube"], ctx["kinuv_k"], ctx["kinms_k"]
    pa0 = float(ctx["truth"]["pa_deg"])
    vsys_opt = ctx["vsys_opt"]
    ra = float(hdr.get("CRVAL1", 0.0))
    dec = float(hdr.get("CRVAL2", 0.0))
    width = 1.5 * abs(float(hdr["CDELT1"])) * 3600.0
    pas = (pa0 - 30.0, pa0 - 15.0, pa0, pa0 + 15.0, pa0 + 90.0)
    labels = (
        r"PA$_{\rm kin}$ − 30°",
        r"PA$_{\rm kin}$ − 15°",
        "Major",
        r"PA$_{\rm kin}$ + 15°",
        "Minor",
    )
    pvs = []
    off = None
    for pa in pas:
        d, off = pv_diagram(data, hdr, ra, dec, pa, LENGTH_ARCSEC, width)
        u, _ = pv_diagram(kinuv, hdr, ra, dec, pa, LENGTH_ARCSEC, width)
        k, _ = pv_diagram(kinms, hdr, ra, dec, pa, LENGTH_ARCSEC, width)
        pvs.append((d, u, k))
    extent = [off[0], off[-1], max(V_PV_LO, ctx["vel_opt"][0]), min(V_PV_HI, ctx["vel_opt"][-1])]
    data_clim = sequential_clim(*[p[0] for p in pvs])
    kuv_clim = symmetric_clim(*[p[0] - p[1] for p in pvs])
    kms_clim = symmetric_clim(*[p[0] - p[2] for p in pvs])
    fig = plt.figure(figsize=(14.8, 9.6))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 0.055],
        left=0.07,
        right=0.96,
        top=0.88,
        bottom=0.08,
        wspace=0.12,
        hspace=0.18,
    )
    row_names = ("Data PV", "Data − kinUV", "Data − KinMS")
    letters = "abcdefghijklmno"
    li = 0
    im_d = im_r = None
    for i in range(3):
        for j in range(5):
            ax = fig.add_subplot(gs[i, j])
            d, u, k = pvs[j]
            if i == 0:
                im_d = imshow_masked(
                    ax, d, extent, data_clim[0], data_clim[1], intensity_cmap(), aspect="auto"
                )
            else:
                resid = d - (u if i == 1 else k)
                rv0, rv1 = kuv_clim if i == 1 else kms_clim
                im_r = imshow_masked(ax, resid, extent, rv0, rv1, residual_cmap(), aspect="auto")
            vsys_line(ax, vsys_opt, orientation="h")
            if i == 0:
                ax.set_title(f"{labels[j]}\n{pas[j]:.1f}°")
            if i == 2:
                ax.set_xlabel("Offset (arcsec)")
            else:
                ax.tick_params(labelbottom=False)
            if j == 0:
                ax.set_ylabel(f"{row_names[i]}\nOptical km/s")
            else:
                ax.tick_params(labelleft=False)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            panel_letter(ax, letters[li])
            li += 1
        cax = fig.add_subplot(gs[i, 5])
        cbar(fig, im_d if i == 0 else im_r, "K", cax=cax)
    fig.suptitle("Controlled mock  ·  PV fan  ·  kinUV vs KinMS residuals")
    fig.text(
        0.50,
        0.015,
        "Positive offset is the receding side of each slit  ·  dashed = vsys",
        ha="center",
        fontsize=8,
        color=COLOUR["muted"],
    )
    save_fig(fig, path, dpi=300)


def _draw_pair(ax, x, y, z, truth_xy, kinuv_xy, kinms_xy, kinms_z, xlab, ylab, title):
    xx, yy = np.meshgrid(x, y)
    dz = np.asarray(z, dtype=np.float64) - float(np.min(z))
    ax.pcolormesh(xx, yy, dz, shading="auto", cmap=intensity_cmap(), alpha=0.35)
    ax.contour(
        xx,
        yy,
        dz,
        levels=[CHI2_68, CHI2_95],
        colors=[COLOUR["model"], COLOUR["model"]],
        linewidths=[1.6, 1.0],
        linestyles=["-", "--"],
    )
    if kinms_z is not None:
        kx = np.asarray(kinms_z["x"])
        ky = np.asarray(kinms_z["y"])
        kz = np.asarray(kinms_z["chi2"])
        kxx, kyy = np.meshgrid(kx, ky)
        kdz = kz - float(np.min(kz))
        ax.contour(
            kxx,
            kyy,
            kdz,
            levels=[CHI2_68, CHI2_95],
            colors=[KINMS_RED, KINMS_RED],
            linewidths=[1.6, 1.0],
            linestyles=["-", "--"],
        )
    ax.plot(*truth_xy, marker="*", ms=13, color=COLOUR["data"], label="Truth", zorder=6)
    ax.plot(*kinuv_xy, marker="o", ms=6, color=COLOUR["model"], label="kinUV MAP", zorder=6)
    ax.plot(*kinms_xy, marker="s", ms=6, color=KINMS_RED, label="KinMS fit", zorder=6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def fig_degeneracies(ctx, path):
    apply_style()
    import matplotlib.pyplot as plt

    sl = ctx["kinuv_slices"]
    ks = ctx["kinms_slices"]["slices"]
    t, u, k = ctx["truth"], ctx["kinuv"], ctx["kinms"]
    fig = plt.figure(figsize=(12.6, 4.4))
    gs = GridSpec(
        1, 3, figure=fig, left=0.07, right=0.99, top=0.82, bottom=0.16, wspace=0.28
    )
    pairs = (
        (
            "rt_v0",
            sl["rt_v0"]["x"],
            sl["rt_v0"]["y"],
            sl["rt_v0"]["chi2"],
            (t["r_t_arcsec"], t["v0_kms"]),
            (u["r_t_arcsec"], u["v0_kms"]),
            (k["r_t_arcsec"], k["v0_kms"]),
            ks.get("rt_v0"),
            "r_t (arcsec)",
            "V_0 (km/s)",
            "r_t vs V_0",
        ),
        (
            "rt_i",
            sl["rt_i"]["x"],
            sl["rt_i"]["y"],
            sl["rt_i"]["chi2"],
            (t["r_t_arcsec"], t["i_deg"]),
            (u["r_t_arcsec"], float(np.degrees(inclination_rad()))),
            (k["r_t_arcsec"], k["i_deg"]),
            ks.get("rt_i"),
            "r_t (arcsec)",
            "i (deg)",
            "r_t vs i  (kinUV i-scan only)",
        ),
        (
            "v0_sigma",
            sl["v0_sigma"]["x"],
            sl["v0_sigma"]["y"],
            sl["v0_sigma"]["chi2"],
            (t["v0_kms"], t["gas_sigma_kms"]),
            (u["v0_kms"], u["gas_sigma_kms"]),
            (k["v0_kms"], k["gas_sigma_kms"]),
            ks.get("v0_sigma"),
            "V_0 (km/s)",
            "gas_sigma (km/s)",
            "V_0 vs σ",
        ),
    )
    axes = []
    for i, spec in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        _draw_pair(ax, *spec[1:])
        panel_letter(ax, "abc"[i])
        axes.append(ax)
    axes[0].legend(loc="upper right", fontsize=7)
    _inset(axes[2], _metric_box_text(t, u, k), loc="lower right")
    fig.suptitle("Controlled mock  ·  vis χ² vs cube χ²  ·  68/95% Δχ² = 2.30 / 5.99")
    fig.text(
        0.50,
        0.02,
        "Blue = kinUV visibility χ²  ·  red = KinMS cube χ²  ·  "
        "Laplace slices, not MCMC  ·  S2 SBC failed  ·  kinUV i frozen except panel b",
        ha="center",
        fontsize=8,
        color=COLOUR["muted"],
    )
    save_fig(fig, path, dpi=300)


def _compute_kinuv_slices(vis, tmpl, grid, base, cache: Path) -> dict:
    if cache.is_file():
        return json.loads(cache.read_text())
    rt = np.linspace(0.18, 0.34, 7)
    v0 = np.linspace(242.0, 258.0, 7)
    ideg = np.linspace(38.0, 50.0, 7)
    sig = np.linspace(8.0, 12.5, 7)
    out = {
        "rt_v0": {
            "x": rt.tolist(),
            "y": v0.tolist(),
            "x_name": "r_t_arcsec",
            "y_name": "v0_kms",
            "chi2": _chi2_slice_2d(vis, tmpl, grid, base, "r_t_arcsec", rt, "v0_kms", v0).tolist(),
        },
        "rt_i": {
            "x": rt.tolist(),
            "y": ideg.tolist(),
            "x_name": "r_t_arcsec",
            "y_name": "i_deg",
            "chi2": _chi2_slice_2d(
                vis, tmpl, grid, base, "r_t_arcsec", rt, "i_deg", ideg, i_scan=True
            ).tolist(),
        },
        "v0_sigma": {
            "x": v0.tolist(),
            "y": sig.tolist(),
            "x_name": "v0_kms",
            "y_name": "gas_sigma_kms",
            "chi2": _chi2_slice_2d(
                vis, tmpl, grid, base, "v0_kms", v0, "gas_sigma_kms", sig
            ).tolist(),
        },
    }
    cache.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock-dir", type=Path, default=MOCK)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args(argv)
    mock = Path(args.mock_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((mock / "summary.json").read_text())
    truth = summary["truth"]
    kinuv = summary["kinuv_mock"]["fitted"]
    kinms = summary["kinms_mock"]["fitted"]
    vsys_opt = float(radio_to_optical_kms(float(truth["vsys_kms"])))
    i_deg = float(truth["i_deg"])
    v_proj = float(truth["v0_kms"]) * np.sin(np.radians(i_deg))
    z = vsys_opt / C_LIGHT_KM_S
    dist = Distance.from_redshift(z)

    cube_path = Path(summary["mock_cube"])
    hdr = fits.getheader(cube_path)
    data_cube = np.asarray(fits.getdata(cube_path), dtype=np.float64)
    vel_opt = spectral_axis_kms(hdr)
    dv = abs(float(hdr["CDELT3"]))
    kinuv_k = _kinuv_cube(kinuv, truth)
    kinms_cube = mock / "kinms_mock" / "model_cube.fits"
    kinms_k = _kinms_jy_beam_to_k(fits.getdata(kinms_cube), hdr, vel_opt)

    real = load_kgas066(CANFAR_NPZ, cube_path=CANFAR_CUBE_10)
    mock_vis = np.load(mock / "mock_vis.npz")
    vis = replace(real, vis=np.asarray(mock_vis["vis"], dtype=np.complex128))
    grid = image_grid_for_vis(vis)
    tmpl = load_sb_template(grid, ico_path=CANFAR_ICO)
    model_vis = predict_binned(vis, kinuv, tmpl, grid)
    dvis = np.asarray(vis.vis) - np.asarray(model_vis)

    mask3d = data_cube > 0
    mom_data = masked_moments(data_cube, vel_opt, mask3d, dv)
    mom_kinuv = masked_moments(kinuv_k, vel_opt, mask3d, dv)
    mom_kinms = masked_moments(kinms_k, vel_opt, mask3d, dv)

    kinuv_slices = _compute_kinuv_slices(vis, tmpl, grid, kinuv, out / "kinuv_chi2_slices.json")
    kinms_slices = _ensure_kinms_slices(out)

    ctx = {
        "truth": truth,
        "kinuv": kinuv,
        "kinms": kinms,
        "hdr": hdr,
        "data_cube": data_cube,
        "kinuv_k": kinuv_k,
        "kinms_k": kinms_k,
        "vel_opt": vel_opt,
        "vsys_opt": vsys_opt,
        "v_proj": v_proj,
        "vis": vis,
        "dvis": dvis,
        "z": z,
        "dist": dist,
        "mom_data": mom_data,
        "mom_kinuv": mom_kinuv,
        "mom_kinms": mom_kinms,
        "kinuv_slices": kinuv_slices,
        "kinms_slices": kinms_slices,
    }
    paths = {
        "fig_dirty_channel_residuals": out / "fig_dirty_channel_residuals.png",
        "fig_radial_gradient_profiles": out / "fig_radial_gradient_profiles.png",
        "fig_pv_angle_fan": out / "fig_pv_angle_fan.png",
        "fig_parameter_degeneracies": out / "fig_parameter_degeneracies.png",
    }
    fig_dirty_channels(ctx, paths["fig_dirty_channel_residuals"])
    fig_radial(ctx, paths["fig_radial_gradient_profiles"])
    fig_pv_fan(ctx, paths["fig_pv_angle_fan"])
    fig_degeneracies(ctx, paths["fig_parameter_degeneracies"])
    rec = {
        "status": "wrote_advanced_diagnostics",
        "figures": {k: str(v) for k, v in paths.items()},
        "note": (
            "kinUV dirty residual is a script-local type-1 DFT adjoint. "
            "Contours are χ² slices, not MCMC. quote_inner_slope true on mock only."
        ),
        "metrics": _metric_box_text(truth, kinuv, kinms),
        "quote_inner_slope_real_data": False,
    }
    (out / "summary.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
