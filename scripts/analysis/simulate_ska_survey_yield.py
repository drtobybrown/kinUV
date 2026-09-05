#!/usr/bin/env python3
"""SKA1-Mid 20,000 deg2 Band 2 HI yield: volume-weighted threshold model.

Integrates the ALFALFA Schechter mass function over comoving dV_c(z).
This is not an SKA visibility simulation and not a KGAS066 result.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from kinuv.constants import ARCSEC_TO_RAD, C_LIGHT_KM_S
from kinuv.diagnostics.style import COLOUR, apply_style, panel_letter, save_fig

REPO = Path(__file__).resolve().parents[2]
OUT = (
    REPO
    / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
    / "advanced_diagnostics"
)

H0 = 70.0
OM0 = 0.3
OL0 = 0.7
Z_MAX = 0.35
Z_MIN = 0.005
SKY_DEG2 = 20_000.0
OMEGA_SKY_SR = SKY_DEG2 * (np.pi / 180.0) ** 2
THETA_BEAM = 7.0
DV_KMS = 10.0
SIGMA_CHAN_JY = 70.0e-6
PHI_STAR = 4.5e-3
ALPHA = -1.30
LOG_MSTAR = 9.96
MSTAR = 10.0 ** LOG_MSTAR
HI_MASS_FACTOR = 2.356e5
RT_OVER_RHI = 0.18
V0_KMS = 130.0
V_TF_INDEX = 0.25
W_TURB_KMS = 15.0
SNR_INT_MIN = 5.0
SNR_CHAN_DET = 4.0
CUBE_D_BEAMS = 3.5
CUBE_RT_BEAMS = 0.60
CUBE_SNR = 10.0
KINUV_D_BEAMS = 1.2
KINUV_SNR = 5.0
NZ, NM, NI = 90, 70, 12
SEED = 66
BLUE = "#2b83ba"
RED = "#d7191c"
ORANGE = "#fdae61"
GRAY = COLOUR["vsys"]


def _ez(z):
    return np.sqrt(OM0 * (1.0 + np.asarray(z, dtype=np.float64)) ** 3 + OL0)


def cosmology_on_grid(z):
    """D_M, D_A, D_L [Mpc] and dV_c/dz [Mpc^3] for Omega_sky."""
    z = np.asarray(z, dtype=np.float64)
    try:
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=H0, Om0=OM0)
        da = np.asarray(cosmo.angular_diameter_distance(z).to_value("Mpc"), dtype=np.float64)
        dl = np.asarray(cosmo.luminosity_distance(z).to_value("Mpc"), dtype=np.float64)
        dm = da * (1.0 + z)
    except Exception:
        nint = 4000
        zg = np.linspace(0.0, float(np.max(z)) + 1.0e-6, nint)
        integ = np.concatenate([[0.0], np.cumulative_trapezoid(1.0 / _ez(zg), zg)])
        dc = (C_LIGHT_KM_S / H0) * integ
        dm = np.interp(z, zg, dc)
        da = dm / (1.0 + z)
        dl = dm * (1.0 + z)
    dvdz = (C_LIGHT_KM_S / H0) * (dm**2 / _ez(z)) * OMEGA_SKY_SR
    return dm, da, dl, dvdz


def schechter_dlogm(logm):
    """φ(M) per dex: ln(10) Φ* (M/M*)^{α+1} exp(-M/M*)."""
    x = 10.0 ** (np.asarray(logm, dtype=np.float64) - LOG_MSTAR)
    return np.log(10.0) * PHI_STAR * (x ** (ALPHA + 1.0)) * np.exp(-x)


def integrate_survey():
    z_e = np.linspace(Z_MIN, Z_MAX, NZ + 1)
    z = 0.5 * (z_e[1:] + z_e[:-1])
    dz = np.diff(z_e)
    logm_e = np.linspace(7.4, 11.4, NM + 1)
    logm = 0.5 * (logm_e[1:] + logm_e[:-1])
    dlogm = np.diff(logm_e)
    mu_e = np.linspace(0.0, 1.0, NI + 1)
    mu = 0.5 * (mu_e[1:] + mu_e[:-1])
    dmu = np.diff(mu_e)
    sini = np.sqrt(np.maximum(1.0 - mu**2, 0.0))

    _, da, dl, dvdz = cosmology_on_grid(z)
    kpc_per = da * 1.0e3 * ARCSEC_TO_RAD
    phi = schechter_dlogm(logm)
    m_hi = 10.0 ** logm
    d_hi = 10.0 ** (0.506 * logm - 3.293)
    r_t = RT_OVER_RHI * (0.5 * d_hi)
    v_rot = V0_KMS * (m_hi / 1.0e10) ** V_TF_INDEX

    # Broadcast: z, M, i  →  (nz, nm, ni)
    zz = z[:, None, None]
    ddl = dl[:, None, None]
    scale = kpc_per[:, None, None]
    mm = m_hi[None, :, None]
    dd = d_hi[None, :, None]
    rt = r_t[None, :, None]
    vv = v_rot[None, :, None]
    ss = sini[None, None, :]
    w50 = (2.0 * vv * ss + W_TURB_KMS) * (1.0 + zz)
    w50 = np.maximum(w50, DV_KMS)
    s_peak = mm / (HI_MASS_FACTOR * ddl**2 * w50)
    theta_d = np.broadcast_to(dd / scale, s_peak.shape).copy()
    theta_rt = np.broadcast_to(rt / scale, s_peak.shape).copy()
    snr_chan = s_peak / SIGMA_CHAN_JY
    snr_int = snr_chan * np.sqrt(w50 / DV_KMS)
    d_b = theta_d / THETA_BEAM
    rt_b = theta_rt / THETA_BEAM

    weight = (
        phi[None, :, None]
        * dlogm[None, :, None]
        * (dvdz[:, None, None] * dz[:, None, None])
        * dmu[None, None, :]
    )

    detect = (snr_int >= SNR_INT_MIN) & (snr_chan >= SNR_CHAN_DET)
    cube = (
        detect
        & (d_b >= CUBE_D_BEAMS)
        & (rt_b >= CUBE_RT_BEAMS)
        & (snr_chan >= CUBE_SNR)
    )
    ast = rt_b >= (1.0 / (2.0 * np.maximum(snr_chan, 1.0e-9)))
    kinuv = detect & (d_b >= KINUV_D_BEAMS) & (snr_chan >= KINUV_SNR) & ast

    n_z_det = np.sum(np.where(detect, weight, 0.0), axis=(1, 2))
    n_z_k = np.sum(np.where(kinuv, weight, 0.0), axis=(1, 2))
    n_z_c = np.sum(np.where(cube, weight, 0.0), axis=(1, 2))
    return {
        "z": z,
        "dz": dz,
        "n_z_det": n_z_det,
        "n_z_kinuv": n_z_k,
        "n_z_cube": n_z_c,
        "weight": weight,
        "detect": detect,
        "cube": cube,
        "kinuv": kinuv,
        "rt_b": rt_b,
        "snr_chan": snr_chan,
        "d_b": d_b,
        "dvdz": dvdz,
        "volume_mpc3": float(np.sum(dvdz * dz)),
    }


def _cum_to(z, n_z, z_cut):
    m = z <= float(z_cut)
    return float(np.sum(n_z[m]))


def _sample_phase(integ, n_plot=14000):
    w = np.where(integ["detect"], integ["weight"], 0.0).ravel()
    tot = float(np.sum(w))
    rng = np.random.default_rng(SEED)
    if tot <= 0.0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    p = w / tot
    idx = rng.choice(p.size, size=min(n_plot, int(np.sum(w > 0))), replace=True, p=p)
    rt = integ["rt_b"].ravel()[idx]
    snr = integ["snr_chan"].ravel()[idx]
    cub = integ["cube"].ravel()[idx]
    kuv = integ["kinuv"].ravel()[idx]
    return rt, snr, cub, kuv


def plot_figure(integ, path):
    apply_style()
    import matplotlib.pyplot as plt

    rt, snr, cub, kuv = _sample_phase(integ)
    exc = kuv & ~cub
    gray = ~cub & ~exc
    fig = plt.figure(figsize=(11.4, 4.7))
    gs = GridSpec(
        1, 2, figure=fig, left=0.08, right=0.92, top=0.86, bottom=0.16, wspace=0.30
    )
    ax = fig.add_subplot(gs[0, 0])
    if rt.size:
        ax.scatter(
            rt[gray], snr[gray], s=5, c=GRAY, alpha=0.18, linewidths=0,
            rasterized=True, label="Unresolved / unfit", zorder=1,
        )
        ax.scatter(
            rt[exc], snr[exc], s=6, c=BLUE, alpha=0.28, linewidths=0,
            rasterized=True, label="kinUV exclusive", zorder=2,
        )
        ax.scatter(
            rt[cub], snr[cub], s=7, c=RED, alpha=0.40, linewidths=0,
            rasterized=True, label="Cube-plane viable", zorder=3,
        )
    xx = np.logspace(np.log10(0.04), np.log10(2.5), 240)
    ax.plot(xx, 1.0 / (2.0 * xx), color=BLUE, lw=1.6, zorder=4, label="Astrometric limit")
    ax.axvline(CUBE_RT_BEAMS, color=RED, ls="--", lw=1.0, zorder=4)
    ax.axhline(CUBE_SNR, color=RED, ls="--", lw=1.0, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.04, 2.5)
    ax.set_ylim(4.0, 150.0)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("r_t / theta_beam")
    ax.set_ylabel("Peak channel SNR")
    ax.set_title("Kinematic recovery phase space")
    ax.text(
        0.08, 0.22, "kinUV\nsub-beam\nadvantage",
        transform=ax.transAxes, color=BLUE, fontsize=8, va="bottom", ha="left",
    )
    ax.text(
        0.96, 0.90, "Conventional\ncube regime",
        transform=ax.transAxes, color=RED, fontsize=8, va="top", ha="right",
    )
    ax.legend(loc="lower right", markerscale=1.8)
    panel_letter(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    z = integ["z"]
    z_plot = np.linspace(0.02, 0.35, 90)
    n_det = np.array([_cum_to(z, integ["n_z_det"], zc) for zc in z_plot])
    n_k = np.array([_cum_to(z, integ["n_z_kinuv"], zc) for zc in z_plot])
    n_c = np.array([_cum_to(z, integ["n_z_cube"], zc) for zc in z_plot])
    with np.errstate(divide="ignore", invalid="ignore"):
        mult = np.where(n_c > 0.0, n_k / n_c, np.nan)
    ax2.plot(z_plot, n_det, ":", color=COLOUR["data"], lw=1.7, label="5σ detections")
    ax2.plot(z_plot, n_k, "-", color=BLUE, lw=1.8, label="kinUV viable")
    ax2.plot(z_plot, n_c, "--", color=RED, lw=1.6, label="Cube-plane viable")
    ax2.set_xlim(0.02, 0.35)
    ymax = max(1.0e4, float(np.nanmax(n_det)) * 1.08)
    ax2.set_ylim(0.0, ymax)
    ax2.set_xlabel("Redshift z")
    ax2.set_ylabel("Cumulative viable targets")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    ax2.set_title("Scientific target yield")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2r = ax2.twinx()
    ax2r.plot(z_plot, mult, "-.", color=ORANGE, lw=1.6, label="Yield multiplier")
    ax2r.set_ylabel("N_kinUV / N_cube", color=ORANGE)
    mmax = float(np.nanmax(mult)) if np.any(np.isfinite(mult)) else 6.0
    ax2r.set_ylim(1.0, max(6.0, mmax * 1.08))
    ax2r.tick_params(axis="y", colors=ORANGE, direction="in")
    ax2r.yaxis.label.set_color(ORANGE)
    ax2r.spines["right"].set_color(ORANGE)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")
    panel_letter(ax2, "b")

    fig.suptitle("SKA1-Mid Band 2  ·  20,000 deg2 HI yield  ·  kinUV vs cube plane")
    fig.text(
        0.50,
        0.018,
        "Omega_sky = 20000 deg2  ·  theta_beam = 7 arcsec  ·  sigma_chan = 70 uJy  ·  "
        "ALFALFA Schechter + Wang+2016  ·  volume integral, not a visibility simulation  ·  "
        "not KGAS066",
        ha="center",
        fontsize=8,
        color=COLOUR["muted"],
    )
    save_fig(fig, path, dpi=300)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    integ = integrate_survey()
    z_cuts = (0.05, 0.10, 0.20, 0.30, 0.35)
    rows = []
    for zc in z_cuts:
        n_det = _cum_to(integ["z"], integ["n_z_det"], zc)
        n_k = _cum_to(integ["z"], integ["n_z_kinuv"], zc)
        n_c = _cum_to(integ["z"], integ["n_z_cube"], zc)
        rows.append(
            {
                "z": float(zc),
                "n_detect": n_det,
                "n_kinuv": n_k,
                "n_cube": n_c,
                "n_kinuv_exclusive": n_k - n_c,
                "multiplier": (n_k / n_c) if n_c > 0.0 else None,
            }
        )
    rec = {
        "status": "wrote_ska_volume_yield",
        "note": (
            "Comoving-volume integral of the ALFALFA Schechter function over "
            "20000 deg2. Analytic SNR and size cuts only. Not an SKA visibility "
            "simulation. Not a KGAS066 fit. quote_inner_slope false on real data."
        ),
        "survey": {
            "sky_deg2": SKY_DEG2,
            "omega_sky_sr": OMEGA_SKY_SR,
            "z_max": Z_MAX,
            "H0": H0,
            "Om0": OM0,
            "Ol0": OL0,
            "volume_mpc3": integ["volume_mpc3"],
            "theta_beam_arcsec": THETA_BEAM,
            "dv_kms": DV_KMS,
            "sigma_chan_ujy": SIGMA_CHAN_JY * 1.0e6,
            "phi_star_mpc3_dex": PHI_STAR,
            "alpha": ALPHA,
            "log_mstar": LOG_MSTAR,
            "wang2016": "log10(D_HI/kpc) = 0.506 log10(M_HI) - 3.293",
            "r_t_over_R_HI": RT_OVER_RHI,
            "s_peak": "M_HI / (2.356e5 D_L^2 W_50) Jy, as written; no beam dilution",
            "W_50": "(2 V_rot sin i + 15 km/s) (1+z); V_rot=130 (M/1e10)^0.25 km/s",
        },
        "cuts": {
            "detect": {"snr_int": SNR_INT_MIN, "snr_chan": SNR_CHAN_DET},
            "cube": {"d_beams": CUBE_D_BEAMS, "rt_beams": CUBE_RT_BEAMS, "snr": CUBE_SNR},
            "kinuv": {
                "d_beams": KINUV_D_BEAMS,
                "snr": KINUV_SNR,
                "astrometric": "r_t/theta_beam >= 1/(2 SNR_chan)",
            },
        },
        "totals_z035": rows[-1],
        "cumulative": [r for r in rows if r["z"] <= 0.30],
        "figure": str(out / "fig_ska_survey_kinuv_impact.png"),
        "quote_inner_slope_real_data": False,
    }
    plot_figure(integ, out / "fig_ska_survey_kinuv_impact.png")
    (out / "survey_yield_metrics.json").write_text(json.dumps(rec, indent=2) + "\n")
    dest_script = out / "simulate_ska_survey_yield.py"
    src = Path(__file__).resolve()
    if dest_script.resolve() != src:
        shutil.copy2(src, dest_script)
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
