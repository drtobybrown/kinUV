#!/usr/bin/env python3
"""Official 10 vs 30 km/s Ico product at frozen MAP θ (DEC-066-SB lock).

Not a Δv A/B. 10 km/s Ico is Briggs / 1.04″ / 0.3″ / 2871 pix. Production
``load_sb_template`` K=(0.02)^2 — empty-corner n=0 on both v1.3 stamps.
Lock 30 km/s this card even if the named conventions fire. No re-MAP.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

from kinuv.diagnostics.style import COLOUR, apply_style, save_fig
from kinuv.forward.sb import load_legacy_sb_template, place_template_on_grid
from kinuv.infer.map import _optimal_flux, image_grid_for_vis, predict_binned
from kinuv.io.vis import load_kgas066
from kinuv.likelihood.chi2 import chi2, chi2_zero
from kinuv.template.fftpad import default_pad_n
from kinuv.template.wiener import empty_corner_rms, ico_to_template
from kinuv.transforms.dft import vis_uv_wavelengths
from kinuv.transforms.grid import ImageGrid

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-ico-10-vs-30"

ICO_30 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
ICO_10 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE_30 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
MAP_JSON = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/production/KGAS066/"
    "kinuv-KGAS066-uvsign-map/stage_a_map.json"
)

CHI2_OFFICIAL = 168675.6
S_OFFICIAL = 0.5136098555284736
DELTA_VS_ZERO = 35552.65225039818
LEFTOVER_SB = 1373.0
K_PROD = (0.02) ** 2
HISTORICAL_NU_OBS_ICO_HZ = 224.3e9
BINS_KLAM = (0.0, 50.0, 146.0, 182.0, math.inf)
BIN_LABELS = ("lt_50", "50_146", "146_182", "gt_182")
THETA_30 = 1.2948
THETA_10 = 1.0400
K_BEAM_30 = 1.0 / THETA_30
K_BEAM_10 = 1.0 / THETA_10
K_NYQ_30 = 1.0 / (2.0 * 0.4)
PARAM_KEYS = (
    "flux",
    "pa_deg",
    "vsys_kms",
    "gas_sigma_kms",
    "dx_arcsec",
    "dy_arcsec",
    "v0_kms",
    "r_t_arcsec",
)


def _self_test_180() -> dict:
    """Review-a unit test: 180² 0.3″ Gaussian, no FITS, no 1709/0.4/1.30."""
    if default_pad_n(180) != 512:
        raise RuntimeError(f"default_pad_n(180)={default_pad_n(180)} != 512")
    cell = 0.3
    ny = nx = 180
    y = (np.arange(ny) - ny // 2) * cell
    x = (np.arange(nx) - nx // 2) * cell
    xx, yy = np.meshgrid(x, y, indexing="xy")
    img = np.exp(-0.5 * (xx**2 + yy**2) / (1.2**2))
    tmpl = ico_to_template(
        img,
        cell,
        HISTORICAL_NU_OBS_ICO_HZ,
        1.04,
        0.95,
        -44.8,
        mask=None,
        sigma_empty=0.02 * float(np.nanmax(np.abs(img))),
        pad_n=None,
    )
    if not np.isclose(tmpl.cell_arcsec, cell):
        raise RuntimeError(f"cell_arcsec={tmpl.cell_arcsec} != 0.3")
    n_mask = int(np.sum(tmpl.mask))
    if n_mask == 1709:
        raise RuntimeError("mask collapsed to 1709")
    grid = ImageGrid(nx=64, ny=64, cell_arcsec=0.15)
    placed = place_template_on_grid(tmpl.sb, tmpl.cell_arcsec, grid)
    integ = float(placed.sum() * grid.cell_arcsec**2)
    if abs(integ - 1.0) >= 1e-4:
        raise RuntimeError(f"ImageGrid integral {integ} not 1±1e-4")
    return {
        "default_pad_n_180": 512,
        "cell_arcsec": float(tmpl.cell_arcsec),
        "mask_sum": n_mask,
        "imagegrid_integral": integ,
        "k_wiener": float(tmpl.k_wiener),
    }


def _header_record(path: Path) -> dict:
    with fits.open(path) as hdul:
        h = hdul[0].header
        data = np.squeeze(np.asarray(hdul[0].data, dtype=np.float64))
    finite = np.isfinite(data)
    return {
        "path": str(path),
        "naxis1": int(h["NAXIS1"]),
        "naxis2": int(h["NAXIS2"]),
        "cell_arcsec": abs(float(h["CDELT2"])) * 3600.0,
        "bmaj_arcsec": float(h["BMAJ"]) * 3600.0,
        "bmin_arcsec": float(h["BMIN"]) * 3600.0,
        "bpa_deg": float(h["BPA"]),
        "bunit": str(h.get("BUNIT", "")),
        "finite": int(np.sum(finite)),
        "i_peak": float(np.nanmax(np.abs(data))),
        "empty_corner_rms": empty_corner_rms(data),
        "default_pad_n": default_pad_n(max(int(h["NAXIS1"]), int(h["NAXIS2"]))),
    }


def _gate_10kms(rec: dict) -> None:
    if rec["naxis1"] != 180 or rec["naxis2"] != 180:
        raise RuntimeError(f"10 km/s NAXIS {rec['naxis1']}x{rec['naxis2']} != 180")
    if not np.isclose(rec["cell_arcsec"], 0.3, atol=1e-4):
        raise RuntimeError(f"10 km/s cell {rec['cell_arcsec']} != 0.3")
    if rec["finite"] != 2871:
        raise RuntimeError(f"10 km/s finite {rec['finite']} != 2871")
    if rec["default_pad_n"] < 360:
        raise RuntimeError(f"pad {rec['default_pad_n']} < 360")


def _map_params(raw: dict) -> dict[str, float]:
    return {k: float(raw[k]) for k in PARAM_KEYS}


def _raw_azimuthal_pk(img, cell_arcsec, n_bin=24):
    """Raw (not floor-normalised) azimuthal P(k) on one ImageGrid."""
    a = np.asarray(img, dtype=np.float64)
    a = np.where(np.isfinite(a), a, 0.0)
    ny, nx = a.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    f = np.fft.fftshift(np.fft.fft2(a * win))
    p2 = f.real**2 + f.imag**2
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=float(cell_arcsec)))
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=float(cell_arcsec)))
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    kk = np.hypot(kkx, kky)
    kmax = 0.45 / float(cell_arcsec)
    edges = np.linspace(0.0, kmax, n_bin + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    pk = np.full(n_bin, np.nan)
    counts = np.zeros(n_bin, dtype=int)
    for i in range(n_bin):
        m = (kk >= edges[i]) & (kk < edges[i + 1])
        counts[i] = int(np.sum(m))
        if counts[i]:
            pk[i] = float(np.median(p2[m]))
    return centres, pk, counts


def _chi2_splits(vis, model, weights, s, b_klam) -> dict:
    mag2 = (vis.real - model.real) ** 2 + (vis.imag - model.imag) ** 2
    ww = np.asarray(weights, dtype=np.float64)
    out = {"total": float(s * np.sum(ww * mag2))}
    edges = BINS_KLAM
    for lab, lo, hi in zip(BIN_LABELS, edges[:-1], edges[1:]):
        m = (b_klam >= lo) & (b_klam < hi) & (ww > 0.0)
        out[lab] = {
            "chi2": float(s * np.sum(ww[m] * mag2[m])),
            "n_vis": int(np.sum(m)),
        }
    m150 = (b_klam > 150.0) & (ww > 0.0)
    out["gt_150"] = {
        "chi2": float(s * np.sum(ww[m150] * mag2[m150])),
        "n_vis": int(np.sum(m150)),
    }
    return out


def _plot_pk(k, p30, p10, path: Path) -> None:
    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(k, p30, color=COLOUR["data"], label="30 km/s Ico product")
    ax.plot(k, p10, color=COLOUR["model"], label="10 km/s Ico product (Briggs)")
    ax.axvline(K_BEAM_30, color=COLOUR["vsys"], ls="--", lw=1.0, label=r"$1/\theta_{30}$")
    ax.axvline(K_BEAM_10, color=COLOUR["muted"], ls=":", lw=1.0, label=r"$1/\theta_{10}$")
    ax.axvline(146e3 * (math.pi / 180.0 / 3600.0), color=COLOUR["zero"], ls="-.", lw=0.8)
    ax.set_xlabel("k (arcsec$^{-1}$)")
    ax.set_ylabel("P(k) raw")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.suptitle("Deconvolved SB power  ·  common ImageGrid  ·  not a Δv A/B")
    save_fig(fig, path, dpi=300)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    self_test = _self_test_180()
    rec30 = _header_record(ICO_30)
    rec10 = _header_record(ICO_10)
    _gate_10kms(rec10)
    rec30["k_used"] = K_PROD
    rec10["k_used"] = K_PROD
    rec30["sigma_empty_used"] = 0.02 * rec30["i_peak"]
    rec10["sigma_empty_used"] = 0.02 * rec10["i_peak"]
    rec30["label"] = "30 km/s Ico product (natural, 1.295\", 0.4\", 1709 pix)"
    rec10["label"] = "10 km/s Ico product (Briggs, 1.04\", 0.3\", 2871 pix)"

    raw = json.loads(MAP_JSON.read_text())
    params = _map_params(raw)
    data = load_kgas066(NPZ, cube_path=CUBE_30)
    s_loaded = float(data.s)
    data.s = float(S_OFFICIAL)
    if tuple(data.vis.shape) != (881, 95):
        raise RuntimeError(f"vis shape {data.vis.shape} != (881, 95)")
    grid = image_grid_for_vis(data)
    replay = {
        "relative_noise_fraction": 0.02,
        "observed_frequency_hz": HISTORICAL_NU_OBS_ICO_HZ,
    }
    tmpl30 = load_legacy_sb_template(grid, ico_path=ICO_30, **replay)
    tmpl10 = load_legacy_sb_template(grid, ico_path=ICO_10, **replay)

    model30 = np.asarray(predict_binned(data, params, tmpl30, grid))
    model10 = np.asarray(predict_binned(data, params, tmpl10, grid))
    c30 = float(chi2(data.vis, model30, data.weights, data.s))
    c10 = float(chi2(data.vis, model10, data.weights, data.s))
    c0 = float(chi2_zero(data.vis, data.weights, data.s))
    identity_ok = abs(c30 - CHI2_OFFICIAL) < 1.0
    dchi_zero = c0 - c30

    p_unit = dict(params)
    p_unit["flux"] = 1.0
    unit30 = np.asarray(predict_binned(data, p_unit, tmpl30, grid))
    unit10 = np.asarray(predict_binned(data, p_unit, tmpl10, grid))
    flux30 = _optimal_flux(data.vis, unit30, data.weights)
    flux10 = _optimal_flux(data.vis, unit10, data.weights)
    c30_opt = float(chi2(data.vis, flux30 * unit30, data.weights, data.s))
    c10_opt = float(chi2(data.vis, flux10 * unit10, data.weights, data.s))

    u_lam, v_lam = vis_uv_wavelengths(data.u_m, data.v_m, data.freqs)
    b_klam = np.hypot(u_lam, v_lam) / 1.0e3
    splits30 = _chi2_splits(data.vis, model30, data.weights, data.s, b_klam)
    splits10 = _chi2_splits(data.vis, model10, data.weights, data.s, b_klam)
    dchi_joint = splits10["50_146"]["chi2"] - splits30["50_146"]["chi2"]
    dchi_gt50 = (
        (splits10["50_146"]["chi2"] + splits10["146_182"]["chi2"] + splits10["gt_182"]["chi2"])
        - (splits30["50_146"]["chi2"] + splits30["146_182"]["chi2"] + splits30["gt_182"]["chi2"])
    )
    dchi_total = c10 - c30
    convention_a = bool(dchi_joint < -9.0)

    k, p30, n_k = _raw_azimuthal_pk(tmpl30, grid.cell_arcsec)
    _, p10, _ = _raw_azimuthal_pk(tmpl10, grid.cell_arcsec)
    ratio = p10 / p30
    hi = np.isfinite(ratio) & (k > K_BEAM_30) & (k < K_NYQ_30) & (n_k > 0)
    if not np.any(hi):
        median_r = float("nan")
        convention_b = False
        pk_empty_high_k = True
    else:
        median_r = float(np.median(ratio[hi]))
        convention_b = bool(median_r <= 1.2)
        pk_empty_high_k = False

    def _r_at(k0: float) -> float:
        if not np.any(np.isfinite(ratio)):
            return float("nan")
        i = int(np.nanargmin(np.abs(k - k0)))
        return float(ratio[i])

    lock_30 = True
    would_unlock = bool(identity_ok and convention_a and convention_b)
    _plot_pk(k, p30, p10, OUT / "pk_deconvolved.png")

    note = {
        "label_10": rec10["label"],
        "not_delta_v_ab": True,
        "confounders": {
            "weighting": "natural vs Briggs robust=0.5",
            "beam_area_ratio": 1.541,
            "cell_area_ratio": 1.778,
            "mask_ratio": 1.680,
            "bpa_diff_deg": 26.5,
            "clean_threshold_mJy": [0.80, 1.48],
        },
        "wiener": {
            "path": "load_sb_template 0.02*peak K; empty_corner n=0 on both stamps",
            "k": K_PROD,
            "ico_30": rec30,
            "ico_10": rec10,
        },
        "self_test_180": self_test,
        "identity": {
            "chi2_30": c30,
            "chi2_official": CHI2_OFFICIAL,
            "abs_diff": abs(c30 - CHI2_OFFICIAL),
            "ok": identity_ok,
            "s_official": S_OFFICIAL,
            "s_loaded": s_loaded,
            "shape": list(data.vis.shape),
            "delta_chi2_vs_zero": dchi_zero,
            "delta_chi2_vs_zero_official": DELTA_VS_ZERO,
        },
        "conditional_chi2": {
            "theta": "official MAP kinuv-KGAS066-uvsign-map; no re-MAP",
            "chi2_30": c30,
            "chi2_10": c10,
            "delta_10_minus_30": dchi_total,
            "leftover_sb_scale": LEFTOVER_SB,
            "do_not_print_3sigma": True,
            "splits_30": splits30,
            "splits_10": splits10,
            "delta_joint_50_146": dchi_joint,
            "delta_gt_50": dchi_gt50,
            "B": "hypot(*vis_uv_wavelengths) per visibility (kλ); not leftover metres",
            "nu_hz": [float(np.min(data.freqs)), float(np.max(data.freqs))],
        },
        "optimal_flux_diagnostic": {
            "flux_30": flux30,
            "flux_10": flux10,
            "chi2_30": c30_opt,
            "chi2_10": c10_opt,
            "delta_10_minus_30": c10_opt - c30_opt,
            "sign_flip_vs_frozen": bool(
                np.sign(c10_opt - c30_opt) != np.sign(dchi_total) and dchi_total != 0.0
            ),
            "not_a_gate": True,
        },
        "pk": {
            "estimator": "raw azimuthal P(k) on common ImageGrid; not S3 floor-normalised",
            "k_arcsec_inv": k.tolist(),
            "P_30": p30.tolist(),
            "P_10": p10.tolist(),
            "R": ratio.tolist(),
            "median_R_highk": median_r,
            "R_at_1_over_theta30": _r_at(K_BEAM_30),
            "R_at_1_over_theta10": _r_at(K_BEAM_10),
            "k_beam_30": K_BEAM_30,
            "k_beam_10": K_BEAM_10,
            "k_taper_30_arcsec_inv": 146e3 * (math.pi / 180.0 / 3600.0),
            "convention_b_median_R_le_1p2": convention_b,
            "empty_high_k_bin": pk_empty_high_k,
        },
        "gate": {
            "convention_a_joint_dchi_lt_m9": convention_a,
            "convention_b_pk": convention_b,
            "would_unlock_if_this_card_allowed": would_unlock,
            "lock_30kms": lock_30,
            "lock_is_artifact_and_STATUS_only": True,
            "do_not_edit_DEC_066_SB": True,
            "do_not_change_sb_py": True,
        },
        "quote_inner_slope": False,
        "pending_007": ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"],
    }
    (OUT / "summary.json").write_text(json.dumps(note, indent=2) + "\n")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Official 10 vs 30 Ico product (not a Δv A/B)",
                "",
                "Candidate is the v1.3 **10 km/s Ico product** "
                "(`KGAS66_Ico_K_kms-1.fits`, Briggs, 1.04″, 0.3″, 2871 pix).",
                "Not cube M0. Not a channel-width A/B. Confounders: natural vs Briggs, "
                "beam-area 1.541, cell-area 1.778, mask 1.680, BPA 26.5°, "
                "CLEAN 0.80 vs 1.48 mJy.",
                "",
                f"30 km/s identity χ² = {c30:.3f} (official {CHI2_OFFICIAL}; "
                f"|Δ|={abs(c30 - CHI2_OFFICIAL):.3f}; ok={identity_ok}).",
                f"Conditional χ² at official MAP θ: 30={c30:.3f}, 10={c10:.3f}, "
                f"Δ(10−30)={dchi_total:.3f}. Joint 50–146 kλ Δχ²={dchi_joint:.3f}.",
                f"P(k) median R(k) on k>1/θ_30 and below 30 km/s Nyquist = {median_r}.",
                "",
                "**Lock DEC-066-SB at 30 km/s.** Gate is archive-only. "
                "Do not edit `DEC-066-SB.md`, `sb.py`, or `BMAJ_ICO_ARCSEC`. "
                "No re-MAP. `quote_inner_slope: false`. Do not start G4.",
                "",
                "Wiener path: production `load_sb_template` K=(0.02)^2. "
                "Empty-corner K is undefined (n=0 finite in all four corners).",
                "",
            ]
        )
        + "\n"
    )
    print(json.dumps({
        "identity_ok": identity_ok,
        "chi2_30": c30,
        "chi2_10": c10,
        "dchi_total": dchi_total,
        "dchi_joint_50_146": dchi_joint,
        "median_R_highk": median_r,
        "lock_30kms": lock_30,
        "out": str(OUT),
    }, indent=2))
    if not identity_ok:
        print("STATUS: 30 km/s identity failed; do not lock-from-garbage; production still 30 km/s.")
    return 0 if identity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
