#!/usr/bin/env python3
"""m=2 SB at frozen official MAP θ. Opt-in; does not change production SB.

36×20 grid in (φ2, A) then L-BFGS-B polish. Raw χ² only. No DEC file.
No kinuv-KGAS066-m2-map tree. quote_inner_slope: false.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from kinuv.diagnostics.figures import plot_leftover_chi2
from kinuv.diagnostics.s1 import leftover_chi2
from kinuv.diagnostics.style import apply_style, residual_cmap, save_fig
from kinuv.forward.model import sky_cube
from kinuv.forward.sb import apply_m2, axisymmetrise_template, load_sb_template
from kinuv.geometry import inclination_rad
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.io.vis import load_kgas066
from kinuv.likelihood.chi2 import chi2, chi2_zero
from kinuv.constants import freq_to_velocity_kms

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-m2-sb"
ICO_30 = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
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
N_PHI = 36
N_A = 20


def _params(raw):
    return {k: float(raw[k]) for k in PARAM_KEYS}


def _chi2(data, params, tmpl, grid, *, xla: bool):
    model = predict_binned(data, params, tmpl, grid, xla=xla)
    model = np.asarray(model)
    return float(chi2(data.vis, model, data.weights, data.s)), model


def _m1_maps(params, tmpl, grid, data):
    cube = np.asarray(
        sky_cube(
            tmpl,
            grid,
            data.freqs_native,
            flux=params["flux"],
            pa_rad=np.radians(params["pa_deg"]),
            vsys_kms=params["vsys_kms"],
            dx_arcsec=params["dx_arcsec"],
            dy_arcsec=params["dy_arcsec"],
            gas_sigma_kms=params["gas_sigma_kms"],
            v0_kms=params["v0_kms"],
            r_t_arcsec=params["r_t_arcsec"],
        )
    )
    vel = freq_to_velocity_kms(data.freqs_native)
    dv = float(np.median(np.abs(np.diff(vel)))) if vel.size > 1 else 1.0
    m0 = np.sum(cube, axis=2) * dv
    num = np.sum(cube * vel[None, None, :], axis=2) * dv
    m1 = np.divide(num, m0, out=np.full_like(m0, np.nan), where=m0 > 0)
    return m0, m1


def _plot_templates(t1, t3, cell, path):
    apply_style()
    import matplotlib.pyplot as plt

    n = t1.shape[0]
    ext = 0.5 * n * cell
    extent = (ext, -ext, -ext, ext)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4))
    vmax = np.nanpercentile(np.abs(t1), 99)
    for ax, img, title in (
        (axes[0], t1, "2-D Ico (unit SB)"),
        (axes[1], t3, "best m=2 (unit SB)"),
        (axes[2], t3 - t1, "m2 minus 2-D Ico"),
    ):
        if title.startswith("m2 minus"):
            lim = np.nanpercentile(np.abs(img), 98)
            im = ax.imshow(img, origin="lower", extent=extent, cmap=residual_cmap(), vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(img, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=vmax)
        ax.set_xlim(5, -5)
        ax.set_ylim(-5, 5)
        ax.set_title(title, fontsize=9)
        ax.tick_params(direction="in", top=True, right=True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Central 5 arcsec  ·  vis grid templates  ·  vis chi2 is the fit")
    fig.tight_layout()
    save_fig(fig, path, dpi=300)


def _plot_heat(phi, amp, z, path):
    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    im = ax.pcolormesh(np.degrees(phi), amp, z, shading="auto")
    ax.set_xlabel("phi2 (deg)")
    ax.set_ylabel("A")
    ax.set_title("chi2 grid  ·  frozen MAP theta  ·  not a MAP")
    ax.tick_params(direction="in", top=True, right=True)
    fig.colorbar(im, ax=ax, label="chi2")
    save_fig(fig, path, dpi=300)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = json.loads(MAP_JSON.read_text())
    params = _params(raw)
    pa_rad = np.radians(params["pa_deg"])
    i_rad = inclination_rad()
    data = load_kgas066(NPZ, cube_path=CUBE_30)
    data.s = float(S_OFFICIAL)
    if tuple(data.vis.shape) != (881, 95):
        raise RuntimeError(f"vis shape {data.vis.shape} != (881, 95)")
    grid = image_grid_for_vis(data)
    tmpl1 = load_sb_template(grid, ico_path=ICO_30)
    xla = True
    try:
        c1, m1 = _chi2(data, params, tmpl1, grid, xla=xla)
    except Exception:
        xla = False
        c1, m1 = _chi2(data, params, tmpl1, grid, xla=False)
    identity_ok = abs(c1 - CHI2_OFFICIAL) < 1.0
    c0 = float(chi2_zero(data.vis, data.weights, data.s))
    dchi_zero = c0 - c1

    i0, i0_info = axisymmetrise_template(tmpl1, grid, pa_rad, i_rad)
    tmpl2, _ = apply_m2(i0, grid, pa_rad, i_rad, 0.0, 0.0)
    c2, m2 = _chi2(data, params, tmpl2, grid, xla=xla)

    phi_grid = np.linspace(0.0, np.pi, N_PHI, endpoint=False)
    a_grid = np.linspace(0.0, 0.8, N_A)
    heat = np.full((N_A, N_PHI), np.nan)
    best = {"chi2": np.inf, "A": 0.0, "phi2": 0.0}
    n_eval = 0
    for ia, amp in enumerate(a_grid):
        for ip, phi2 in enumerate(phi_grid):
            try:
                tmpl, _ = apply_m2(i0, grid, pa_rad, i_rad, float(amp), float(phi2))
            except ValueError:
                continue
            c, _ = _chi2(data, params, tmpl, grid, xla=xla)
            heat[ia, ip] = c
            n_eval += 1
            if c < best["chi2"]:
                best = {"chi2": c, "A": float(amp), "phi2": float(phi2)}
        print(f"grid row A={amp:.3f} best={best['chi2']:.3f} n={n_eval}", flush=True)

    def _obj(x):
        amp = float(np.clip(x[0], 0.0, 0.8))
        phi2 = float(x[1] % np.pi)
        try:
            tmpl, _ = apply_m2(i0, grid, pa_rad, i_rad, amp, phi2)
        except ValueError:
            return 1e12
        c, _ = _chi2(data, params, tmpl, grid, xla=xla)
        return c

    polish = minimize(
        _obj,
        np.array([best["A"], best["phi2"]]),
        method="L-BFGS-B",
        bounds=((0.0, 0.8), (0.0, np.pi - 1e-8)),
    )
    polish_chi2 = float(polish.fun)
    polish_A = float(np.clip(polish.x[0], 0.0, 0.8))
    polish_phi = float(polish.x[1] % np.pi)
    if polish.success and polish_chi2 < best["chi2"]:
        chosen = {"A": polish_A, "phi2": polish_phi, "chi2": polish_chi2, "source": "polish"}
    else:
        chosen = {**best, "source": "grid"}

    tmpl3, m2_info = apply_m2(i0, grid, pa_rad, i_rad, chosen["A"], chosen["phi2"])
    c3, m3 = _chi2(data, params, tmpl3, grid, xla=xla)

    mom0_1, mom1_1 = _m1_maps(params, tmpl1, grid, data)
    mom0_2, mom1_2 = _m1_maps(params, tmpl2, grid, data)
    mom0_3, mom1_3 = _m1_maps(params, tmpl3, grid, data)
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.cell_arcsec
    xe, yn = np.meshgrid(x, y, indexing="xy")
    sky_r = np.hypot(xe - params["dx_arcsec"], yn - params["dy_arcsec"])
    peak = float(np.nanmax(mom0_3))
    mask = np.isfinite(mom0_3) & (mom0_3 > 0.05 * peak) & (sky_r < 5.0)
    d32 = np.abs(mom1_3 - mom1_2)[mask]
    d31 = np.abs(mom1_3 - mom1_1)[mask]
    med32 = float(np.nanmedian(d32)) if d32.size else float("nan")
    med31 = float(np.nanmedian(d31)) if d31.size else float("nan")
    m1_pass = bool(med32 < 2.0 and med31 < 2.0)

    for tag, model in (("ico2d", m1), ("a0", m2), ("m2", m3)):
        b_m, per_row, vel, per_chan = leftover_chi2(data, model)
        plot_leftover_chi2(b_m, per_row, vel, per_chan, OUT / f"leftover_{tag}.png")
    _plot_templates(tmpl1, tmpl3, grid.cell_arcsec, OUT / "m0_templates_central5.png")
    _plot_heat(phi_grid, a_grid, heat, OUT / "chi2_grid.png")

    note = {
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "not_a_map": True,
        "identity": {
            "chi2_1_ico2d": c1,
            "ok": identity_ok,
            "abs_diff": abs(c1 - CHI2_OFFICIAL),
            "delta_vs_zero": dchi_zero,
            "delta_vs_zero_official": DELTA_VS_ZERO,
            "s": S_OFFICIAL,
            "shape": list(data.vis.shape),
            "ico_path": str(ICO_30),
        },
        "chi2": {
            "ico2d": c1,
            "A0": c2,
            "m2": c3,
            "tax_A0_minus_ico2d": c2 - c1,
            "gain_m2_minus_A0": c3 - c2,
            "leftover_m2_minus_ico2d": c3 - c1,
            "leftover_sb_scale": LEFTOVER_SB,
            "do_not_print_sigma_or_dlnL": True,
        },
        "grid": {
            "n_phi": N_PHI,
            "n_A": N_A,
            "best_A": best["A"],
            "best_phi2": best["phi2"],
            "best_chi2": best["chi2"],
            "heat": heat.tolist(),
            "phi2": phi_grid.tolist(),
            "A": a_grid.tolist(),
        },
        "polish": {
            "success": bool(polish.success),
            "chi2": polish_chi2,
            "A": polish_A,
            "phi2": polish_phi,
            "kept": chosen["source"],
        },
        "chosen": chosen | {"info": m2_info},
        "i0": {
            "dr_arcsec": i0_info["dr_arcsec"],
            "r_centre_arcsec": i0_info["r_centre_arcsec"].tolist(),
            "I0_R": i0_info["I0_R"].tolist(),
            "n_positive": i0_info["n_positive"].tolist(),
            "grid_cell_arcsec": grid.cell_arcsec,
            "grid_n": grid.nx,
        },
        "m1_invariance": {
            "median_abs_dM1_m2_minus_A0": med32,
            "median_abs_dM1_m2_minus_ico2d": med31,
            "pass_lt_2kms": m1_pass,
            "n_pix": int(np.sum(mask)),
            "model_model_sky_cube": True,
        },
        "pending_007": ["b1mqxsov", "xkytxih1", "y5tspgit", "zq1olquy"],
    }
    if not identity_ok:
        note["blocked"] = "identity failed; do not quote m=2 delta chi2"
    (OUT / "summary.json").write_text(json.dumps(note, indent=2) + "\n")
    lines = [
        "# m=2 SB at frozen official MAP θ (not a MAP)",
        "",
        "I0 = azimuthal mean of the official 30 km/s Ico. Production `load_sb_template` unchanged.",
        f"Identity χ²_2D = {c1:.3f} (official {CHI2_OFFICIAL}; ok={identity_ok}).",
        f"A=0 χ² = {c2:.3f} (tax vs 2-D Ico = {c2 - c1:.3f}).",
        f"m=2 χ² = {c3:.3f} at A={chosen['A']:.3f}, phi2={np.degrees(chosen['phi2']):.1f} deg ({chosen['source']}).",
        f"Deltas: (2)-(1)={c2 - c1:.3f}; (3)-(2)={c3 - c2:.3f}; (3)-(1)={c3 - c1:.3f}.",
        "Raw χ² only. Not 3σ, not ΔlnL, not nested. leftover SB scale = 1373; vs V=0 = +35553.",
        f"M1 model-model medians |ΔM1| (3)-(2)={med32:.3f} km/s, (3)-(1)={med31:.3f} km/s; pass={m1_pass}.",
        "",
        "`quote_inner_slope: false`. Official MAP unchanged. No G4.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))
    print(json.dumps({
        "identity_ok": identity_ok,
        "chi2_1": c1,
        "chi2_2": c2,
        "chi2_3": c3,
        "chosen": chosen,
        "m1_pass": m1_pass,
        "xla": xla,
    }, indent=2))
    if not identity_ok:
        print("STATUS: identity failed; do not quote m=2 Δχ².")
        return 2
    if not m1_pass:
        print("STATUS: M1 invariance failed; do not claim SB/M1 decoupling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
