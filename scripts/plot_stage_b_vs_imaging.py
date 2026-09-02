"""Moments, spectra, and PV of Stage B vs the 10 km/s imaging cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec

from kinuv.constants import C_LIGHT_KM_S
from kinuv.diagnostics.imaging import (
    flux_weighted_velocity,
    match_model_to_imaging,
    masked_moments,
    offset_world,
    pv_diagram,
    spectral_wcs_report,
)
from kinuv.diagnostics.style import (
    COLOUR,
    CROP_ARCSEC,
    apply_style,
    beam_ellipse,
    cbar,
    data_model_residual_grid,
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
from kinuv.forward.model import VSYS_SEED_KM_S
from kinuv.io.vis import radio_to_optical_kms

ROOT_10KMS = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms"
)
MAP_DIR = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map"
)
ARTIFACT = Path("docs/reviews/artifacts/2026-08-30-final-fit")
LENGTH_ARCSEC = 16.0


def _moment_figure(data, model, residual, extent, vsys, beam, centre, out, *, model_label="Stage B"):
    apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.4, 8.9))
    axes, cax_pair, cax_res = data_model_residual_grid(
        fig, 3, left=0.11, right=0.90, top=0.90, bottom=0.08
    )
    rows = (
        ("mom0", "M0", "K km/s", "K km/s", intensity_cmap(), False),
        ("mom1", "M1", "v − vsys (km/s)", "km/s", velocity_cmap(), True),
        ("mom2", "M2", "km/s", "km/s", intensity_cmap(), False),
    )
    bmaj, bmin, bpa = beam
    cx, cy = centre
    crop = CROP_ARCSEC
    for i, (key, row_name, unit, res_unit, cmap, is_vel) in enumerate(rows):
        d, m_img, r = data[key], model[key], residual[key]
        if is_vel:
            d, m_img = d - vsys, m_img - vsys
            vmin, vmax = symmetric_clim(d, m_img)
        else:
            vmin, vmax = sequential_clim(d, m_img)
        rv0, rv1 = symmetric_clim(r)
        ims = []
        for j, (img, v0, v1, cm) in enumerate(
            (
                (d, vmin, vmax, cmap),
                (m_img, vmin, vmax, cmap),
                (r, rv0, rv1, residual_cmap()),
            )
        ):
            ax = axes[i][j]
            ims.append(imshow_masked(ax, img, extent, v0, v1, cm))
            format_sky_ax(ax, crop, centre, xlabel=(i == 2), ylabel=(j == 0))
            if i == 0:
                ax.set_title(("Data", "Model", "Residual")[j])
        axes[i][0].text(
            -0.32, 0.5, row_name, transform=axes[i][0].transAxes,
            rotation=90, va="center", ha="center", fontsize=11,
        )
        cbar(fig, ims[0], unit, cax=cax_pair[i])
        cbar(fig, ims[2], res_unit, cax=cax_res[i])
    beam_ellipse(axes[0][0], bmaj, bmin, bpa, (cx + crop - 2.1, cy - crop + 2.1))
    fig.suptitle(f"KGAS066  ·  {model_label} vs 10 km/s cube", fontsize=11, y=0.97)
    fig.text(
        0.50, 0.015,
        "east left, north up  ·  same 2-D spatial mask  ·  M1 shown as v − vsys (optical, LSRK)",
        ha="center", fontsize=8, color=COLOUR["muted"],
    )
    save_fig(fig, out)


def _spectra_figure(
    v, aper, vsys, pa, dv_model_minus_data, out, *, catalog_vsys=None, model_label="Stage B"
):
    apply_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.4, 6.4))
    gs = GridSpec(
        2, 2, figure=fig,
        left=0.10, right=0.97, top=0.86, bottom=0.12, wspace=0.16, hspace=0.28,
    )
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
    titles = (
        "Mask-integrated",
        "Centre (1 beam)",
        f"Approaching along fitted PA ({pa:.1f}°)",
        f"Receding along fitted PA ({pa:.1f}°)",
    )
    keys = ("mask", "centre", "approaching", "receding")
    for ax, title, key, letter in zip(axes.ravel(), titles, keys, "abcd"):
        yd, ym = aper[key]
        ax.plot(v, yd, color=COLOUR["data"], lw=1.5, label="data", zorder=2)
        ax.plot(v, ym, color=COLOUR["model"], lw=1.35, label=model_label, zorder=3)
        ax.axhline(0.0, color=COLOUR["zero"], lw=0.6, zorder=1)
        vsys_line(ax, vsys, orientation="v")
        if catalog_vsys is not None:
            ax.axvline(
                float(catalog_vsys), color=COLOUR["vsys"], ls=":", lw=0.8, zorder=4,
            )
        dv = dv_model_minus_data[key]
        ax.text(
            0.97, 0.92, rf"$\Delta v_{{\rm M-D}}={dv:+.1f}$ km/s",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=COLOUR["muted"],
        )
        ax.set_title(title, fontsize=10)
        panel_letter(ax, letter)
    for ax in axes[:, 0]:
        ax.set_ylabel("Flux (mJy)")
    for ax in axes[1, :]:
        ax.set_xlabel("Optical velocity (km/s, LSRK)")
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(0.97, 0.995), fontsize=8)
    fig.suptitle(
        "Spectra  ·  1-beam apertures along the fitted receding PA",
        fontsize=11, y=0.995, x=0.42,
    )
    fig.text(
        0.50, 0.015,
        "dashed = MAP $v_{\\rm sys}$ (optical)  ·  dotted = catalogue optical $v_{\\rm sys}$  ·  "
        r"$\Delta v_{\rm M-D}$ is the flux-weighted centroid, not a WCS/Hann shift",
        ha="center", fontsize=7.5, color=COLOUR["muted"],
    )
    save_fig(fig, out)


def _pv_figure(v, offsets, data_pv, model_pv, title, vsys, out):
    apply_style()
    import matplotlib.pyplot as plt

    resid = data_pv - model_pv
    vmin, vmax = sequential_clim(data_pv, model_pv)
    rv0, rv1 = symmetric_clim(resid)
    extent = [offsets[0], offsets[-1], v[0], v[-1]]
    fig = plt.figure(figsize=(11.4, 6.5))
    axes, cax_pair, cax_res = data_model_residual_grid(
        fig, 1, left=0.09, right=0.90, top=0.86, bottom=0.14
    )
    row = axes[0]
    panels = (
        (data_pv, vmin, vmax, intensity_cmap(), "Data"),
        (model_pv, vmin, vmax, intensity_cmap(), "Model"),
        (resid, rv0, rv1, residual_cmap(), "Residual"),
    )
    ims = []
    for j, (img, v0, v1, cmap, name) in enumerate(panels):
        ax = row[j]
        ims.append(imshow_masked(ax, img, extent, v0, v1, cmap, aspect="auto"))
        ax.set_title(name)
        ax.set_xlabel("Offset (arcsec; receding +)")
        vsys_line(ax, vsys, orientation="h")
        panel_letter(ax, "abc"[j])
    row[0].set_ylabel("Optical velocity (km/s, LSRK)")
    cbar(fig, ims[0], "K", cax=cax_pair[0])
    cbar(fig, ims[2], "K", cax=cax_res[0])
    fig.suptitle(title, fontsize=11, y=0.97)
    save_fig(fig, out)


def _spectrum_mjy(cube_k, mask2d, header, vel_kms):
    """Sum of Jy over a 2-D spatial mask, one value per channel."""
    from kinuv.template.wiener import k_to_jy_per_beam

    bmaj = float(header["BMAJ"]) * 3600.0
    bmin = float(header["BMIN"]) * 3600.0
    rest = float(header.get("RESTFRQ", 230.538e9))
    v_mid = float(np.median(vel_kms))
    nu = rest / (1.0 + v_mid / C_LIGHT_KM_S)
    cell = abs(float(header["CDELT1"])) * 3600.0
    omega_beam = np.pi * bmaj * bmin / (4.0 * np.log(2.0))
    omega_pix = cell * cell
    pix_per_beam = omega_beam / omega_pix
    jy_per_k = float(k_to_jy_per_beam(1.0, nu, bmaj, bmin)) / pix_per_beam
    t = np.asarray(cube_k, dtype=np.float64)
    m = np.asarray(mask2d, dtype=bool)
    return 1.0e3 * jy_per_k * np.nansum(np.where(m, t, 0.0), axis=(1, 2))


def _aperture_mask(header, ra, dec, radius_arcsec):
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    w = WCS(header).celestial
    yy, xx = np.mgrid[0:ny, 0:nx]
    world = w.all_pix2world(np.stack([xx.ravel(), yy.ravel()], axis=1), 0)
    dra = (world[:, 0] - ra) * np.cos(np.deg2rad(dec)) * 3600.0
    ddec = (world[:, 1] - dec) * 3600.0
    r = np.hypot(dra, ddec).reshape(ny, nx)
    return r <= float(radius_arcsec)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-cube", type=Path, default=ROOT_10KMS / "KGAS66_clipped_cube.fits")
    p.add_argument("--mask-cube", type=Path, default=ROOT_10KMS / "KGAS66_mask_cube.fits")
    p.add_argument("--model-cube", type=Path, default=MAP_DIR / "stage_b_model_cube.fits")
    p.add_argument("--stage-a", type=Path, default=MAP_DIR / "stage_a_map.json")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--matched-fits",
        type=Path,
        default=None,
        help="Matched K cube path. Default: <out-dir>/model_on_10kms.fits (never the official MAP tree).",
    )
    p.add_argument(
        "--model-label",
        default="Stage B",
        help="Legend/suptitle for the model (Stage A MAP / NUTS-mean Stage A / Stage B rings).",
    )
    args = p.parse_args(argv)
    for label, path in (("data-cube", args.data_cube), ("mask-cube", args.mask_cube)):
        if "30kms" in str(path):
            raise SystemExit(f"{label} must be the 10 km/s cube, not 30kms: {path}")

    out_dir = args.out_dir
    if out_dir is None:
        repo = Path(__file__).resolve().parents[1]
        out_dir = repo / ARTIFACT
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_fits = args.matched_fits if args.matched_fits is not None else out_dir / "model_on_10kms.fits"
    map_root = MAP_DIR.resolve()
    dest = matched_fits.resolve()
    if dest.parent == map_root or map_root in dest.parents:
        raise SystemExit(
            f"--matched-fits must not write under the official MAP tree: {matched_fits}"
        )

    data_hdu = fits.open(args.data_cube)[0]
    mask_hdu = fits.open(args.mask_cube)[0]
    model_hdu = fits.open(args.model_cube)[0]
    data = np.asarray(data_hdu.data, dtype=np.float64)
    mask = np.asarray(mask_hdu.data) > 0
    model_jy = np.asarray(model_hdu.data, dtype=np.float64)
    geom = json.loads(args.stage_a.read_text())
    pa = float(geom["pa_deg"]) % 360.0
    dx, dy = float(geom["dx_arcsec"]), float(geom["dy_arcsec"])
    vsys_radio = float(geom["vsys_kms"])
    vsys_opt = float(radio_to_optical_kms(vsys_radio))
    catalog_vsys = float(VSYS_SEED_KM_S)

    wcs_rows = [
        spectral_wcs_report(data_hdu.header, label="data_10kms"),
        spectral_wcs_report(model_hdu.header, label=str(args.model_label).replace(" ", "_")),
    ]
    print("spectral WCS:", json.dumps(wcs_rows, indent=2), flush=True)

    matched, vel, dv = match_model_to_imaging(
        model_jy, model_hdu.header, data_hdu.header
    )
    hdr = data_hdu.header
    fits.PrimaryHDU(matched.astype(np.float32), header=hdr).writeto(
        matched_fits, overwrite=True
    )
    print(f"wrote {matched_fits} {matched.shape}", flush=True)

    # Data cube is already blanked outside the 3-D mask. Applying that 3-D
    # mask to the model would clip flux in velocity (kinematic mismatch)
    # and starve moment 0. Same clipping = 2-D spatial footprint.
    mask2d = np.any(mask, axis=0)
    clip = np.broadcast_to(mask2d, data.shape)
    d_m = masked_moments(data, vel, clip, dv)
    m_m = masked_moments(np.nan_to_num(matched, nan=0.0), vel, clip, dv)
    keys = ("mom0", "mom1", "mom2")
    data_m = dict(zip(keys, d_m))
    model_m = dict(zip(keys, m_m))
    resid = {k: data_m[k] - model_m[k] for k in keys}
    extent = sky_extent_arcsec(hdr)
    bmaj = float(hdr["BMAJ"]) * 3600.0
    bmin = float(hdr["BMIN"]) * 3600.0
    bpa = float(hdr["BPA"])
    _moment_figure(
        data_m,
        model_m,
        resid,
        extent,
        vsys_opt,
        (bmaj, bmin, bpa),
        (dx, dy),
        out_dir / "moments.png",
        model_label=args.model_label,
    )

    spec_d = _spectrum_mjy(data, mask2d, hdr, vel)
    spec_m = _spectrum_mjy(np.nan_to_num(matched, nan=0.0), mask2d, hdr, vel)
    ra0, dec0 = float(hdr["CRVAL1"]), float(hdr["CRVAL2"])
    ra_c, dec_c = offset_world(ra0, dec0, dx, dy)
    aper = {"mask": (spec_d, spec_m)}
    for name, east, north in (
        ("centre", 0.0, 0.0),
        ("receding", 4.0 * np.sin(np.deg2rad(pa)), 4.0 * np.cos(np.deg2rad(pa))),
        ("approaching", -4.0 * np.sin(np.deg2rad(pa)), -4.0 * np.cos(np.deg2rad(pa))),
    ):
        ap = _aperture_mask(hdr, *offset_world(ra_c, dec_c, east, north), bmaj)
        aper[name] = (
            _spectrum_mjy(data, ap, hdr, vel),
            _spectrum_mjy(np.nan_to_num(matched, nan=0.0), ap, hdr, vel),
        )
    dv_md = {}
    centroids = {}
    for key, (yd, ym) in aper.items():
        vd = flux_weighted_velocity(yd, vel)
        vm = flux_weighted_velocity(ym, vel)
        centroids[key] = {"data_kms": vd, "model_kms": vm, "model_minus_data_kms": vm - vd}
        dv_md[key] = vm - vd
    _spectra_figure(
        vel, aper, vsys_opt, pa, dv_md, out_dir / "spectra.png",
        catalog_vsys=catalog_vsys,
        model_label=args.model_label,
    )

    width = bmaj
    matched0 = np.nan_to_num(matched, nan=0.0)
    d_maj, off = pv_diagram(data, hdr, ra_c, dec_c, pa, LENGTH_ARCSEC, width)
    m_maj, _ = pv_diagram(matched0, hdr, ra_c, dec_c, pa, LENGTH_ARCSEC, width)
    _pv_figure(
        vel,
        off,
        d_maj,
        m_maj,
        "Major-axis PV  ·  fitted PA = {:.1f}°  ·  receding +".format(pa),
        vsys_opt,
        out_dir / "pv_major.png",
    )
    d_min, _ = pv_diagram(data, hdr, ra_c, dec_c, pa + 90.0, LENGTH_ARCSEC, width)
    m_min, _ = pv_diagram(matched0, hdr, ra_c, dec_c, pa + 90.0, LENGTH_ARCSEC, width)
    _pv_figure(
        vel,
        off,
        d_min,
        m_min,
        "Minor-axis PV  ·  fitted PA + 90°",
        vsys_opt,
        out_dir / "pv_minor.png",
    )

    summary = {
        "data_cube": str(args.data_cube),
        "mask_cube": str(args.mask_cube),
        "model_cube": str(args.model_cube),
        "matched_fits": str(matched_fits),
        "pa_deg": pa,
        "dx_arcsec": dx,
        "dy_arcsec": dy,
        "vsys_radio_kms": vsys_radio,
        "vsys_optical_kms": vsys_opt,
        "catalog_vsys_optical_kms": catalog_vsys,
        "vsys_map_minus_catalog_kms": vsys_opt - catalog_vsys,
        "spectral_wcs": wcs_rows,
        "centroid_model_minus_data_kms": {k: centroids[k]["model_minus_data_kms"] for k in centroids},
        "mom0_sum_data": float(np.nansum(data_m["mom0"])),
        "mom0_sum_model": float(np.nansum(model_m["mom0"])),
        "plots": [str(out_dir / n) for n in ("moments.png", "spectra.png", "pv_major.png", "pv_minor.png")],
        "root_cause": (
            "Image-plane centroid redshift is vis-weighted MAP vsys vs CLEAN-cube "
            "brightness weighting (frozen Wiener Ico, r_t floor), not radio/optical, "
            "RESTFRQ, CRPIX, or Hann phase. Do not fudge the model velocity axis."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shift = {
        "vsys_radio_kms": vsys_radio,
        "vsys_optical_kms": vsys_opt,
        "catalog_vsys_optical_kms": catalog_vsys,
        "vsys_map_minus_catalog_kms": vsys_opt - catalog_vsys,
        "spectral_wcs": wcs_rows,
        "centroids": centroids,
        "root_cause": summary["root_cause"],
    }
    (out_dir / "vsys_shift.json").write_text(json.dumps(shift, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
