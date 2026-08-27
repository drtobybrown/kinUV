"""Moments, spectra, and PV of Stage B vs the 10 km/s imaging cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from kinuv.diagnostics.imaging import (
    match_model_to_imaging,
    masked_moments,
    offset_world,
    pv_diagram,
    spectral_axis_kms,
)

ROOT_10KMS = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms"
)
MAP_DIR = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-f47bc9-map"
)
ARTIFACT = Path("docs/reviews/artifacts/2026-08-27-stage-b-imaging")
LENGTH_ARCSEC = 16.0


def _extent_arcsec(header) -> tuple[float, float, float, float]:
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    dx = float(header["CDELT1"]) * 3600.0
    dy = float(header["CDELT2"]) * 3600.0
    x0 = (0.5 - float(header["CRPIX1"])) * dx
    x1 = (nx + 0.5 - float(header["CRPIX1"])) * dx
    y0 = (0.5 - float(header["CRPIX2"])) * dy
    y1 = (ny + 0.5 - float(header["CRPIX2"])) * dy
    return x0, x1, y0, y1


def _finite_share(*arrays):
    m = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _imshow(ax, img, extent, vmin, vmax, cmap, title):
    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("East offset (arcsec)")
    ax.set_ylabel("North offset (arcsec)")
    return im


def _moment_figure(data, model, residual, labels, extents, vsys, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.5), constrained_layout=True)
    rows = (
        ("mom0", "K km/s", "inferno"),
        ("mom1", "km/s", "RdBu_r"),
        ("mom2", "km/s", "viridis"),
    )
    for i, (key, unit, cmap) in enumerate(rows):
        d, m = data[key], model[key]
        r = residual[key]
        share = _finite_share(d, m)
        if key == "mom1":
            span = np.nanpercentile(np.abs(d[share] - vsys), 95) if np.any(share) else 150.0
            vmin, vmax = vsys - span, vsys + span
            rv = np.nanpercentile(np.abs(r[share]), 95) if np.any(share) else 20.0
        else:
            vmax = np.nanpercentile(d[share], 99) if np.any(share) else 1.0
            vmin = 0.0
            rv = np.nanpercentile(np.abs(r[share]), 95) if np.any(share) else 1.0
        for ax, img, title, v0, v1, cm in (
            (axes[i, 0], d, f"Data {labels[key]}", vmin, vmax, cmap),
            (axes[i, 1], m, f"Stage B {labels[key]}", vmin, vmax, cmap),
            (axes[i, 2], r, f"Residual (data−model)", -rv, rv, "RdBu_r"),
        ):
            im = _imshow(ax, img, extents, v0, v1, cm, title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=unit)
    fig.suptitle("KGAS066 Stage B vs 10 km/s imaging cube (same 2-D spatial mask)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _spectra_figure(v, spec_d, spec_m, aper, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    titles = ("Mask-integrated", "Centre", "Approaching", "Receding")
    keys = ("mask", "centre", "approaching", "receding")
    for ax, title, key in zip(axes, titles, keys):
        ax.plot(v, aper[key][0], color="k", lw=1.4, label="data")
        ax.plot(v, aper[key][1], color="C0", lw=1.4, label="Stage B")
        ax.set_title(title)
        ax.set_ylabel("Flux (mJy)")
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[2:]:
        ax.set_xlabel("Optical velocity (km/s, LSRK)")
    fig.suptitle("Spectra (same mask / 1-beam apertures)")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    # mask-integrated stored as spec_d/spec_m too
    _ = spec_d, spec_m


def _pv_figure(v, offsets, data_pv, model_pv, title, out):
    import matplotlib.pyplot as plt

    resid = data_pv - model_pv
    vmax = np.nanpercentile(np.abs(data_pv[np.isfinite(data_pv)]), 99)
    rv = np.nanpercentile(np.abs(resid[np.isfinite(resid)]), 95)
    extent = [offsets[0], offsets[-1], v[0], v[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, img, name, v0, v1, cmap in (
        (axes[0], data_pv, "Data", 0.0, vmax, "inferno"),
        (axes[1], model_pv, "Stage B", 0.0, vmax, "inferno"),
        (axes[2], resid, "Residual (data−model)", -rv, rv, "RdBu_r"),
    ):
        im = ax.imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=v0,
            vmax=v1,
            cmap=cmap,
        )
        ax.set_title(name)
        ax.set_xlabel("Offset (arcsec, receding +)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="K")
    axes[0].set_ylabel("Optical velocity (km/s)")
    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _spectrum_mjy(cube_k, mask2d, header, vel_kms):
    """Sum of Jy over a 2-D spatial mask, one value per channel."""
    from kinuv.template.wiener import k_to_jy_per_beam

    bmaj = float(header["BMAJ"]) * 3600.0
    bmin = float(header["BMIN"]) * 3600.0
    rest = float(header.get("RESTFRQ", 230.538e9))
    v_mid = float(np.median(vel_kms))
    nu = rest / (1.0 + v_mid / 2.99792458e5)
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
    p.add_argument("--matched-fits", type=Path, default=MAP_DIR / "stage_b_model_on_10kms.fits")
    args = p.parse_args(argv)

    out_dir = args.out_dir
    if out_dir is None:
        repo = Path(__file__).resolve().parents[1]
        out_dir = repo / ARTIFACT
    out_dir.mkdir(parents=True, exist_ok=True)

    data_hdu = fits.open(args.data_cube)[0]
    mask_hdu = fits.open(args.mask_cube)[0]
    model_hdu = fits.open(args.model_cube)[0]
    data = np.asarray(data_hdu.data, dtype=np.float64)
    mask = np.asarray(mask_hdu.data) > 0
    model_jy = np.asarray(model_hdu.data, dtype=np.float64)
    geom = json.loads(args.stage_a.read_text())
    pa = float(geom["pa_deg"]) % 360.0
    dx, dy = float(geom["dx_arcsec"]), float(geom["dy_arcsec"])
    vsys_opt = float(geom["vsys_kms"]) / (1.0 - float(geom["vsys_kms"]) / 2.99792458e5)

    matched, vel, dv = match_model_to_imaging(
        model_jy, model_hdu.header, data_hdu.header
    )
    hdr = data_hdu.header
    fits.PrimaryHDU(matched.astype(np.float32), header=hdr).writeto(
        args.matched_fits, overwrite=True
    )
    print(f"wrote {args.matched_fits} {matched.shape}", flush=True)

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
    labels = {"mom0": "moment 0", "mom1": "moment 1", "mom2": "moment 2"}
    extent = _extent_arcsec(hdr)
    _moment_figure(data_m, model_m, resid, labels, extent, vsys_opt, out_dir / "moments.png")

    spec_d = _spectrum_mjy(data, mask2d, hdr, vel)
    spec_m = _spectrum_mjy(np.nan_to_num(matched, nan=0.0), mask2d, hdr, vel)
    ra0, dec0 = float(hdr["CRVAL1"]), float(hdr["CRVAL2"])
    ra_c, dec_c = offset_world(ra0, dec0, dx, dy)
    bmaj = float(hdr["BMAJ"]) * 3600.0
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
    _spectra_figure(vel, spec_d, spec_m, aper, out_dir / "spectra.png")

    width = bmaj
    matched0 = np.nan_to_num(matched, nan=0.0)
    d_maj, off = pv_diagram(data, hdr, ra_c, dec_c, pa, LENGTH_ARCSEC, width)
    m_maj, _ = pv_diagram(matched0, hdr, ra_c, dec_c, pa, LENGTH_ARCSEC, width)
    _pv_figure(vel, off, d_maj, m_maj, "Major-axis PV (PA = {:.1f}°)".format(pa), out_dir / "pv_major.png")
    d_min, _ = pv_diagram(data, hdr, ra_c, dec_c, pa + 90.0, LENGTH_ARCSEC, width)
    m_min, _ = pv_diagram(matched0, hdr, ra_c, dec_c, pa + 90.0, LENGTH_ARCSEC, width)
    _pv_figure(vel, off, d_min, m_min, "Minor-axis PV (PA + 90°)", out_dir / "pv_minor.png")

    summary = {
        "data_cube": str(args.data_cube),
        "mask_cube": str(args.mask_cube),
        "model_cube": str(args.model_cube),
        "matched_fits": str(args.matched_fits),
        "pa_deg": pa,
        "dx_arcsec": dx,
        "dy_arcsec": dy,
        "vsys_optical_kms": vsys_opt,
        "mom0_sum_data": float(np.nansum(data_m["mom0"])),
        "mom0_sum_model": float(np.nansum(model_m["mom0"])),
        "plots": [str(out_dir / n) for n in ("moments.png", "spectra.png", "pv_major.png", "pv_minor.png")],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
