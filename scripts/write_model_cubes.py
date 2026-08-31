#!/usr/bin/env python3
"""Write Stage B (and Stage A) sky cubes as FITS for inspection. Not a fit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from kinuv.constants import F_REST_CO21_HZ, freq_to_velocity_kms
from kinuv.forward.model import sky_cube
from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis
from kinuv.io.vis import load_kgas066

NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)
CUBE = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_clipped_cube.fits"
)
MAP_DIR = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map"
)


def _wcs(grid, vel_kms, ico_hdr) -> WCS:
    w = WCS(naxis=3)
    w.wcs.crpix = [grid.nx // 2 + 1, grid.ny // 2 + 1, 1.0]
    cdelt = float(grid.cell_arcsec) / 3600.0
    dv = float(np.median(np.diff(vel_kms))) if vel_kms.size > 1 else 1.0
    w.wcs.cdelt = np.array([-cdelt, cdelt, dv])
    w.wcs.crval = [
        float(ico_hdr.get("CRVAL1", 0.0)),
        float(ico_hdr.get("CRVAL2", 0.0)),
        float(vel_kms[0]),
    ]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "VRAD"]
    w.wcs.cunit = ["deg", "deg", "km/s"]
    w.wcs.specsys = "LSRK"
    return w


def _hdu(cube_yxv, grid, vel_kms, ico_hdr, extra: dict) -> fits.PrimaryHDU:
    # FITS spectral-first: (v, y, x)
    data = np.moveaxis(np.asarray(cube_yxv, dtype=np.float32), 2, 0)
    # sky_cube +x is east; CDELT1 < 0 means FITS NAXIS1 increases west.
    data = np.flip(data, axis=2)
    hdr = _wcs(grid, vel_kms, ico_hdr).to_header()
    hdr["BUNIT"] = "Jy/pixel"
    hdr["RESTFRQ"] = (float(F_REST_CO21_HZ), "CO(2-1) rest frequency [Hz]")
    hdr["OBJECT"] = "KGAS066"
    hdr["ORIGIN"] = "kinUV Stage B sky_cube"
    hdr["COMMENT"] = "Array NAXIS1 increases west (CDELT1<0); sky +x east was flipped on write."
    for k, v in extra.items():
        hdr[k] = v
    return fits.PrimaryHDU(data=data, header=hdr)


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-dir", type=Path, default=MAP_DIR)
    args = p.parse_args(argv)
    map_dir = args.map_dir
    a = json.loads((map_dir / "stage_a_map.json").read_text())
    b = json.loads((map_dir / "stage_b_map.json").read_text())
    data = load_kgas066(NPZ, cube_path=CUBE if CUBE.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO)
    freqs = data.freqs_native
    vel = freq_to_velocity_kms(freqs)
    print(
        f"grid {grid.nx}x{grid.ny} cell={grid.cell_arcsec:.4f}\" "
        f"n_chan={freqs.size} dv~{np.median(np.diff(vel)):.4f}",
        flush=True,
    )
    kw = dict(
        template=tmpl,
        grid=grid,
        freqs_hz=freqs,
        flux=a["flux"],
        pa_rad=np.deg2rad(a["pa_deg"]),
        vsys_kms=a["vsys_kms"],
        dx_arcsec=a["dx_arcsec"],
        dy_arcsec=a["dy_arcsec"],
        gas_sigma_kms=a["gas_sigma_kms"],
    )
    cube_b = sky_cube(
        **kw,
        r_knots_arcsec=np.asarray(b["r_knots_arcsec"]),
        v_knots_kms=np.asarray(b["v_knots_kms"]),
    )
    cube_a = sky_cube(
        **kw,
        v0_kms=a["v0_kms"],
        r_t_arcsec=a["r_t_arcsec"],
    )
    ico_hdr = fits.getheader(ICO)
    hdu_b = _hdu(
        cube_b,
        grid,
        vel,
        ico_hdr,
        {
            "LAM_REG": (float(b["lam_reg"]), "Stage B lambda_reg"),
            "NRINGS": (int(b["n_rings"]), "N rings"),
            "CHI2": (float(b["chi2_map"]), "vis chi2 MAP"),
        },
    )
    hdu_a = _hdu(
        cube_a,
        grid,
        vel,
        ico_hdr,
        {"V0": (float(a["v0_kms"]), "km/s"), "RT": (float(a["r_t_arcsec"]), "arcsec")},
    )
    hdu_a.header["ORIGIN"] = "kinUV Stage A sky_cube"
    dest_b = map_dir / "stage_b_model_cube.fits"
    dest_a = map_dir / "stage_a_model_cube.fits"
    hdu_b.writeto(dest_b, overwrite=True)
    hdu_a.writeto(dest_a, overwrite=True)
    print(f"wrote {dest_b} {hdu_b.data.shape} Jy/pixel native chan", flush=True)
    print(f"wrote {dest_a} {hdu_a.data.shape}", flush=True)


if __name__ == "__main__":
    main()
