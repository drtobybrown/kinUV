#!/usr/bin/env python3
"""Live 3D-Barolo + KinMS image-plane comparators for 066 S3.

Vis chi2 is the fit. quote_inner_slope: false. Do not import from src/kinuv
or scripts. Isolated env only:
/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CUBE_DIR = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/10kms/"
)
CUBE = CUBE_DIR / "KGAS66_clipped_cube.fits"
MOM0 = CUBE_DIR / "KGAS66_Ico_K_kms-1.fits"
MOM1 = CUBE_DIR / "KGAS66_mom1.fits"
MOM2 = CUBE_DIR / "KGAS66_mom2.fits"
DEST = REPO / "docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark"
LIVE = DEST / "live_fitters"
FITTERS = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/external_fitters"
)
NUTS_RT = 0.22392216472996415
NUTS_V0 = 254.9834109292598
NUTS_PA = 200.05
NUTS_I = 43.86
# Official MAP vsys (radio); cube comparators only.
NUTS_VSYS = 8098.773150512066


def _receipt(**extra) -> dict:
    rec = {
        "quote_inner_slope": False,
        "intervals_calibrated": False,
        "leftover_gate": "SB-dominated",
        "note": (
            "vis chi2 is the fit; quote_inner_slope: false. "
            "Cube tools are comparators. Do not quote inner dV/dr."
        ),
    }
    rec.update(extra)
    return rec


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")


def _which_barolo(env_bin: Path | None) -> str | None:
    names = ("BBarolo", "3dbarolo", "bbarolo")
    if env_bin is not None and env_bin.is_dir():
        for name in names:
            cand = env_bin / name
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _kinms_importable() -> tuple[bool, str | None]:
    try:
        import kinms  # noqa: F401

        return True, getattr(sys.modules["kinms"], "__file__", None)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _write_barolo_param(work: Path) -> Path:
    """Minimal 3D-Barolo rings. Writes only under live_fitters/, never the cube tree."""
    param = work / "barolo.param"
    # 10 km/s cube; ~0.5" rings across the inner few beams. Not a kinUV posterior.
    param.write_text(
        "\n".join(
            [
                f"FITSFILE    {CUBE}",
                f"OUTFOLDER   {work / 'out'}",
                "THREADS     4",
                "NRADII      10",
                "RADSEP      0.4",
                f"VSYS        {NUTS_VSYS}",
                f"VROT        {NUTS_V0}",
                f"INC         {NUTS_I}",
                f"PA          {NUTS_PA}",
                "FREE        VROT VDISP",
                "MASK        SEARCH",
                "LINEAR      0",
                "DISTANCE    1",
                "LTYPE       1",
                "FTYPE       2",
                "NORM        LOCAL",
            ]
        )
        + "\n"
    )
    return param


def _run_barolo(exe: str) -> dict:
    work = LIVE / "barolo"
    work.mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    param = _write_barolo_param(work)
    logp = work / "stdout.txt"
    errp = work / "stderr.txt"
    try:
        proc = subprocess.run(
            [exe, "-p", str(param)],
            cwd=str(work),
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        rec = _receipt(
            tool="3D-Barolo",
            ran=False,
            status="failed",
            executable=exe,
            error=str(exc),
        )
        _write(LIVE / "barolo.json", rec)
        return rec
    logp.write_text(proc.stdout or "")
    errp.write_text(proc.stderr or "")
    rings = []
    for cand in (work / "out").rglob("*ring*.txt"):
        rings.append(str(cand))
    for cand in (work / "out").rglob("*.csv"):
        rings.append(str(cand))
    rec = _receipt(
        tool="3D-Barolo",
        ran=proc.returncode == 0,
        status="ran" if proc.returncode == 0 else "failed",
        executable=exe,
        returncode=proc.returncode,
        param=str(param),
        ring_tables=rings,
        kinuv_nuts_r_t_arcsec=NUTS_RT,
        kinuv_nuts_v0_kms=NUTS_V0,
    )
    _write(LIVE / "barolo.json", rec)
    _write(DEST / "barolo.json", rec)
    return rec


def _run_kinms() -> dict:
    ok, detail = _kinms_importable()
    if not ok:
        rec = _receipt(
            tool="KinMS",
            ran=False,
            status="missing",
            error=detail,
        )
        _write(LIVE / "kinms.json", rec)
        _write(DEST / "kinms.json", rec)
        return rec
    work = LIVE / "kinms"
    work.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        from astropy.io import fits
        from kinms import KinMS

        hdr = fits.getheader(CUBE)
        nx = int(hdr["NAXIS1"])
        ny = int(hdr["NAXIS2"])
        nv = int(hdr["NAXIS3"])
        cdelt = abs(float(hdr.get("CDELT1", -5.5e-5))) * 3600.0
        dv = abs(float(hdr.get("CDELT3", 10.0)))
        if dv > 200:
            dv = abs(float(hdr.get("CDELT3", 10.0))) / 1000.0
        r = np.linspace(0.05, 4.0, 40)
        vcirc = NUTS_V0 * (2.0 / np.pi) * np.arctan(r / max(NUTS_RT, 1e-3))
        sb = np.exp(-0.5 * (r / 1.5) ** 2)
        model = KinMS(
            xs=nx * cdelt,
            ys=ny * cdelt,
            vs=nv * dv,
            cellSize=cdelt,
            dv=dv,
            beamSize=[
                float(hdr.get("BMAJ", 0.0002)) * 3600.0,
                float(hdr.get("BMIN", 0.0002)) * 3600.0,
                float(hdr.get("BPA", 0.0)),
            ],
            nSamps=5e5,
        )
        cube = model.model_cube(
            inc=NUTS_I,
            posAng=NUTS_PA,
            vSys=0.0,
            sbProf=sb,
            sbRad=r,
            velProf=vcirc,
            velRad=r,
            gasSigma=8.0,
            intFlux=1.0,
        )
        np.save(work / "kinms_arctan_cube.npy", np.asarray(cube))
        rec = _receipt(
            tool="KinMS",
            ran=True,
            status="ran",
            kinms_module=detail,
            parameterization="arctan V=V0*(2/pi)*arctan(r/r_t) matching kinUV",
            r_t_arcsec=NUTS_RT,
            v0_kms=NUTS_V0,
            pa_deg=NUTS_PA,
            i_deg=NUTS_I,
            cube_npy=str(work / "kinms_arctan_cube.npy"),
        )
    except Exception as exc:
        rec = _receipt(
            tool="KinMS",
            ran=False,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    _write(LIVE / "kinms.json", rec)
    _write(DEST / "kinms.json", rec)
    return rec


def _overlay_plots() -> dict:
    """Major/minor PV and moment slices. Cube vs analytic kinUV arctan (not a fit)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from astropy.io import fits
    except Exception as exc:
        rec = _receipt(status="plot_import_failed", error=str(exc))
        _write(LIVE / "overlays.json", rec)
        return rec
    if not CUBE.is_file():
        rec = _receipt(status="missing_cube")
        _write(LIVE / "overlays.json", rec)
        return rec
    cube = np.asarray(fits.getdata(CUBE), dtype=np.float64)
    if cube.ndim != 3:
        rec = _receipt(status="cube_not_3d")
        _write(LIVE / "overlays.json", rec)
        return rec
    hdr = fits.getheader(CUBE)
    nv, ny, nx = cube.shape
    x = np.arange(nx) - (nx * 0.5)
    y = np.arange(ny) - (ny * 0.5)
    xx, yy = np.meshgrid(x, y)
    cell = abs(float(hdr.get("CDELT1", -5.5e-5))) * 3600.0
    xx_as = xx * cell
    yy_as = yy * cell
    pa = np.deg2rad(NUTS_PA)
    xmaj = xx_as * np.sin(pa) + yy_as * np.cos(pa)
    xmin = xx_as * np.cos(pa) - yy_as * np.sin(pa)
    vel = (
        float(hdr.get("CRVAL3", 0.0))
        + (np.arange(nv) + 1 - float(hdr.get("CRPIX3", 1.0)))
        * float(hdr.get("CDELT3", 10.0))
    )
    if abs(vel[1] - vel[0]) > 200:
        vel = vel / 1000.0
    # Collapse a 0.4" slit on the major/minor axes.
    slit = 0.4
    maj_mask = np.abs(xmin) <= slit
    min_mask = np.abs(xmaj) <= slit
    pv_maj = cube[:, maj_mask].mean(axis=1) if maj_mask.any() else np.zeros((nv, 1))
    pv_min = cube[:, min_mask].mean(axis=1) if min_mask.any() else np.zeros((nv, 1))
    r_maj = xmaj[maj_mask] if maj_mask.any() else np.array([0.0])
    # Moment slices
    mom0 = np.asarray(fits.getdata(MOM0), dtype=np.float64) if MOM0.is_file() else cube.sum(0)
    mom1 = np.asarray(fits.getdata(MOM1), dtype=np.float64) if MOM1.is_file() else np.zeros_like(mom0)
    mom2 = np.asarray(fits.getdata(MOM2), dtype=np.float64) if MOM2.is_file() else np.zeros_like(mom0)
    rgrid = np.sqrt(xx_as**2 + yy_as**2)
    v_arctan = NUTS_V0 * (2.0 / np.pi) * np.arctan(rgrid / max(NUTS_RT, 1e-3))
    vlos = v_arctan * np.sin(np.deg2rad(NUTS_I)) * np.cos(
        np.arctan2(xmin, xmaj)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    if pv_maj.size and r_maj.size:
        # Approximate PV image: mean spectrum vs sorted offset bins
        rb = np.linspace(-3.0, 3.0, 40)
        img = np.zeros((nv, rb.size - 1))
        for i in range(rb.size - 1):
            sel = (r_maj >= rb[i]) & (r_maj < rb[i + 1])
            if sel.any():
                img[:, i] = cube[:, maj_mask][:, sel].mean(axis=1)
        axes[0].imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[rb[0], rb[-1], vel[0], vel[-1]],
            cmap="gray_r",
        )
        axes[0].plot(
            np.linspace(-3, 3, 80),
            NUTS_VSYS
            + NUTS_V0
            * (2 / np.pi)
            * np.arctan(np.abs(np.linspace(-3, 3, 80)) / max(NUTS_RT, 1e-3))
            * np.sin(np.deg2rad(NUTS_I))
            * np.sign(np.linspace(-3, 3, 80)),
            "C1-",
            lw=1.2,
            label="kinUV arctan (NUTS mean; not a slope quote)",
        )
    axes[0].set_title("major-axis PV (cube vs kinUV arctan)")
    axes[0].set_xlabel("offset (arcsec)")
    axes[0].set_ylabel("v (km/s)")
    if pv_min.size:
        rb = np.linspace(-3.0, 3.0, 40)
        rmin = xmin[min_mask] if min_mask.any() else np.array([0.0])
        img = np.zeros((nv, rb.size - 1))
        for i in range(rb.size - 1):
            sel = (rmin >= rb[i]) & (rmin < rb[i + 1])
            if sel.any():
                img[:, i] = cube[:, min_mask][:, sel].mean(axis=1)
        axes[1].imshow(
            img,
            origin="lower",
            aspect="auto",
            extent=[rb[0], rb[-1], vel[0], vel[-1]],
            cmap="gray_r",
        )
    axes[1].axhline(NUTS_VSYS, color="C1", lw=1.2)
    axes[1].set_title("minor-axis PV")
    axes[1].set_xlabel("offset (arcsec)")
    fig.savefig(LIVE / "pv_major_minor.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for ax, arr, title in (
        (axes[0], mom0, "moment 0"),
        (axes[1], mom1, "moment 1"),
        (axes[2], mom2, "moment 2"),
    ):
        finite = np.asarray(arr, dtype=np.float64)
        finite = np.where(np.isfinite(finite), finite, np.nan)
        ax.imshow(finite, origin="lower", cmap="gray_r")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].contour(
        vlos,
        levels=[-80, -40, 0, 40, 80],
        colors="C1",
        linewidths=0.6,
        origin="lower",
    )
    fig.savefig(LIVE / "moments_slices.png", dpi=120)
    plt.close(fig)
    rec = _receipt(
        status="wrote_overlays",
        pv=str(LIVE / "pv_major_minor.png"),
        moments=str(LIVE / "moments_slices.png"),
        kinuv_nuts_r_t_arcsec=NUTS_RT,
    )
    _write(LIVE / "overlays.json", rec)
    return rec


def _rebuild_s3(barolo: dict, kin: dict) -> None:
    table_path = DEST / "s3_table.json"
    table = json.loads(table_path.read_text()) if table_path.is_file() else {}
    table["likelihood"] = (
        "vis chi2 = s * sum w |d-m|^2 on 881x95; cube fitters are comparators"
    )
    table["header_note"] = "vis chi2 is the fit; quote_inner_slope: false"
    table["leftover_gate"] = "SB-dominated"
    table["quote_inner_slope"] = False
    table["intervals_calibrated"] = False
    table["nuts_mean_r_t_arcsec"] = NUTS_RT
    table["nuts_mean_v0_kms"] = NUTS_V0
    table["live_fitters"] = {
        "barolo": barolo,
        "kinms": kin,
        "dir": str(LIVE),
    }
    table["barolo"] = barolo
    table["kinms"] = kin
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    readme = DEST / "README.md"
    first = (
        "Vis χ² is the fit; quote_inner_slope: false. "
        "KinMS/Barolo are image-plane comparators, not a kinUV likelihood.\n"
    )
    extra = (
        f"\n## Live fitters (2026-09-05)\n\n"
        f"- 3D-Barolo: `{barolo.get('status')}`\n"
        f"- KinMS: `{kin.get('status')}`\n"
        f"- Overlays: `live_fitters/pv_major_minor.png`, "
        f"`live_fitters/moments_slices.png`\n"
        f"- Isolated env: `{FITTERS}`\n"
        f"- kinUV NUTS mean r_t = {NUTS_RT:.3f} arcsec (not a quoted inner scale)\n"
    )
    if readme.is_file():
        text = readme.read_text()
        if not text.startswith("Vis"):
            text = first + "\n" + text
        if "## Live fitters" not in text:
            text = text.rstrip() + extra
        readme.write_text(text)
    else:
        readme.write_text(first + extra)
    (LIVE / "README.md").write_text(
        first
        + extra
        + "\nOfficial MAP `kinuv-KGAS066-uvsign-map` was not written. Do not start G4.\n"
    )


def main() -> int:
    LIVE.mkdir(parents=True, exist_ok=True)
    env_bin = None
    for cand in (FITTERS / "venv" / "bin", FITTERS / "conda" / "bin"):
        if cand.is_dir():
            env_bin = cand
            if str(cand) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
            break
    exe = _which_barolo(env_bin)
    barolo = (
        _run_barolo(exe)
        if exe
        else _receipt(
            tool="3D-Barolo",
            ran=False,
            status="missing_on_path",
            executable=None,
            isolated_env=str(FITTERS),
        )
    )
    if not exe:
        _write(LIVE / "barolo.json", barolo)
        _write(DEST / "barolo.json", barolo)
    kin = _run_kinms()
    overlays = _overlay_plots()
    _rebuild_s3(barolo, kin)
    _write(
        LIVE / "summary.json",
        _receipt(
            barolo=barolo.get("status"),
            kinms=kin.get("status"),
            overlays=overlays.get("status"),
            env_bin=str(env_bin) if env_bin else None,
        ),
    )
    print(
        json.dumps(
            {
                "live": str(LIVE),
                "barolo": barolo.get("status"),
                "kinms": kin.get("status"),
                "overlays": overlays.get("status"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
