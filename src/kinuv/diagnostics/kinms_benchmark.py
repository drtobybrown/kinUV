"""Downstream cube diagnostics for kinUV and KinMS products.

The comparison is deliberately outside the visibility likelihood. It consumes
completed model cubes, places them on the official image-cube grid, and writes
matched diagnostic products. No result from this module can alter a fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kinuv.constants import C_LIGHT_KM_S, F_REST_CO21_HZ, JY_W_M2_HZ, K_BOLTZMANN_J_K
from kinuv.diagnostics.imaging import masked_moments, pv_diagram, spectral_axis_kms
from kinuv.geometry import sky_to_galaxy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cube_metrics(data, model, mask) -> dict[str, float | int]:
    """Return matched-voxel residual metrics without refitting an amplitude."""
    d = np.asarray(data, dtype=np.float64)
    m = np.asarray(model, dtype=np.float64)
    use = np.asarray(mask, dtype=bool) & np.isfinite(d) & np.isfinite(m)
    if d.shape != m.shape or use.shape != d.shape:
        raise ValueError("data, model, and mask must share a cube shape")
    if not np.any(use):
        raise ValueError("benchmark mask selects no finite voxels")
    residual = d[use] - m[use]
    data_rms = float(np.sqrt(np.mean(d[use] ** 2)))
    return {
        "n_voxel": int(np.sum(use)),
        "residual_rms_k": float(np.sqrt(np.mean(residual**2))),
        "residual_mae_k": float(np.mean(np.abs(residual))),
        "normalized_rmse": float(np.sqrt(np.mean(residual**2)) / data_rms)
        if data_rms > 0.0
        else float("nan"),
        "model_to_data_flux": float(np.sum(m[use]) / np.sum(d[use]))
        if np.sum(d[use]) != 0.0
        else float("nan"),
    }


def aperture_spectrum(cube, mask3d) -> np.ndarray:
    """Integrated spectrum on the spatial footprint of a 3-D source mask."""
    a = np.asarray(cube, dtype=np.float64)
    footprint = np.any(np.asarray(mask3d, dtype=bool), axis=0)
    return np.nansum(np.where(footprint[None, :, :], a, np.nan), axis=(1, 2))


def major_axis_rotation_profile(
    moment0,
    moment1,
    header,
    *,
    pa_deg: float,
    inclination_deg: float,
    vsys_kms: float,
    dx_arcsec: float = 0.0,
    dy_arcsec: float = 0.0,
    slit_half_width_arcsec: float = 0.5,
    radius_max_arcsec: float = 8.0,
    n_bin: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust moment-1 rotation profile for image-domain comparison only."""
    m0 = np.asarray(moment0, dtype=np.float64)
    m1 = np.asarray(moment1, dtype=np.float64)
    ny, nx = m0.shape
    x = (np.arange(nx) + 1.0 - float(header["CRPIX1"])) * float(header["CDELT1"]) * 3600.0
    y = (np.arange(ny) + 1.0 - float(header["CRPIX2"])) * float(header["CDELT2"]) * 3600.0
    east = -x if float(header["CDELT1"]) < 0.0 else x
    xe, yn = np.meshgrid(east - dx_arcsec, y - dy_arcsec, indexing="xy")
    xg, yg = sky_to_galaxy(
        xe, yn, np.radians(float(pa_deg)), np.radians(float(inclination_deg))
    )
    radius = np.hypot(xg, yg)
    cos_theta = np.divide(xg, radius, out=np.zeros_like(radius), where=radius > 0.0)
    denom = np.sin(np.radians(float(inclination_deg))) * cos_theta
    vc = np.divide(
        m1 - float(vsys_kms), denom, out=np.full_like(m1, np.nan), where=np.abs(denom) > 0.15
    )
    threshold = 0.05 * float(np.nanmax(m0))
    use = (
        np.isfinite(vc)
        & np.isfinite(m0)
        & (m0 > threshold)
        & (np.abs(yg) <= float(slit_half_width_arcsec))
    )
    edges = np.linspace(0.0, float(radius_max_arcsec), int(n_bin) + 1)
    out_r: list[float] = []
    out_v: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pick = use & (radius >= lo) & (radius < hi)
        if np.count_nonzero(pick) >= 3:
            out_r.append(0.5 * (lo + hi))
            out_v.append(float(np.nanmedian(vc[pick])))
    return np.asarray(out_r), np.asarray(out_v)


def jy_beam_to_kelvin(cube, header, vel_kms) -> np.ndarray:
    """Convert a matched Jy/beam cube to Rayleigh-Jeans kelvin."""
    rest = float(header.get("RESTFRQ", F_REST_CO21_HZ))
    nu = rest / (1.0 + float(np.median(vel_kms)) / C_LIGHT_KM_S)
    theta_maj = float(header["BMAJ"]) * np.pi / 180.0
    theta_min = float(header["BMIN"]) * np.pi / 180.0
    omega = np.pi * theta_maj * theta_min / (4.0 * np.log(2.0))
    jy_per_beam_per_k = (2.0 * K_BOLTZMANN_J_K * nu**2 / (C_LIGHT_KM_S * 1e3) ** 2) * omega / JY_W_M2_HZ
    return np.asarray(cube, dtype=np.float64) / jy_per_beam_per_k


def _moment_products(cube, vel, mask, dv):
    m0, m1, m2 = masked_moments(cube, vel, mask, dv)
    return {"moment0": m0, "moment1": m1, "moment2": m2}


def _plot_moments(products, out: Path, target_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(15, 9), constrained_layout=True)
    columns = ("data", "kinuv", "kinms", "data_minus_kinuv", "data_minus_kinms")
    labels = ("Data", "kinUV", "KinMS", "Data - kinUV", "Data - KinMS")
    rows = ("moment0", "moment1", "moment2")
    for i, key in enumerate(rows):
        base = [products[name][key] for name in columns[:3]]
        residuals = [base[0] - base[1], base[0] - base[2]]
        finite = np.concatenate([x[np.isfinite(x)] for x in base])
        lo, hi = np.nanpercentile(finite, (2, 98))
        rmax = max(float(np.nanpercentile(np.abs(x[np.isfinite(x)]), 98)) for x in residuals)
        for j, image in enumerate((*base, *residuals)):
            kw = {"origin": "lower", "aspect": "equal", "cmap": "RdBu_r" if j >= 3 else "viridis"}
            if j >= 3:
                kw.update(vmin=-rmax, vmax=rmax)
            else:
                kw.update(vmin=lo, vmax=hi)
            im = axes[i, j].imshow(image, **kw)
            fig.colorbar(im, ax=axes[i, j], fraction=0.046)
            if i == 0:
                axes[i, j].set_title(labels[j])
            if j == 0:
                axes[i, j].set_ylabel(key)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(f"{target_id}: official cube vs visibility-fit kinUV and cube-fit KinMS")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_spectra(vel, spectra, out: Path, target_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for label, colour in (("data", "black"), ("kinuv", "#4477AA"), ("kinms", "#CC6677")):
        ax.plot(vel, spectra[label], label=label, color=colour)
    ax.set(xlabel="Velocity (km/s)", ylabel="Aperture sum (K)", title=f"{target_id}: integrated spectrum")
    ax.legend()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_pv(vel, offset, pvs, out: Path, target_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), constrained_layout=True, sharey=True)
    images = (pvs["data"], pvs["kinuv"], pvs["kinms"], pvs["data"] - pvs["kinuv"], pvs["data"] - pvs["kinms"])
    labels = ("Data", "kinUV", "KinMS", "Data - kinUV", "Data - KinMS")
    vmax = max(float(np.nanpercentile(np.abs(x), 99)) for x in images)
    extent = [float(offset[0]), float(offset[-1]), float(vel[0]), float(vel[-1])]
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("Major-axis offset (arcsec)")
    axes[0].set_ylabel("Velocity (km/s)")
    fig.suptitle(f"{target_id}: major-axis PVD")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_rotation(profiles, out: Path, target_id: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    for label, colour in (("data", "black"), ("kinuv", "#4477AA"), ("kinms", "#CC6677")):
        radius, speed = profiles[label]
        ax.plot(radius, speed, marker="o", ms=3, label=label, color=colour)
    ax.set(xlabel="Angular radius (arcsec)", ylabel="Circular speed (km/s)", title=f"{target_id}: moment-1 rotation profiles")
    ax.legend()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_channels(vel, cubes, mask, out: Path, target_id: str, n_channel: int = 5) -> None:
    import matplotlib.pyplot as plt

    signal = np.abs(aperture_spectrum(cubes["data"], mask))
    active = np.flatnonzero(signal > 0.1 * np.nanmax(signal))
    if active.size == 0:
        active = np.arange(cubes["data"].shape[0])
    idx = np.unique(np.linspace(active[0], active[-1], n_channel).round().astype(int))
    rows = ("data", "kinuv", "kinms", "data_minus_kinuv", "data_minus_kinms")
    images = {
        **cubes,
        "data_minus_kinuv": cubes["data"] - cubes["kinuv"],
        "data_minus_kinms": cubes["data"] - cubes["kinms"],
    }
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(3 * len(idx), 12), constrained_layout=True)
    axes = np.asarray(axes, dtype=object).reshape(len(rows), len(idx))
    for i, row in enumerate(rows):
        for j, chan in enumerate(idx):
            image = images[row][chan]
            vmax = float(np.nanpercentile(np.abs(image), 99))
            axes[i, j].imshow(image, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"{vel[chan]:.1f} km/s")
            if j == 0:
                axes[i, j].set_ylabel(row.replace("_", " "))
    fig.suptitle(f"{target_id}: matched channel maps")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_cube_benchmark(
    *,
    target_id: str,
    data_cube: Path,
    mask_cube: Path,
    kinuv_cube: Path,
    kinms_cube: Path,
    output_dir: Path,
    pa_deg: float,
    inclination_deg: float,
    vsys_kms: float,
    dx_arcsec: float = 0.0,
    dy_arcsec: float = 0.0,
) -> dict:
    """Write the complete downstream comparison for one completed target."""
    from astropy.io import fits

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(data_cube, memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    mask = np.asarray(fits.getdata(mask_cube), dtype=np.float64) > 0.5
    kinuv = np.asarray(fits.getdata(kinuv_cube), dtype=np.float64)
    with fits.open(kinms_cube, memmap=False) as hdul:
        kinms = np.asarray(hdul[0].data, dtype=np.float64)
        kinms_header = hdul[0].header.copy()
    if not (data.shape == mask.shape == kinuv.shape == kinms.shape):
        raise ValueError(
            f"matched cube shapes required; data={data.shape}, mask={mask.shape}, "
            f"kinuv={kinuv.shape}, kinms={kinms.shape}"
        )
    vel = spectral_axis_kms(header)
    unit = str(kinms_header.get("BUNIT", "")).lower().replace(" ", "")
    if "jy/beam" in unit or "jybeam" in unit:
        kinms = jy_beam_to_kelvin(kinms, header, vel)
    elif unit not in {"k", "kelvin"}:
        raise ValueError(f"unsupported KinMS cube BUNIT={kinms_header.get('BUNIT')!r}")
    dv = float(np.median(np.abs(np.diff(vel))))
    cubes = {"data": data, "kinuv": kinuv, "kinms": kinms}
    moments = {name: _moment_products(cube, vel, mask, dv) for name, cube in cubes.items()}
    moments["data_minus_kinuv"] = {key: moments["data"][key] - moments["kinuv"][key] for key in moments["data"]}
    moments["data_minus_kinms"] = {key: moments["data"][key] - moments["kinms"][key] for key in moments["data"]}
    spectra = {name: aperture_spectrum(cube, mask) for name, cube in cubes.items()}
    ra = float(header["CRVAL1"])
    dec = float(header["CRVAL2"])
    pvs = {}
    offset = None
    for name, cube in cubes.items():
        pvs[name], offset = pv_diagram(cube, header, ra, dec, pa_deg, 16.0, 0.8)
    profiles = {
        name: major_axis_rotation_profile(
            moments[name]["moment0"], moments[name]["moment1"], header,
            pa_deg=pa_deg, inclination_deg=inclination_deg, vsys_kms=vsys_kms,
            dx_arcsec=dx_arcsec, dy_arcsec=dy_arcsec,
        )
        for name in cubes
    }
    out_header = header.copy()
    out_header["BUNIT"] = "K"
    out_header["ORIGIN"] = "kinUV downstream KinMS benchmark"
    fits.PrimaryHDU(kinms.astype(np.float32), out_header).writeto(output_dir / "kinms_model_k.fits", overwrite=True)
    np.savez(output_dir / "moments.npz", **{f"{name}_{key}": value for name, item in moments.items() for key, value in item.items()})
    np.savez(
        output_dir / "profiles.npz",
        velocity_kms=vel,
        pv_offset_arcsec=offset,
        **{f"spectrum_{name}": value for name, value in spectra.items()},
        **{f"pv_{name}": value for name, value in pvs.items()},
        **{f"rotation_radius_{name}": value[0] for name, value in profiles.items()},
        **{f"rotation_speed_{name}": value[1] for name, value in profiles.items()},
    )
    _plot_moments(moments, output_dir / "moments_comparison.png", target_id)
    _plot_spectra(vel, spectra, output_dir / "spectra_comparison.png", target_id)
    _plot_pv(vel, offset, pvs, output_dir / "pvd_major_comparison.png", target_id)
    _plot_rotation(profiles, output_dir / "rotation_curve_comparison.png", target_id)
    _plot_channels(vel, cubes, mask, output_dir / "channel_maps_comparison.png", target_id)
    result = {
        "schema_version": "kinuv-kinms-benchmark-v1",
        "target_id": target_id,
        "fit_domain": "kinUV likelihood is visibility chi2; KinMS is a downstream image-cube comparator",
        "inputs": {
            "data_cube": {"path": str(data_cube), "sha256": _sha256(data_cube)},
            "mask_cube": {"path": str(mask_cube), "sha256": _sha256(mask_cube)},
            "kinuv_cube": {"path": str(kinuv_cube), "sha256": _sha256(kinuv_cube)},
            "kinms_cube": {"path": str(kinms_cube), "sha256": _sha256(kinms_cube)},
        },
        "geometry": {
            "pa_deg": float(pa_deg),
            "inclination_deg": float(inclination_deg),
            "vsys_kms": float(vsys_kms),
            "dx_arcsec": float(dx_arcsec),
            "dy_arcsec": float(dy_arcsec),
        },
        "metrics": {
            "kinuv": cube_metrics(data, kinuv, mask),
            "kinms": cube_metrics(data, kinms, mask),
        },
        "products": [
            "kinms_model_k.fits", "moments.npz", "profiles.npz",
            "moments_comparison.png", "spectra_comparison.png",
            "pvd_major_comparison.png", "rotation_curve_comparison.png",
            "channel_maps_comparison.png",
        ],
    }
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


__all__ = [
    "aperture_spectrum", "cube_metrics", "jy_beam_to_kelvin",
    "major_axis_rotation_profile", "write_cube_benchmark",
]
