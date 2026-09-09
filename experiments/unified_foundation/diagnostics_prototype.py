#!/usr/bin/env python3
"""Scratch adapter and renderer for the unified diagnostic contract.

This module reads frozen production products.  It does not fit, rescore, or
write below ``results/production``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits

from kinuv.diagnostics.imaging import offset_world, pv_diagram, spectral_axis_kms
from kinuv.diagnostics.kinms_benchmark import fits_sky_offsets_arcsec


SCHEMA_VERSION = "kinuv-unified-diagnostics-v0"
PROFILE_FIELDS = (
    "radius_arcsec",
    "u_projected_map_kms",
    "vc_intrinsic_map_kms",
    "sigma_map_kms",
    "du_dradius_map_kms_per_arcsec",
    "dvc_dradius_map_kms_per_arcsec",
    "dsigma_dradius_map_kms_per_arcsec",
    "du_dradius_q16_kms_per_arcsec",
    "du_dradius_q50_kms_per_arcsec",
    "du_dradius_q84_kms_per_arcsec",
    "dvc_dradius_q16_kms_per_arcsec",
    "dvc_dradius_q50_kms_per_arcsec",
    "dvc_dradius_q84_kms_per_arcsec",
    "dsigma_dradius_q16_kms_per_arcsec",
    "dsigma_dradius_q50_kms_per_arcsec",
    "dsigma_dradius_q84_kms_per_arcsec",
    "u_projected_q16_kms",
    "u_projected_q50_kms",
    "u_projected_q84_kms",
    "vc_intrinsic_q16_kms",
    "vc_intrinsic_q50_kms",
    "vc_intrinsic_q84_kms",
    "sigma_q16_kms",
    "sigma_q50_kms",
    "sigma_q84_kms",
    "measured_support",
    "subbeam",
    "derivative_supported",
    "kappa2_negative",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def first_rising_half_crossing(radius, profile, reference, *, reference_constrained):
    """Return R50 only for one in-support rising half crossing."""

    if not reference_constrained:
        return {"status": "unresolved", "value_arcsec": None, "reason": "outer_reference_not_demonstrably_constrained"}
    radius = np.asarray(radius, dtype=float)
    profile = np.asarray(profile, dtype=float)
    half = 0.5 * float(reference)
    rising = np.flatnonzero((profile[:-1] < half) & (profile[1:] >= half))
    if rising.size == 0:
        return {"status": "unresolved", "value_arcsec": None, "reason": "no_rising_half_crossing_in_measured_support"}
    if rising.size > 1:
        return {"status": "nonunique", "value_arcsec": None, "reason": "multiple_rising_half_crossings"}
    i = int(rising[0])
    dr = radius[i + 1] - radius[i]
    du = profile[i + 1] - profile[i]
    value = radius[i] + dr * (half - profile[i]) / du
    return {"status": "resolved", "value_arcsec": float(value), "reason": None}


def _piecewise_draws(radius, knot_radius, values):
    rows = []
    for value in values:
        curve = np.interp(radius, knot_radius, value, left=value[0], right=value[-1])
        curve[radius < knot_radius[0]] = value[0] * radius[radius < knot_radius[0]] / knot_radius[0]
        rows.append(curve)
    return np.asarray(rows)


def _posterior_bands(target_root: Path, radius, knot_radius):
    status_path = target_root / "nuts/status.json"
    posterior_path = target_root / "nuts/posterior_samples.npz"
    if not status_path.is_file() or load_json(status_path).get("state") != "ACCEPTED" or not posterior_path.is_file():
        return {}, {"status": "missing", "reason": "posterior_for_selected_checkpoint_not_accepted"}
    with np.load(posterior_path, allow_pickle=False) as archive:
        names = set(archive.files)
        u_names = sorted(name for name in names if name.startswith("u_knot_") and name.endswith("_kms"))
        v_names = sorted(name for name in names if name.startswith("v_knot_") and name.endswith("_kms"))
        if len(u_names) != len(knot_radius) or len(v_names) != len(knot_radius):
            return {}, {"status": "missing", "reason": "legacy_posterior_has_no_supported_common_profile_evaluator"}
        u = np.stack([archive[name].reshape(-1) for name in u_names], axis=1)
        v = np.stack([archive[name].reshape(-1) for name in v_names], axis=1)
        sigma = archive["sigma_inner_kms"].reshape(-1)
        geometry = {
            key: np.percentile(archive[key].reshape(-1), (16, 50, 84)).tolist()
            for key in ("pa_deg", "inclination_deg", "dx_arcsec", "dy_arcsec")
            if key in names
        }
    u_draw = _piecewise_draws(radius, knot_radius, u)
    v_draw = _piecewise_draws(radius, knot_radius, v)
    s_draw = np.broadcast_to(sigma[:, None], u_draw.shape)
    bands = {}
    for prefix, draws in (("u_projected", u_draw), ("vc_intrinsic", v_draw), ("sigma", s_draw)):
        for quantile, value in zip(("q16", "q50", "q84"), np.percentile(draws, (16, 50, 84), axis=0)):
            bands[f"{prefix}_{quantile}_kms"] = value
    for prefix, draws in (("du", u_draw), ("dvc", v_draw), ("dsigma", s_draw)):
        derivatives = np.gradient(draws, radius, axis=1, edge_order=2)
        for quantile, value in zip(("q16", "q50", "q84"), np.percentile(derivatives, (16, 50, 84), axis=0)):
            bands[f"{prefix}_dradius_{quantile}_kms_per_arcsec"] = value
    covariance_path = target_root / "nuts/covariance.npz"
    return bands, {
        "status": "accepted_conditional",
        "reason": None,
        "source": source(posterior_path),
        "covariance_source": source(covariance_path) if covariance_path.is_file() else None,
        "geometry_q16_q50_q84": geometry,
        "conditionality": ["fixed_empirical_emissivity", "selected_local_mode"],
    }


def adapt_target(target_root: Path, output: Path) -> dict:
    started = time.perf_counter()
    best = target_root / "best_model"
    params_path = best / "parameters.json"
    checkpoint_path = best / "checkpoint.json"
    curve_path = best / "rotation_curve.npz"
    config_path = best / "config.json"
    replay_path = best / "replay.json"
    moments_path = target_root / "benchmarks/moments.npz"
    model_path = best / "model_on_science_grid.fits"
    kinms_path = target_root / "benchmarks/kinms_model_k.fits"
    parameters = load_json(params_path)
    checkpoint = load_json(checkpoint_path)
    config = load_json(config_path)
    replay = load_json(replay_path)
    fitted = parameters["fitted_native_parameters"]
    with np.load(curve_path, allow_pickle=False) as archive:
        radius = np.asarray(archive["radius_arcsec"], dtype=float)
        projected = np.asarray(archive["projected_speed_kms"], dtype=float)
        intrinsic = np.asarray(archive["intrinsic_speed_kms"], dtype=float)
    knot_radius = np.asarray(checkpoint["velocity_support"]["knot_radii_arcsec"], dtype=float)
    beam = float(checkpoint["bmaj_arcsec"])
    selected_family = parameters["candidate"]

    # This is the only legacy-family adaptation.  Unified consumers below use
    # the same sigma field and never inspect this provenance label.
    if selected_family == "two_zone_dispersion":
        transition, width = knot_radius[1], 0.25 * beam
        fraction = 0.5 * (1.0 + np.tanh((radius - transition) / width))
        sigma = fitted["sigma_inner_kms"] * (1.0 - fraction) + fitted["sigma_outer_kms"] * fraction
        sigma_source = "legacy_exact_two_zone_tanh"
    else:
        sigma = np.full_like(radius, fitted["sigma_inner_kms"])
        sigma_source = "legacy_constant"

    derivative_supported = np.ones(radius.shape, dtype=bool)
    regularity = "smooth_analytic"
    if selected_family == "supported_rings":
        regularity = "C0_piecewise_linear_derivative_jumps_at_knots"
        for knot in knot_radius:
            derivative_supported[np.argmin(np.abs(radius - knot))] = False
    du = np.gradient(projected, radius, edge_order=2)
    dvc = np.gradient(intrinsic, radius, edge_order=2)
    dsigma = np.gradient(sigma, radius, edge_order=2)
    v_over_r = np.divide(intrinsic, radius, out=np.full_like(radius, dvc[0]), where=radius > 0)
    kappa2 = 2.0 * v_over_r * (v_over_r + dvc)
    bands, posterior = _posterior_bands(target_root, radius, knot_radius)
    arrays = {
        "radius_arcsec": radius,
        "u_projected_map_kms": projected,
        "vc_intrinsic_map_kms": intrinsic,
        "sigma_map_kms": sigma,
        "du_dradius_map_kms_per_arcsec": du,
        "dvc_dradius_map_kms_per_arcsec": dvc,
        "dsigma_dradius_map_kms_per_arcsec": dsigma,
        "measured_support": radius <= float(checkpoint["velocity_support"]["r95_arcsec"]),
        "subbeam": radius < beam,
        "derivative_supported": derivative_supported,
        "kappa2_negative": kappa2 < 0.0,
    }
    for name in PROFILE_FIELDS:
        if name not in arrays:
            arrays[name] = bands.get(name, np.full_like(radius, np.nan))
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "profiles.npz"
    np.savez(profile_path, **arrays)

    if selected_family == "supported_rings":
        turnover = first_rising_half_crossing(radius, projected, projected[-1], reference_constrained=False)
        turnover["reference_kind"] = "forced_flat_after_last_supported_knot"
    else:
        turnover = first_rising_half_crossing(
            radius, projected, fitted["arctan_u_kms"], reference_constrained=True
        )
        turnover.update({
            "reference_kind": "fitted_arctan_asymptote",
            "analytic_value_arcsec": float(fitted["turnover_over_bmaj"]) * beam,
        })
    cube_data = Path(config["diagnostic_cube"])
    mask_path = Path(config["diagnostic_mask"])
    geometry = parameters["diagnostic_geometry_lsrk_optical"]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "artifact_state": "ADAPTER_PROTOTYPE",
        "target_id": parameters.get("target_id", target_root.name),
        "inference": {
            "point_stage": "MAP",
            "posterior": posterior,
            "likelihood": "visibility_chi2",
            "image_products_role": "supporting_diagnostics",
        },
        "legacy_adapter": {
            "source_family": selected_family,
            "kinematic_regularity": regularity,
            "dispersion_source": sigma_source,
            "does_not_refit_or_smooth_profile": True,
        },
        "geometry": {
            "frame": "optical_LSRK",
            "pa_convention": "east_of_north_receding",
            "map": geometry,
            "bands": posterior.get("geometry_q16_q50_q84"),
            "vsys_band_missing_reason": "posterior_saved_in_native_frame" if posterior["status"] != "missing" else posterior["reason"],
        },
        "profiles": {
            "array_file": "profiles.npz",
            "fields": list(PROFILE_FIELDS),
            "velocity_convention": "native_TOPO_radio_velocity_difference",
            "reporting_transform": {
                "kind": "transform_each_vsys_plus_or_minus_u_endpoint_to_LSRK_optical",
                "native_vsys_radio_topo_kms": float(fitted["vsys_kms"]),
                "frequency_equivalent_correction_kms": float(replay["frame"]["frequency_equivalent_correction_kms"]),
            },
            "band_semantics": "samplewise_profile_q16_q50_q84" if bands else "NaN_missing",
            "turnover": {
                "definition": "first unique rising crossing of half a demonstrably constrained outer/asymptotic reference",
                **turnover,
            },
            "resolution": {"bmaj_arcsec": beam, "measured_radius_max_arcsec": float(knot_radius[-1])},
            "flags": {
                "inclination_conditional": True,
                "derivative_discontinuity_present": bool(np.any(~derivative_supported)),
                "kappa2_negative_present": bool(np.any(kappa2 < 0.0)),
                "kappa2_interpretation": "shape diagnostic only; no mass or gravitational-circular-speed claim",
            },
        },
        "diagnostic_inputs": {
            "moments": source(moments_path),
            "data_cube": source(cube_data),
            "model_cube": source(model_path),
            "benchmark_cube": source(kinms_path),
            "frozen_mask": source(mask_path),
            "mask_semantics": "existing canonical mask; renderer derives no new residual-hiding mask",
        },
        "sources": {"parameters": source(params_path), "checkpoint": source(checkpoint_path), "rotation_curve": source(curve_path), "replay": source(replay_path)},
        "adapter_seconds": time.perf_counter() - started,
    }
    contract_path = output / "contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def validate_contract(contract_path: Path):
    contract = load_json(contract_path)
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported diagnostics schema")
    profile_path = contract_path.parent / contract["profiles"]["array_file"]
    with np.load(profile_path, allow_pickle=False) as archive:
        missing = set(PROFILE_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"missing common profile fields: {sorted(missing)}")
        n = archive["radius_arcsec"].size
        if any(archive[name].shape != (n,) for name in PROFILE_FIELDS):
            raise ValueError("profile fields do not share the common radial grid")
        if not np.all(np.diff(archive["radius_arcsec"]) > 0):
            raise ValueError("radius must increase")
    return contract


def _load_cube(path):
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=float).squeeze(), hdul[0].header.copy()


def _navy_teal():
    return LinearSegmentedColormap.from_list("kinuv_navy_teal", ("#06182b", "#124e66", "#0f7c82", "#45b8ac", "#d4f3ee")).with_extremes(bad="white")


def _residual_cmap():
    colours = plt.get_cmap("RdBu_r")(np.linspace(0, 1, 257)); colours[128] = (1, 1, 1, 1)
    return ListedColormap(colours).with_extremes(bad="white")


def _limits(images, symmetric=False):
    values = np.concatenate([np.asarray(image)[np.isfinite(image)] for image in images])
    span = float(np.percentile(np.abs(values), 97.5)) if symmetric else float(np.percentile(values, 99))
    return (-span, span) if symmetric else (0.0, span)


def _crop(moment0, header, geometry, beam):
    east, north = fits_sky_offsets_arcsec(header, moment0.shape)
    support = np.isfinite(moment0) & (moment0 > 0.05 * np.nanmax(moment0))
    radius = max(np.max(np.abs(east[support] - geometry["dx_arcsec"])), np.max(np.abs(north[support] - geometry["dy_arcsec"])))
    return float(max(3 * beam, 1.15 * radius + beam))


def render_contract(contract_path: Path, output: Path):
    started = time.perf_counter(); contract = validate_contract(contract_path)
    inputs = contract["diagnostic_inputs"]; geometry = contract["geometry"]["map"]
    data, header = _load_cube(inputs["data_cube"]["path"])
    model, _ = _load_cube(inputs["model_cube"]["path"])
    benchmark, _ = _load_cube(inputs["benchmark_cube"]["path"])
    mask, _ = _load_cube(inputs["frozen_mask"]["path"]); mask = mask > 0.5
    with np.load(inputs["moments"]["path"], allow_pickle=False) as archive:
        moments = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(contract_path.parent / contract["profiles"]["array_file"], allow_pickle=False) as archive:
        profiles = {name: np.asarray(archive[name]) for name in archive.files}
    output.mkdir(parents=True, exist_ok=True)

    # 3x5 moments; colourbars occupy their own gutters and cannot overlap panels.
    fig = plt.figure(figsize=(14.2, 7.4)); gs = GridSpec(3, 8, figure=fig, width_ratios=(1, 1, 1, 1, 1, .04, .20, .04), wspace=.10, hspace=.10)
    for row, key in enumerate(("moment0", "moment1", "moment2")):
        images = [moments[f"data_{key}"], moments[f"kinuv_{key}"], moments[f"kinms_{key}"], moments[f"data_minus_kinuv_{key}"], moments[f"data_minus_kinms_{key}"]]
        if key == "moment1": images[:3] = [x - geometry["vsys_kms"] for x in images[:3]]
        common = _limits(images[:3], symmetric=key == "moment1"); residual = _limits(images[3:], symmetric=True)
        for col, image in enumerate(images):
            ax = fig.add_subplot(gs[row, col]); lim = common if col < 3 else residual
            cmap = (plt.get_cmap("coolwarm") if key == "moment1" else _navy_teal()) if col < 3 else _residual_cmap()
            artist = ax.imshow(np.ma.masked_invalid(image), origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.text(.04, .94, f"({chr(97 + row*5 + col)})", transform=ax.transAxes, va="top", bbox={"facecolor":"white","edgecolor":"none","alpha":.82})
            if row == 0: ax.set_title(("Data", "kinUV", "KinMS", "Data - kinUV", "Data - KinMS")[col], fontsize=10)
            if col == 0: ax.set_ylabel(("Moment 0", "Moment 1", "Moment 2")[row])
            if col == 2: common_artist = artist
            if col == 4: residual_artist = artist
        cb = fig.colorbar(common_artist, cax=fig.add_subplot(gs[row, 5])); cb.set_label(("K km s$^{-1}$", "km s$^{-1}$", "km s$^{-1}$")[row], fontsize=8)
        rb = fig.colorbar(residual_artist, cax=fig.add_subplot(gs[row, 7])); rb.set_label("residual", fontsize=8)
    fig.suptitle(f"{contract['target_id']}: frozen-mask 3 x 5 moment diagnostic")
    fig.savefig(output / "moments_3x5.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Major and minor PVDs, with no new threshold/mask beyond the frozen cube mask.
    ra, dec = offset_world(float(header["CRVAL1"]), float(header["CRVAL2"]), geometry["dx_arcsec"], geometry["dy_arcsec"])
    beam = contract["profiles"]["resolution"]["bmaj_arcsec"]; crop = _crop(moments["data_moment0"], header, geometry, beam)
    fig = plt.figure(figsize=(13.5, 6.8)); pgs = GridSpec(3, 5, figure=fig, height_ratios=(1, 1, .055), hspace=.28, wspace=.12)
    axes = np.asarray([[fig.add_subplot(pgs[row, col]) for col in range(5)] for row in range(2)])
    for row, (axis_name, angle) in enumerate((("Major", geometry["pa_deg"]), ("Minor", geometry["pa_deg"] + 90))):
        pvs=[]
        for cube in (data, model, benchmark):
            pv, offsets = pv_diagram(np.where(mask, cube, np.nan), header, ra, dec, angle, 2*crop, float(header["BMIN"])*3600)
            pvs.append(pv)
        images = pvs + [pvs[0]-pvs[1], pvs[0]-pvs[2]]; common=_limits(pvs); residual=_limits(images[3:], symmetric=True)
        velocity=spectral_axis_kms(header); extent=(offsets[0], offsets[-1], velocity[0], velocity[-1])
        for col, image in enumerate(images):
            ax=axes[row,col]; lim=common if col<3 else residual; cmap=_navy_teal() if col<3 else _residual_cmap()
            artist=ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1], rasterized=True)
            if col<3: common_artist=artist
            else: residual_artist=artist
            ax.axhline(geometry["vsys_kms"], color="white", ls=":", lw=.7); ax.text(.04,.94,f"({chr(97+row*5+col)})",transform=ax.transAxes,va="top",bbox={"facecolor":"white","edgecolor":"none","alpha":.82})
            if row==0:
                ax.set_title(("Data","kinUV MAP","KinMS","Data - kinUV MAP","Data - KinMS")[col],fontsize=10)
                if col in (0, 1, 3):
                    def projected_track(values):
                        transform=contract["profiles"]["reporting_transform"]
                        native=transform["native_vsys_radio_topo_kms"] + np.sign(offsets) * np.interp(np.abs(offsets), profiles["radius_arcsec"], values)
                        c=299792.458; correction=transform["frequency_equivalent_correction_kms"]
                        radio_lsrk=native-correction*(1.0-native/c)
                        return radio_lsrk/(1.0-radio_lsrk/c)
                    track=projected_track(profiles["u_projected_map_kms"])
                    lo=profiles["u_projected_q16_kms"]
                    hi=profiles["u_projected_q84_kms"]
                    if np.any(np.isfinite(lo)):
                        lo_track=projected_track(lo); hi_track=projected_track(hi)
                        ax.fill_between(offsets,np.minimum(lo_track,hi_track),np.maximum(lo_track,hi_track),color="#45B8AC",alpha=.28)
                    ax.plot(offsets,track,color="#00E5FF",lw=1.3,label="kinUV MAP projected")
                    if col==0: ax.legend(fontsize=7,loc="upper right")
            if col==0: ax.set_ylabel(f"{axis_name} PVD\noptical LSRK (km s$^{{-1}}$)")
            if row==1: ax.set_xlabel("Offset (arcsec)")
    fig.colorbar(common_artist,cax=fig.add_subplot(pgs[2,:3]),orientation="horizontal",label="Brightness temperature (K)")
    fig.colorbar(residual_artist,cax=fig.add_subplot(pgs[2,3:]),orientation="horizontal",label="Residual brightness temperature (K)")
    fig.suptitle(f"{contract['target_id']}: MAP major and minor PVDs; canonical mask only")
    fig.savefig(output / "pvd_major_minor.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fixed full-mask aperture spectrum.
    velocity=spectral_axis_kms(header); spatial=np.any(mask,axis=0)
    spectra=[np.nansum(np.where(spatial[None],cube,np.nan),axis=(1,2)) for cube in (data,model,benchmark)]
    fig,(ax,res)=plt.subplots(2,1,figsize=(7.1,5),sharex=True,gridspec_kw={"height_ratios":(2,1)})
    for values,label,color in zip(spectra,("Data","kinUV","KinMS"),("black","#2A6F97","#D55E00")): ax.step(velocity,values,where="mid",label=label,color=color)
    res.step(velocity,spectra[0]-spectra[1],where="mid",label="Data - kinUV",color="#2A6F97"); res.step(velocity,spectra[0]-spectra[2],where="mid",label="Data - KinMS",color="#D55E00"); res.axhline(0,color=".7")
    ax.legend(); res.legend(); res.set_xlabel("Optical LSRK velocity (km s$^{-1}$)"); ax.set_ylabel("Frozen-aperture sum"); res.set_ylabel("Residual")
    fig.suptitle(f"{contract['target_id']}: canonical-mask spectrum")
    fig.savefig(output / "spectra.png",dpi=150,bbox_inches="tight"); plt.close(fig)

    # The renderer consumes only common schema fields here.
    r=profiles["radius_arcsec"]; fig,axes=plt.subplots(2,1,figsize=(7.1,5.8),sharex=True)
    for ax in axes: ax.axvspan(0,beam,color=".92")
    for ax,prefix,map_key,label in ((axes[0],"vc_intrinsic","vc_intrinsic_map_kms",r"$V_c$ (km s$^{-1}$)"),(axes[1],"sigma","sigma_map_kms",r"$\sigma$ (km s$^{-1}$)")):
        ax.plot(r,profiles[map_key],color="#2A6F97",label="MAP")
        lo,med,hi=(profiles[f"{prefix}_{q}_kms"] for q in ("q16","q50","q84"))
        if np.any(np.isfinite(lo)): ax.fill_between(r,lo,hi,color="#45B8AC",alpha=.3,label="conditional q16-q84"); ax.plot(r,med,color="#0F7C82",lw=1)
        ax.set_ylabel(label); ax.legend(loc="best")
    turnover=contract["profiles"]["turnover"]
    if turnover["status"]=="resolved": axes[0].axvline(turnover["value_arcsec"],color="#0F7C82",ls=":",label="$R_{50}$")
    axes[1].set_xlabel("Radius (arcsec)"); fig.suptitle(f"{contract['target_id']}: common rotation and dispersion schema")
    fig.savefig(output / "rotation_dispersion.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    return {"render_seconds": time.perf_counter()-started, "files": 4, "products": ["moments_3x5","pvd_major_minor","spectra","rotation_dispersion"]}


def self_check():
    radius=np.linspace(0,10,10001); rt=.7; u=210*(2/np.pi)*np.arctan(radius/rt)
    result=first_rising_half_crossing(radius,u,210,reference_constrained=True)
    assert result["status"]=="resolved" and abs(result["value_arcsec"]-rt)<1e-5
    wave=np.array([0,.6,.4,.7,.3,.8]); ambiguous=first_rising_half_crossing(np.arange(6),wave,1,reference_constrained=True)
    assert ambiguous["status"]=="nonunique"
    assert first_rising_half_crossing(radius,u,210,reference_constrained=False)["status"]=="unresolved"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--production-root",type=Path); parser.add_argument("--output-root",type=Path); parser.add_argument("--targets",nargs="+",default=("KGAS066","KGAS007")); parser.add_argument("--self-check",action="store_true")
    args=parser.parse_args()
    if args.self_check: self_check(); print("self-check: pass")
    if args.production_root and args.output_root:
        summary={}
        for target in args.targets:
            out=args.output_root/target; contract=adapt_target(args.production_root/target,out); render=render_contract(out/"contract.json",out)
            summary[target]={"adapter_seconds":contract["adapter_seconds"],**render,"posterior":contract["inference"]["posterior"]["status"],"r50":contract["profiles"]["turnover"]["status"]}
        (args.output_root/"summary.json").write_text(json.dumps({"schema_version":SCHEMA_VERSION,"targets":summary},indent=2,sort_keys=True)+"\n")
        print(json.dumps(summary,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
