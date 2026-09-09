"""Prototype unified smooth radial representations for the foundation spike.

This module is deliberately outside :mod:`kinuv`.  It reuses the production
forward operator and C1 visibility likelihood, but it is not a production
model or an accepted prior.  Both candidate families expose the same 15
unconstrained coordinates and the same fixed-emissivity target context.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from kinuv.forward.sb import galaxy_r_phi, load_sb_template
from kinuv.infer.map import image_grid_for_vis, predict_binned
from kinuv.infer.s2 import FitCovariance, correlated_chi2, propagate_equal_weight_ar1
from kinuv.infer.s3 import build_positive_emissivity_basis
from kinuv.io.vis import load_target_vis


REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
RECOVERY = PROJECT / "results/validation/crossdomain-recovery-s4-remediation-20260907-r1"
COVARIANCE_RECORD = PROJECT / "results/validation/crossdomain-recovery-s2-20260907-r1/metrics.json"
FAMILIES = ("pspline", "finite_gp")

PARAMETER_NAMES = (
    "log_flux",
    "pa_rad",
    "vsys_channel_offset",
    "dx_over_bmaj",
    "dy_over_bmaj",
    "cos_inclination_logit",
    "log_u_reference_kms",
    "log_rotation_scale_over_bmaj",
    "rotation_shape_1",
    "rotation_shape_2",
    "rotation_shape_3",
    "rotation_shape_4",
    "log_sigma0_kms",
    "dispersion_deviation_1",
    "dispersion_deviation_2",
)


@dataclass(frozen=True)
class RadialDesign:
    """Fixed radial scales and basis transforms for one target."""

    bmaj_arcsec: float
    cell_arcsec: float
    r20_arcsec: float
    r50_emission_arcsec: float
    r70_arcsec: float
    r95_arcsec: float
    reference_radius_arcsec: float
    outer_radius_arcsec: float
    pspline_knots_arcsec: np.ndarray
    pspline_transform: np.ndarray
    gp_centres_arcsec: np.ndarray
    gp_length_arcsec: float
    gp_whitener: np.ndarray
    dispersion_knots_arcsec: np.ndarray
    dispersion_transform: np.ndarray
    rotation_shape_scale: float = 0.35
    dispersion_shape_scale: float = 0.25


@dataclass(frozen=True)
class TargetContext:
    """One target, one representation, and the unchanged production operator."""

    target_id: str
    family: str
    data: Any
    grid: Any
    template: np.ndarray
    covariance: FitCovariance
    design: RadialDesign
    config: dict
    parent_parameters: dict
    initial_z: np.ndarray
    provenance: dict


def _weighted_quantiles(radius, weight, quantiles):
    r = np.asarray(radius, dtype=np.float64).ravel()
    w = np.maximum(np.asarray(weight, dtype=np.float64).ravel(), 0.0)
    good = np.isfinite(r) & np.isfinite(w) & (w > 0.0)
    order = np.argsort(r[good])
    rs = r[good][order]
    ws = w[good][order]
    cdf = np.cumsum(ws) / np.sum(ws)
    return np.interp(np.asarray(quantiles, dtype=float), cdf, rs)


def _bspline_derivative_row(knots: np.ndarray, degree: int, x: float) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    row = np.empty(n_basis, dtype=np.float64)
    for index in range(n_basis):
        coefficient = np.zeros(n_basis)
        coefficient[index] = 1.0
        row[index] = BSpline(knots, coefficient, degree).derivative()(x)
    return row


def _proper_pspline_transform(knots: np.ndarray, degree: int) -> np.ndarray:
    """Return four proper, endpoint-flat cubic P-spline coefficient modes."""
    n_basis = len(knots) - degree - 1
    if n_basis != 6:
        raise ValueError("rotation P-spline prototype requires six raw coefficients")
    endpoint_derivative = _bspline_derivative_row(knots, degree, float(knots[-1]))
    constraints = np.vstack((np.ones(n_basis), endpoint_derivative))
    modes = null_space(constraints)
    second_difference = np.diff(np.eye(n_basis), n=2, axis=0)
    # D2 alone has a linear nullspace.  The small coefficient ridge makes the
    # prior proper instead of silently leaving a slope mode unregularized.
    precision = second_difference.T @ second_difference + 0.05 * np.eye(n_basis)
    reduced = modes.T @ precision @ modes
    chol = np.linalg.cholesky(reduced)
    return modes @ np.linalg.solve(chol.T, np.eye(modes.shape[1]))


def _dispersion_transform(knots: np.ndarray, degree: int) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    endpoint_derivative = _bspline_derivative_row(knots, degree, float(knots[-1]))
    constraints = np.vstack((np.ones(n_basis), endpoint_derivative))
    modes = null_space(constraints)
    # Normalize the two modes without changing their constrained subspace.
    return modes @ np.diag(1.0 / np.maximum(np.linalg.norm(modes, axis=0), 1.0e-12))


def _radial_design(template, grid, pa_deg, inclination_deg, bmaj_arcsec) -> RadialDesign:
    radius, _ = galaxy_r_phi(
        grid, np.radians(float(pa_deg)), np.radians(float(inclination_deg))
    )
    r20, r50, r70, r95 = _weighted_quantiles(
        radius, np.maximum(template, 0.0), (0.20, 0.50, 0.70, 0.95)
    )
    bmaj = float(bmaj_arcsec)
    cell = float(grid.cell_arcsec)
    outer = max(float(r95) + 0.5 * bmaj, 2.0 * bmaj)
    inner_node = max(2.0 * cell, min(0.25 * bmaj, 0.5 * float(r20)))
    middle_node = max(0.75 * bmaj, float(r50))
    middle_node = min(middle_node, outer - max(2.0 * cell, 0.2 * bmaj))
    if not 0.0 < inner_node < middle_node < outer:
        raise ValueError("automatic radial nodes are not strictly ordered")

    pspline_knots = np.array(
        [0.0] * 4 + [inner_node, middle_node] + [outer] * 4,
        dtype=np.float64,
    )
    pspline_transform = _proper_pspline_transform(pspline_knots, 3)

    # Four inducing locations and a fixed beam/data-derived kernel.  The inner
    # point is below one beam by construction; the beam is a scale, not a cutoff.
    gp_centres = np.array(
        [0.0, inner_node, max(0.75 * bmaj, float(r20)), float(r70)],
        dtype=np.float64,
    )
    gp_centres = np.maximum.accumulate(gp_centres)
    for index in range(1, gp_centres.size):
        gp_centres[index] = max(gp_centres[index], gp_centres[index - 1] + cell)
    gp_length = max(2.0 * cell, 0.45 * bmaj)
    delta = gp_centres[:, None] - gp_centres[None, :]
    kernel = np.exp(-0.5 * (delta / gp_length) ** 2)
    gp_chol = np.linalg.cholesky(kernel + 1.0e-6 * np.eye(kernel.shape[0]))
    gp_whitener = np.linalg.solve(gp_chol.T, np.eye(kernel.shape[0]))

    dispersion_knots = np.array([0.0] * 4 + [outer] * 4, dtype=np.float64)
    return RadialDesign(
        bmaj_arcsec=bmaj,
        cell_arcsec=cell,
        r20_arcsec=float(r20),
        r50_emission_arcsec=float(r50),
        r70_arcsec=float(r70),
        r95_arcsec=float(r95),
        reference_radius_arcsec=float(r70),
        outer_radius_arcsec=float(outer),
        pspline_knots_arcsec=pspline_knots,
        pspline_transform=pspline_transform,
        gp_centres_arcsec=gp_centres,
        gp_length_arcsec=float(gp_length),
        gp_whitener=gp_whitener,
        dispersion_knots_arcsec=dispersion_knots,
        dispersion_transform=_dispersion_transform(dispersion_knots, 3),
    )


def _bspline_basis(x, knots, degree: int, xp):
    """Cox-de Boor basis with a constant C1 extension at the outer endpoint."""
    values = xp.asarray(x)
    knot = xp.asarray(knots)
    outer = float(np.asarray(knots)[-1])
    bounded = xp.clip(values, 0.0, outer)
    n_basis = len(knots) - degree - 1
    basis = []
    for index in range(len(knots) - 1):
        inside = (bounded >= knot[index]) & (bounded < knot[index + 1])
        basis.append(inside.astype(values.dtype))
    basis = xp.stack(basis, axis=-1)
    for order in range(1, degree + 1):
        next_basis = []
        for index in range(len(knots) - order - 1):
            left_den = float(knots[index + order] - knots[index])
            right_den = float(knots[index + order + 1] - knots[index + 1])
            left = xp.zeros_like(bounded) if left_den == 0.0 else (bounded - knot[index]) * basis[..., index] / left_den
            right = xp.zeros_like(bounded) if right_den == 0.0 else (knot[index + order + 1] - bounded) * basis[..., index + 1] / right_den
            next_basis.append(left + right)
        basis = xp.stack(next_basis, axis=-1)
    endpoint = bounded == outer
    if hasattr(basis, "at"):
        basis = xp.where(endpoint[..., None], xp.zeros_like(basis), basis)
        basis = basis.at[..., n_basis - 1].set(
            xp.where(endpoint, 1.0, basis[..., n_basis - 1])
        )
    else:
        basis = np.where(np.asarray(endpoint)[..., None], np.zeros_like(basis), basis)
        basis[..., n_basis - 1] = np.where(
            np.asarray(endpoint), 1.0, basis[..., n_basis - 1]
        )
    return basis


def rotation_shape_features(radius, design: RadialDesign, family: str):
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(radius)
    r = xp.asarray(radius)
    reference = xp.asarray(design.reference_radius_arcsec)
    if family == "pspline":
        transform = xp.asarray(design.pspline_transform)
        values = _bspline_basis(r, design.pspline_knots_arcsec, 3, xp) @ transform
        at_reference = _bspline_basis(reference, design.pspline_knots_arcsec, 3, xp) @ transform
        return design.rotation_shape_scale * (values - at_reference)
    if family == "finite_gp":
        centres = xp.asarray(design.gp_centres_arcsec)
        whitener = xp.asarray(design.gp_whitener)
        length = float(design.gp_length_arcsec)
        kernel = xp.exp(-0.5 * ((r[..., None] - centres) / length) ** 2)
        kernel_ref = xp.exp(-0.5 * ((reference - centres) / length) ** 2)
        return design.rotation_shape_scale * ((kernel - kernel_ref) @ whitener)
    raise ValueError(f"unknown family {family!r}")


def dispersion_shape_features(radius, design: RadialDesign):
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(radius)
    transform = xp.asarray(design.dispersion_transform)
    return design.dispersion_shape_scale * (
        _bspline_basis(xp.asarray(radius), design.dispersion_knots_arcsec, 3, xp)
        @ transform
    )


def z_to_physical(z, context: TargetContext):
    """Decode the common unconstrained chart into physical coordinates."""
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(z)
    q = xp.asarray(z)
    cos_lo, cos_hi = 1.0e-3, 0.999
    logistic = 1.0 / (1.0 + xp.exp(-q[5]))
    cos_i = cos_lo + (cos_hi - cos_lo) * logistic
    return {
        "flux": xp.exp(q[0]),
        "pa_rad": q[1],
        "vsys_kms": float(context.config["stage_a"]["parameter_seed"]["vsys_kms"])
        + q[2] * float(context.data.dv_kms),
        "dx_arcsec": q[3] * context.design.bmaj_arcsec,
        "dy_arcsec": q[4] * context.design.bmaj_arcsec,
        "cos_inclination": cos_i,
        "i_rad": xp.arccos(cos_i),
        "u_reference_kms": xp.exp(q[6]),
        "rotation_scale_arcsec": context.design.bmaj_arcsec * xp.exp(q[7]),
        "rotation_shape": q[8:12],
        "sigma0_kms": xp.exp(q[12]),
        "dispersion_deviation": q[13:15],
    }


def physical_to_z(parameters: dict, context: TargetContext) -> np.ndarray:
    cos_lo, cos_hi = 1.0e-3, 0.999
    cos_i = float(parameters["cos_inclination"])
    probability = np.clip((cos_i - cos_lo) / (cos_hi - cos_lo), 1.0e-10, 1.0 - 1.0e-10)
    return np.array(
        [
            np.log(float(parameters["flux"])),
            float(parameters["pa_rad"]),
            (float(parameters["vsys_kms"]) - float(context.config["stage_a"]["parameter_seed"]["vsys_kms"])) / float(context.data.dv_kms),
            float(parameters["dx_arcsec"]) / context.design.bmaj_arcsec,
            float(parameters["dy_arcsec"]) / context.design.bmaj_arcsec,
            np.log(probability) - np.log1p(-probability),
            np.log(float(parameters["u_reference_kms"])),
            np.log(float(parameters["rotation_scale_arcsec"]) / context.design.bmaj_arcsec),
            *np.asarray(parameters["rotation_shape"], dtype=float),
            np.log(float(parameters["sigma0_kms"])),
            *np.asarray(parameters["dispersion_deviation"], dtype=float),
        ],
        dtype=np.float64,
    )


def log_abs_det_jacobian(z, context: TargetContext):
    """Jacobian from common chart coordinates to the physical parameter vector."""
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(z)
    q = xp.asarray(z)
    logistic = 1.0 / (1.0 + xp.exp(-q[5]))
    positive_terms = q[0] + q[6] + q[7] + q[12]
    # dx, dy, and rotation scale each contribute one BMAJ factor; vsys
    # contributes one channel-width factor.
    scale_terms = 3.0 * np.log(context.design.bmaj_arcsec) + np.log(context.data.dv_kms)
    inclination_term = np.log(0.998) + xp.log(logistic) + xp.log1p(-logistic)
    return positive_terms + scale_terms + inclination_term


def profiles_from_z(radius_arcsec, z, context: TargetContext):
    """Return projected speed, intrinsic speed, and positive dispersion."""
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(radius_arcsec, z)
    radius = xp.asarray(radius_arcsec)
    p = z_to_physical(z, context)
    reference = float(context.design.reference_radius_arcsec)
    scale = p["rotation_scale_arcsec"]
    carrier = xp.arctan(radius / scale) / xp.arctan(reference / scale)
    features = rotation_shape_features(radius, context.design, context.family)
    log_deviation = features @ p["rotation_shape"]
    # The analytic carrier is exactly zero at the origin.  Do not mask it with
    # where(): that would replace the finite autodiff central slope by zero.
    u_projected = p["u_reference_kms"] * carrier * xp.exp(log_deviation)
    sin_i = xp.maximum(xp.sin(p["i_rad"]), 1.0e-6)
    vc_intrinsic = u_projected / sin_i
    sigma_features = dispersion_shape_features(radius, context.design)
    sigma = p["sigma0_kms"] * xp.exp(sigma_features @ p["dispersion_deviation"])
    return u_projected, vc_intrinsic, sigma


def _model_and_chi2(z, context: TargetContext):
    import jax.numpy as jnp

    p = z_to_physical(z, context)

    def velocity_profile(radius):
        return profiles_from_z(radius, z, context)[1]

    def dispersion_profile(radius):
        return profiles_from_z(radius, z, context)[2]

    params = {
        "flux": p["flux"],
        "pa_deg": p["pa_rad"] * (180.0 / np.pi),
        "vsys_kms": p["vsys_kms"],
        "gas_sigma_kms": p["sigma0_kms"],
        "dx_arcsec": p["dx_arcsec"],
        "dy_arcsec": p["dy_arcsec"],
        # Required by the shared call signature; the custom profile owns speed.
        "v0_kms": p["u_reference_kms"] / jnp.maximum(jnp.sin(p["i_rad"]), 1.0e-6),
        "r_t_arcsec": p["rotation_scale_arcsec"],
    }
    model = predict_binned(
        context.data,
        params,
        context.template,
        context.grid,
        i_rad=p["i_rad"],
        xla=True,
        velocity_profile=velocity_profile,
        dispersion_profile=dispersion_profile,
    )
    chi2 = correlated_chi2(
        jnp.asarray(context.data.vis), model, jnp.asarray(context.data.weights), context.covariance
    )
    return model, chi2


def log_likelihood(z, context: TargetContext):
    return -0.5 * _model_and_chi2(z, context)[1]


def log_prior_chart(z, context: TargetContext):
    """Fixed conditional spike prior in chart coordinates.

    The four rotation latents are standard normal for both families.  The two
    dispersion deviations are also standard normal and may be exactly zero,
    in which case the dispersion is constant at every radius.
    """
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(z)
    q = xp.asarray(z)
    parent = context.parent_parameters
    flux_seed = float(parent["flux"])
    sigma_seed = np.sqrt(
        float(parent["sigma_inner_kms"]) * float(parent["sigma_outer_kms"])
    )
    u_seed = _parent_projected_speed(
        context.design.reference_radius_arcsec, parent, context
    )
    scale_seed = _parent_scale(parent, context.design.bmaj_arcsec)
    dx = q[3] * context.design.bmaj_arcsec
    dy = q[4] * context.design.bmaj_arcsec
    terms = (
        ((q[0] - np.log(flux_seed)) / 0.7) ** 2
        + ((q[1] - np.radians(float(parent["pa_deg"]))) / np.pi) ** 2
        + (q[2] * context.data.dv_kms / 50.0) ** 2
        + (dx / 0.5) ** 2
        + (dy / 0.5) ** 2
        + ((q[6] - np.log(max(u_seed, 1.0))) / 0.7) ** 2
        + ((q[7] - np.log(scale_seed / context.design.bmaj_arcsec)) / 0.8) ** 2
        + xp.sum(q[8:12] ** 2)
        + ((q[12] - np.log(sigma_seed)) / 0.5) ** 2
        + xp.sum(q[13:15] ** 2)
    )
    # The unwrapped PA has a proper, broad local normal prior around the parent
    # mode.  This spike therefore conditions on one PA lift; production should
    # use a normalized periodic prior or an explicitly bounded periodic chart.
    # Cos(i) is uniform in physical space, so its logistic chart Jacobian is
    # part of the chart density.
    logistic = 1.0 / (1.0 + xp.exp(-q[5]))
    cos_jacobian = xp.log(logistic) + xp.log1p(-logistic)
    return -0.5 * terms + cos_jacobian


def log_posterior(z, context: TargetContext):
    return log_likelihood(z, context) + log_prior_chart(z, context)


def value_and_grad(context: TargetContext) -> Callable:
    import jax

    return jax.jit(jax.value_and_grad(lambda z: -log_posterior(z, context)))


def likelihood_chi2(z, context: TargetContext):
    return _model_and_chi2(z, context)[1]


def _parent_scale(parent, bmaj):
    ratio = float(parent.get("turnover_over_bmaj", 0.5))
    return max(0.03 * bmaj, ratio * bmaj)


def _parent_projected_speed(radius, parent, context):
    knots = np.asarray(parent.get("u_knots_kms", []), dtype=float)
    knot_radii = np.asarray(context.provenance["parent_knot_radii_arcsec"], dtype=float)
    preferred = context.provenance["parent_candidate"]
    if preferred == "supported_rings" and knots.size == knot_radii.size:
        if radius < knot_radii[0]:
            return float(knots[0] * radius / knot_radii[0])
        return float(np.interp(radius, knot_radii, knots, right=knots[-1]))
    u = float(parent["arctan_u_kms"])
    scale = _parent_scale(parent, context.design.bmaj_arcsec)
    return float(u * (2.0 / np.pi) * np.arctan(radius / scale))


def _initial_chart(context: TargetContext) -> np.ndarray:
    p = context.parent_parameters
    design = context.design
    scale = _parent_scale(p, design.bmaj_arcsec)
    reference = design.reference_radius_arcsec
    u_reference = max(_parent_projected_speed(reference, p, context), 1.0)
    sample_radius = np.unique(
        np.concatenate(
            ([0.05 * design.bmaj_arcsec], design.gp_centres_arcsec, np.linspace(0.1, 1.0, 12) * design.outer_radius_arcsec)
        )
    )
    parent_u = np.array([max(_parent_projected_speed(r, p, context), 1.0e-6) for r in sample_radius])
    carrier = u_reference * np.arctan(sample_radius / scale) / np.arctan(reference / scale)
    target_log_deviation = np.log(parent_u / np.maximum(carrier, 1.0e-8))
    features = np.asarray(rotation_shape_features(sample_radius, design, context.family))
    shape = np.linalg.lstsq(features, target_log_deviation, rcond=1.0e-8)[0]
    shape = np.clip(shape, -2.5, 2.5)

    sigma_inner = float(p["sigma_inner_kms"])
    sigma_outer = float(p["sigma_outer_kms"])
    sigma0 = np.sqrt(sigma_inner * sigma_outer)
    sigma_target = np.log(
        sigma_inner
        + (sigma_outer - sigma_inner)
        * 0.5
        * (1.0 + np.tanh((sample_radius - design.r50_emission_arcsec) / (0.25 * design.bmaj_arcsec)))
    ) - np.log(sigma0)
    sigma_features = np.asarray(dispersion_shape_features(sample_radius, design))
    sigma_deviation = np.linalg.lstsq(sigma_features, sigma_target, rcond=1.0e-8)[0]
    sigma_deviation = np.clip(sigma_deviation, -2.5, 2.5)
    physical = {
        "flux": float(p["flux"]),
        "pa_rad": np.radians(float(p["pa_deg"])),
        "vsys_kms": float(p["vsys_kms"]),
        "dx_arcsec": float(p["dx_arcsec"]),
        "dy_arcsec": float(p["dy_arcsec"]),
        "cos_inclination": np.cos(np.radians(float(p["inclination_deg"]))),
        "u_reference_kms": u_reference,
        "rotation_scale_arcsec": scale,
        "rotation_shape": shape,
        "sigma0_kms": sigma0,
        "dispersion_deviation": sigma_deviation,
    }
    return physical_to_z(physical, context)


def _covariance_for(target_id: str, n_bin: int):
    metrics = json.loads(COVARIANCE_RECORD.read_text(encoding="utf-8"))
    row = next(item for item in metrics["targets"] if item["target_id"] == target_id)
    source = next(iter(row["covariance"]["parameters"]["C1"].values()))
    return propagate_equal_weight_ar1(source["scale"], source["rho"], n_bin), source


def build_target_context(target_id: str, family: str) -> TargetContext:
    """Load one real target with its frozen C1 data and accepted emissivity."""
    if target_id not in ("KGAS066", "KGAS007"):
        raise ValueError("target_id must be KGAS066 or KGAS007")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    config_path = REPO / "configs/targets" / f"{target_id}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_path = RECOVERY / "s3" / target_id / "ablations.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    candidate = checkpoint["selection"]["preferred_candidate"]
    selected = next(row for row in checkpoint["fits"] if row["candidate"] == candidate)
    parent = selected["parameters"]
    phase = config.get("phase_center_deg")
    phase_rad = None if phase is None else np.radians(np.asarray(phase, dtype=float))
    data, load_metadata = load_target_vis(
        config["visibility_npz"], cube_path=config["fit_window_cube"], phase_dir_rad=phase_rad
    )
    grid = image_grid_for_vis(data)
    base_template = load_sb_template(grid, Path(config["template_ico"]))
    s2 = json.loads(Path(checkpoint["s2_summary_path"]).read_text(encoding="utf-8"))
    s2_parameters = next(row for row in s2["targets"] if row["target_id"] == target_id)["best_joint"]["parameters"]
    emissivity = build_positive_emissivity_basis(
        base_template,
        grid,
        np.radians(float(s2_parameters["pa_deg"])),
        np.radians(float(s2_parameters["inclination_deg"])),
    )
    fixed_weights = np.asarray(parent["emissivity_weights"], dtype=np.float64)
    template = np.sum(fixed_weights[:, None, None] * emissivity.images, axis=0)
    design = _radial_design(
        template,
        grid,
        parent["pa_deg"],
        parent["inclination_deg"],
        config["diagnostic_beam"]["bmaj_arcsec"],
    )
    covariance, source_covariance = _covariance_for(target_id, data.n_bin)
    provenance = {
        "schema_version": "kinuv-unified-representation-context-v1",
        "prototype_only": True,
        "visibility_likelihood": "complex visibility C1 AR(1), real and imaginary components",
        "image_likelihood": False,
        "mass_model": False,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "covariance_path": str(COVARIANCE_RECORD),
        "load_metadata": load_metadata,
        "parent_candidate": candidate,
        "parent_chi2": float(selected["chi2"]),
        "parent_knot_radii_arcsec": checkpoint["velocity_support"]["knot_radii_arcsec"],
        "fixed_emissivity_weights": fixed_weights.tolist(),
        "source_c1": source_covariance,
    }
    placeholder = TargetContext(
        target_id=target_id,
        family=family,
        data=data,
        grid=grid,
        template=template,
        covariance=covariance,
        design=design,
        config=config,
        parent_parameters=parent,
        initial_z=np.zeros(len(PARAMETER_NAMES)),
        provenance=provenance,
    )
    initial = _initial_chart(placeholder)
    return TargetContext(**{**placeholder.__dict__, "initial_z": initial})


def profile_artifact(z, context: TargetContext, n_radius: int = 256) -> dict:
    """Create a target/family-neutral deterministic profile record."""
    radius = np.linspace(0.0, context.design.outer_radius_arcsec, int(n_radius))
    u, vc, sigma = (np.asarray(x, dtype=float) for x in profiles_from_z(radius, z, context))
    du = np.gradient(u, radius, edge_order=2)
    dvc = np.gradient(vc, radius, edge_order=2)
    dsigma = np.gradient(sigma, radius, edge_order=2)
    inclination = float(np.degrees(z_to_physical(np.asarray(z), context)["i_rad"]))
    return {
        "schema_version": "kinuv-unified-radial-profile-v1",
        "status": "PROTOTYPE_CONDITIONAL_MAP_PROFILE",
        "source": context.provenance,
        "target_id": context.target_id,
        "model_family": context.family,
        "geometry": {"inclination_deg": inclination, "inclination_conditional": True},
        "units": {
            "radius_arcsec": "arcsec",
            "u_projected_kms": "km/s",
            "vc_intrinsic_kms": "km/s",
            "sigma_kms": "km/s",
            "du_dR": "km/s/arcsec",
            "dvc_dR": "km/s/arcsec",
            "dsigma_dR": "km/s/arcsec",
        },
        "derivative_method": "numpy.gradient second-order finite difference on uniform reporting grid",
        "radius_arcsec": radius.tolist(),
        "u_projected_kms": u.tolist(),
        "vc_intrinsic_kms": vc.tolist(),
        "sigma_kms": sigma.tolist(),
        "du_dR": du.tolist(),
        "dvc_dR": dvc.tolist(),
        "dsigma_dR": dsigma.tolist(),
        "support": {
            "beam_bmaj_arcsec": context.design.bmaj_arcsec,
            "emission_r95_arcsec": context.design.r95_arcsec,
            "reference_radius_arcsec": context.design.reference_radius_arcsec,
            "reported_outer_radius_arcsec": context.design.outer_radius_arcsec,
            "subbeam_is_diagnostic_not_excluded": True,
        },
        "flags": {
            "posterior_unavailable": True,
            "r50_status": "UNASSESSED_PLATEAU_IDENTIFIABILITY",
            "derivative_supported": True,
            "kappa2_negative_diagnostic": bool(np.any(np.gradient(du, radius, edge_order=2) < 0.0)),
        },
    }
