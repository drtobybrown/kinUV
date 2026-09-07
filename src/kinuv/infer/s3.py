"""Joint, target-neutral S3 visibility-model ablations.

The chart keeps geometry and nuisance parameters joint while adding one model
component at a time: a positive low-rank emissivity basis, four
emission-supported projected-velocity knots, and a smooth two-zone dispersion.
All radii remain angular and no mass model enters the likelihood.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import minimize

from kinuv.forward.sb import galaxy_r_phi
from kinuv.infer.map import predict_binned
from kinuv.infer.s2 import FitCovariance, correlated_chi2, projected_gradient


S3_PARAMETER_NAMES = (
    "log_flux",
    "pa_rad",
    "vsys_channel_offset",
    "log_sigma_inner",
    "dx_over_bmaj",
    "dy_over_bmaj",
    "cos_inclination",
    "turnover_over_bmaj",
    "arctan_u_over_100",
    "u_knot_0_over_100",
    "u_knot_1_over_100",
    "u_knot_2_over_100",
    "u_knot_3_over_100",
    "emissivity_logit_1",
    "emissivity_logit_2",
    "log_sigma_outer",
)

CANDIDATES = (
    "baseline_arctan",
    "joint_emissivity",
    "supported_rings",
    "two_zone_dispersion",
)


@dataclass(frozen=True)
class EmissivityBasis:
    images: np.ndarray
    natural_weights: np.ndarray
    radial_edges_arcsec: np.ndarray
    r95_arcsec: float
    transition_width_arcsec: float


@dataclass(frozen=True)
class S3FitResult:
    candidate: str
    parameters: dict
    objective: float
    chi2: float
    prior: float
    regularization: float
    projected_gradient_inf: float
    projected_gradient_inf_raw: float
    projected_gradient_normalization: str
    nfev: int
    njev: int
    nit: int
    success: bool
    message: str
    boundary_parameters: tuple[str, ...]
    hessian: dict

    def to_dict(self) -> dict:
        row = asdict(self)
        row["boundary_parameters"] = list(self.boundary_parameters)
        return row


def _weighted_radius_quantiles(radius, weight, quantiles):
    r = np.asarray(radius, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.float64).ravel()
    good = np.isfinite(r) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        raise ValueError("positive emissivity support is empty")
    order = np.argsort(r[good])
    rs = r[good][order]
    ws = w[good][order]
    cdf = np.cumsum(ws)
    cdf /= cdf[-1]
    return np.interp(np.asarray(quantiles, dtype=np.float64), cdf, rs)


def build_positive_emissivity_basis(template, grid, pa_rad, i_rad) -> EmissivityBasis:
    """Build a positive, smooth radial partition of the template.

    The former hard annular cuts let two emissivity logits create artificial
    brightness steps at the one-third and two-third flux radii.  A quintic
    smoothstep partition retains the same compact three-component chart while
    making the represented sky brightness twice continuously differentiable
    across both transitions.  The natural mixture reconstructs the clipped,
    normalized input exactly.
    """

    image = np.maximum(np.asarray(template, dtype=np.float64), 0.0)
    area = float(grid.cell_arcsec) ** 2
    total = float(np.sum(image) * area)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("template has no positive finite flux")
    image /= total
    radius, _ = galaxy_r_phi(grid, float(pa_rad), float(i_rad))
    q33, q67, r95 = _weighted_radius_quantiles(
        radius, image, (1.0 / 3.0, 2.0 / 3.0, 0.95)
    )
    separation = min(float(q33), float(q67 - q33), float(r95 - q67))
    width = max(float(grid.cell_arcsec), 0.25 * separation)
    width = min(width, 0.45 * float(q67 - q33))

    def transition(center):
        t = np.clip((radius - (center - width)) / (2.0 * width), 0.0, 1.0)
        return t**3 * (10.0 + t * (-15.0 + 6.0 * t))

    inner_to_middle = transition(q33)
    middle_to_outer = transition(q67)
    windows = (
        1.0 - inner_to_middle,
        inner_to_middle - middle_to_outer,
        middle_to_outer,
    )
    components = []
    weights = []
    for window in windows:
        component = image * np.maximum(window, 0.0)
        fraction = float(np.sum(component) * area)
        if fraction <= 0.0:
            raise ValueError("emissivity radial basis contains an empty component")
        components.append(component / fraction)
        weights.append(fraction)
    natural = np.asarray(weights, dtype=np.float64)
    natural /= natural.sum()
    return EmissivityBasis(
        images=np.stack(components),
        natural_weights=natural,
        radial_edges_arcsec=np.array([0.0, q33, q67], dtype=np.float64),
        r95_arcsec=float(r95),
        transition_width_arcsec=float(width),
    )


def supported_knot_radii(template, grid, pa_rad, i_rad, bmaj_arcsec):
    """Four beam-aware radii from positive-emission cumulative support."""

    image = np.maximum(np.asarray(template, dtype=np.float64), 0.0)
    radius, _ = galaxy_r_phi(grid, float(pa_rad), float(i_rad))
    outer = _weighted_radius_quantiles(radius, image, (0.40, 0.70, 0.95))
    floor = 0.5 * float(bmaj_arcsec)
    separation = 0.20 * float(bmaj_arcsec)
    knots = np.concatenate(([floor], np.asarray(outer, dtype=np.float64)))
    for index in range(1, knots.size):
        knots[index] = max(knots[index], knots[index - 1] + separation)
    if knots[-1] > outer[-1] + 0.5 * float(bmaj_arcsec):
        raise ValueError("emission support cannot identify four beam-aware knots")
    return knots, float(outer[-1])


def _softmax_weights(logit_1, logit_2, xp):
    logits = xp.stack((xp.asarray(0.0), xp.asarray(logit_1), xp.asarray(logit_2)))
    logits = logits - xp.max(logits)
    values = xp.exp(logits)
    return values / xp.sum(values)


def initial_chart(s2_parameters, knot_radii_arcsec, natural_weights, *, vsys_seed_kms, dv_kms, bmaj_arcsec):
    p = s2_parameters
    u0 = float(p["u_kms"])
    rt = float(p["r_t_arcsec"])
    knot_u = u0 * (2.0 / np.pi) * np.arctan(np.asarray(knot_radii_arcsec) / rt)
    w = np.asarray(natural_weights, dtype=np.float64)
    return np.array(
        [
            np.log(float(p["flux"])),
            np.radians(float(p["pa_deg"])),
            (float(p["vsys_kms"]) - float(vsys_seed_kms)) / float(dv_kms),
            np.log(float(p["gas_sigma_kms"])),
            float(p["dx_arcsec"]) / float(bmaj_arcsec),
            float(p["dy_arcsec"]) / float(bmaj_arcsec),
            np.cos(np.radians(float(p["inclination_deg"]))),
            float(p["r_t_arcsec"]) / float(bmaj_arcsec),
            u0 / 100.0,
            *(knot_u / 100.0),
            np.log(w[1] / w[0]),
            np.log(w[2] / w[0]),
            np.log(float(p["gas_sigma_kms"])),
        ],
        dtype=np.float64,
    )


def chart_from_parameters(
    parameters,
    *,
    vsys_seed_kms,
    dv_kms,
    bmaj_arcsec,
):
    """Encode a complete saved S3 parameter record in the 16-slot chart."""
    p = parameters
    weights = np.asarray(p["emissivity_weights"], dtype=np.float64)
    if weights.shape != (3,) or np.any(weights <= 0.0):
        raise ValueError("emissivity_weights must contain three positive values")
    weights = weights / weights.sum()
    return np.array(
        [
            np.log(float(p["flux"])),
            np.radians(float(p["pa_deg"])),
            (float(p["vsys_kms"]) - float(vsys_seed_kms)) / float(dv_kms),
            np.log(float(p["sigma_inner_kms"])),
            float(p["dx_arcsec"]) / float(bmaj_arcsec),
            float(p["dy_arcsec"]) / float(bmaj_arcsec),
            np.cos(np.radians(float(p["inclination_deg"]))),
            float(p["turnover_over_bmaj"]),
            float(p["arctan_u_kms"]) / 100.0,
            *(np.asarray(p["u_knots_kms"], dtype=np.float64) / 100.0),
            np.log(weights[1] / weights[0]),
            np.log(weights[2] / weights[0]),
            np.log(float(p["sigma_outer_kms"])),
        ],
        dtype=np.float64,
    )


def chart_bounds(
    pa_seed_deg,
    dv_kms,
    bmaj_arcsec,
    candidate,
    *,
    two_zone_uses_rings=True,
):
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown S3 candidate {candidate!r}")
    pa = math.radians(float(pa_seed_deg))
    shift = 2.0 / float(bmaj_arcsec)
    bounds = [
        (math.log(1.0e-8), math.log(100.0)),
        (pa - math.pi, pa + math.pi),
        (-100.0 / float(dv_kms), 100.0 / float(dv_kms)),
        (math.log(2.0), math.log(50.0)),
        (-shift, shift),
        (-shift, shift),
        (1.0e-3, 0.999),
        (0.05, 1.6),
        (0.0, 5.0),
        *((0.0, 5.0),) * 4,
        (-5.0, 5.0),
        (-5.0, 5.0),
        (math.log(2.0), math.log(50.0)),
    ]
    active = set(range(9))
    if candidate in {"joint_emissivity", "supported_rings", "two_zone_dispersion"}:
        active.update((13, 14))
    use_rings = candidate == "supported_rings" or (
        candidate == "two_zone_dispersion" and two_zone_uses_rings
    )
    if use_rings:
        active.difference_update((7, 8))
        active.update((9, 10, 11, 12))
    if candidate == "two_zone_dispersion":
        active.add(15)
    return bounds, tuple(sorted(active))


def _decode(z, *, vsys_seed_kms, dv_kms, bmaj_arcsec):
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(z)
    values = xp.asarray(z)
    cos_i = values[6]
    i_rad = xp.arccos(cos_i)
    return {
        "flux": xp.exp(values[0]),
        "pa_deg": values[1] * (180.0 / np.pi),
        "vsys_kms": float(vsys_seed_kms) + values[2] * float(dv_kms),
        "gas_sigma_kms": xp.exp(values[3]),
        "dx_arcsec": values[4] * float(bmaj_arcsec),
        "dy_arcsec": values[5] * float(bmaj_arcsec),
        "i_rad": i_rad,
        "r_t_arcsec": values[7] * float(bmaj_arcsec),
        "arctan_u_kms": 100.0 * values[8],
        "u_knots_kms": 100.0 * values[9:13],
        "emissivity_logits": values[13:15],
        "sigma_outer_kms": xp.exp(values[15]),
    }


def unpack_chart(z, *, vsys_seed_kms, dv_kms, bmaj_arcsec, natural_weights):
    p = _decode(z, vsys_seed_kms=vsys_seed_kms, dv_kms=dv_kms, bmaj_arcsec=bmaj_arcsec)
    weights = _softmax_weights(p["emissivity_logits"][0], p["emissivity_logits"][1], np)
    return {
        "flux": float(p["flux"]),
        "pa_deg": float(p["pa_deg"] % 360.0),
        "vsys_kms": float(p["vsys_kms"]),
        "sigma_inner_kms": float(p["gas_sigma_kms"]),
        "sigma_outer_kms": float(p["sigma_outer_kms"]),
        "dx_arcsec": float(p["dx_arcsec"]),
        "dy_arcsec": float(p["dy_arcsec"]),
        "inclination_deg": float(np.degrees(p["i_rad"])),
        "turnover_over_bmaj": float(np.asarray(z)[7]),
        "arctan_u_kms": float(p["arctan_u_kms"]),
        "u_knots_kms": np.asarray(p["u_knots_kms"], dtype=float).tolist(),
        "emissivity_weights": np.asarray(weights, dtype=float).tolist(),
        "natural_emissivity_weights": np.asarray(natural_weights, dtype=float).tolist(),
    }


def _projected_ring_profile(radius, knot_radii, u_knots, i_rad, xp):
    radii = xp.asarray(knot_radii)
    speeds = xp.asarray(u_knots)
    interpolated = xp.interp(radius, radii, speeds)
    projected = xp.where(radius < radii[0], speeds[0] * radius / radii[0], interpolated)
    projected = xp.where(radius > radii[-1], speeds[-1], projected)
    return projected / xp.maximum(xp.sin(i_rad), 1.0e-6)


def _spacing_curvature(u_knots, knot_radii, xp):
    u = xp.asarray(u_knots) / 100.0
    r = xp.asarray(knot_radii) / xp.asarray(knot_radii[-1])
    slopes = (u[1:] - u[:-1]) / (r[1:] - r[:-1])
    return xp.sum((slopes[1:] - slopes[:-1]) ** 2)


def build_s3_objective(
    data,
    baseline_template,
    emissivity_basis: EmissivityBasis,
    knot_radii_arcsec,
    grid,
    covariance: FitCovariance,
    *,
    candidate,
    vsys_seed_kms,
    bmaj_arcsec,
    initial_emissivity_logits,
    two_zone_uses_rings=True,
    fixed_emissivity_weights=None,
):
    import jax
    import jax.numpy as jnp

    if candidate not in CANDIDATES:
        raise ValueError(f"unknown S3 candidate {candidate!r}")
    vis = jnp.asarray(data.vis)
    weights = jnp.asarray(data.weights)
    base = jnp.asarray(baseline_template)
    basis = jnp.asarray(emissivity_basis.images)
    knots = jnp.asarray(knot_radii_arcsec)
    initial_logits = jnp.asarray(initial_emissivity_logits)
    use_emissivity = candidate != "baseline_arctan" and fixed_emissivity_weights is None
    fixed_weights = (
        None
        if fixed_emissivity_weights is None
        else jnp.asarray(fixed_emissivity_weights)
    )
    use_rings = candidate == "supported_rings" or (
        candidate == "two_zone_dispersion" and two_zone_uses_rings
    )
    use_two_zone = candidate == "two_zone_dispersion"

    def objective(z):
        p = _decode(
            z,
            vsys_seed_kms=vsys_seed_kms,
            dv_kms=data.dv_kms,
            bmaj_arcsec=bmaj_arcsec,
        )
        if fixed_weights is not None:
            template = jnp.sum(fixed_weights[:, None, None] * basis, axis=0)
        elif use_emissivity:
            ew = _softmax_weights(p["emissivity_logits"][0], p["emissivity_logits"][1], jnp)
            template = jnp.sum(ew[:, None, None] * basis, axis=0)
        else:
            template = base
        if use_rings:
            def velocity_profile(radius):
                return _projected_ring_profile(
                    radius, knots, p["u_knots_kms"], p["i_rad"], jnp
                )
        else:
            velocity_profile = None
        if use_two_zone:
            transition = knots[1]
            width = 0.25 * float(bmaj_arcsec)

            def dispersion_profile(radius):
                outer_fraction = 0.5 * (1.0 + jnp.tanh((radius - transition) / width))
                return (
                    p["gas_sigma_kms"] * (1.0 - outer_fraction)
                    + p["sigma_outer_kms"] * outer_fraction
                )
        else:
            dispersion_profile = None
        model_params = {
            "flux": p["flux"],
            "pa_deg": p["pa_deg"],
            "vsys_kms": p["vsys_kms"],
            "gas_sigma_kms": p["gas_sigma_kms"],
            "dx_arcsec": p["dx_arcsec"],
            "dy_arcsec": p["dy_arcsec"],
            "v0_kms": p["arctan_u_kms"] / jnp.maximum(jnp.sin(p["i_rad"]), 1.0e-6),
            "r_t_arcsec": p["r_t_arcsec"],
        }
        model = predict_binned(
            data,
            model_params,
            template,
            grid,
            i_rad=p["i_rad"],
            xla=True,
            velocity_profile=velocity_profile,
            dispersion_profile=dispersion_profile,
        )
        c2 = correlated_chi2(vis, model, weights, covariance)
        center_prior = (p["dx_arcsec"] / 0.5) ** 2 + (p["dy_arcsec"] / 0.5) ** 2
        emissivity_prior = (
            0.25 * jnp.sum((p["emissivity_logits"] - initial_logits) ** 2)
            if use_emissivity
            else 0.0
        )
        ring_regularization = (
            _spacing_curvature(p["u_knots_kms"], knots, jnp) if use_rings else 0.0
        )
        return 0.5 * c2 + 0.5 * center_prior + emissivity_prior + ring_regularization

    return objective, jax.jit(jax.value_and_grad(objective))


def fit_s3_candidate(
    data,
    baseline_template,
    emissivity_basis,
    knot_radii_arcsec,
    grid,
    covariance,
    z0,
    *,
    candidate,
    pa_seed_deg,
    vsys_seed_kms,
    bmaj_arcsec,
    maxiter=120,
    two_zone_uses_rings=True,
    fixed_emissivity_weights=None,
):
    import jax.numpy as jnp

    bounds, active = chart_bounds(
        pa_seed_deg,
        data.dv_kms,
        bmaj_arcsec,
        candidate,
        two_zone_uses_rings=two_zone_uses_rings,
    )
    if fixed_emissivity_weights is not None:
        active = tuple(index for index in active if index not in (13, 14))
    z_start = np.asarray(z0, dtype=np.float64).copy()
    active_set = set(active)
    for index, (lo, hi) in enumerate(bounds):
        if index not in active_set:
            bounds[index] = (z_start[index], z_start[index])
    natural = np.asarray(emissivity_basis.natural_weights)
    initial_logits = np.log(natural[1:] / natural[0])
    objective, value_gradient = build_s3_objective(
        data,
        baseline_template,
        emissivity_basis,
        knot_radii_arcsec,
        grid,
        covariance,
        candidate=candidate,
        vsys_seed_kms=vsys_seed_kms,
        bmaj_arcsec=bmaj_arcsec,
        initial_emissivity_logits=initial_logits,
        two_zone_uses_rings=two_zone_uses_rings,
        fixed_emissivity_weights=fixed_emissivity_weights,
    )
    _ = value_gradient(jnp.asarray(z_start))
    normalization = 1.0 / max(1, int(data.vis.size))

    def raw(z):
        value, gradient = value_gradient(jnp.asarray(z))
        return float(value), np.asarray(gradient, dtype=np.float64)

    best_seen = {"value": float("inf"), "z": z_start.copy()}

    def scaled(z):
        value, gradient = raw(z)
        if value < best_seen["value"]:
            best_seen["value"] = value
            best_seen["z"] = np.asarray(z, dtype=np.float64).copy()
        return normalization * value, normalization * gradient

    opt = minimize(
        scaled,
        z_start,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1.0e-15, "gtol": 1.0e-8, "maxls": 40},
    )
    optimum = (
        best_seen["z"]
        if best_seen["value"] < float(opt.fun) / normalization
        else np.asarray(opt.x, dtype=np.float64)
    )
    value, gradient = raw(optimum)
    params = unpack_chart(
        optimum,
        vsys_seed_kms=vsys_seed_kms,
        dv_kms=data.dv_kms,
        bmaj_arcsec=bmaj_arcsec,
        natural_weights=natural,
    )
    if fixed_emissivity_weights is not None:
        params["emissivity_weights"] = np.asarray(
            fixed_emissivity_weights, dtype=np.float64
        ).tolist()
    prior = (params["dx_arcsec"] / 0.5) ** 2 + (params["dy_arcsec"] / 0.5) ** 2
    if candidate != "baseline_arctan" and fixed_emissivity_weights is None:
        logits = np.asarray(optimum[13:15])
        prior += 0.5 * float(np.sum((logits - initial_logits) ** 2))
    regularization = 0.0
    uses_rings = candidate == "supported_rings" or (
        candidate == "two_zone_dispersion" and two_zone_uses_rings
    )
    if uses_rings:
        regularization = float(
            _spacing_curvature(np.asarray(optimum[9:13]) * 100.0, knot_radii_arcsec, np)
        )
    chi2 = 2.0 * float(value) - float(prior) - 2.0 * regularization
    pg = projected_gradient(optimum, gradient, bounds)
    pg_raw = float(np.max(np.abs(pg[list(active)])))
    pg_per_complex = pg_raw / max(1, int(data.vis.size))
    boundary = []
    for index in active:
        lo, hi = bounds[index]
        span = hi - lo
        if span > 0.0 and (
            optimum[index] - lo <= 1.0e-4 * span
            or hi - optimum[index] <= 1.0e-4 * span
        ):
            boundary.append(S3_PARAMETER_NAMES[index])

    # The observed Hessian is a compact local identifiability diagnostic. It
    # is reported, not used to claim calibrated posterior uncertainty.
    hessian_full = np.zeros((len(S3_PARAMETER_NAMES), len(S3_PARAMETER_NAMES)))
    for index in active:
        step = 1.0e-4 * max(1.0, abs(float(optimum[index])))
        lo, hi = bounds[index]
        plus = optimum.copy()
        minus = optimum.copy()
        plus[index] = min(hi, plus[index] + step)
        minus[index] = max(lo, minus[index] - step)
        denominator = plus[index] - minus[index]
        if denominator <= 0.0:
            continue
        _, gradient_plus = raw(plus)
        _, gradient_minus = raw(minus)
        hessian_full[:, index] = (gradient_plus - gradient_minus) / denominator
    active_index = np.asarray(active, dtype=np.int64)
    hessian_active_raw = hessian_full[np.ix_(active_index, active_index)]
    hessian_active = 0.5 * (hessian_active_raw + hessian_active_raw.T) / max(
        1, int(data.vis.size)
    )
    singular = np.linalg.svd(hessian_active, compute_uv=False)
    positive = singular[singular > singular[0] * 1.0e-8] if singular.size else singular
    hessian = {
        "active_parameter_names": [S3_PARAMETER_NAMES[i] for i in active],
        "rank_relative_1e-8": int(positive.size),
        "dimension": int(len(active)),
        "condition_on_identified_subspace": (
            float(positive[0] / positive[-1]) if positive.size else None
        ),
        "singular_values_per_complex": singular.tolist(),
    }
    if uses_rings:
        knot_hessian = 0.5 * (
            hessian_full[9:13, 9:13] + hessian_full[9:13, 9:13].T
        ) / max(1, int(data.vis.size))
        knot_singular = np.linalg.svd(knot_hessian, compute_uv=False)
        knot_identified = knot_singular[
            knot_singular > knot_singular[0] * 1.0e-8
        ]
        hessian["velocity_knots"] = {
            "rank_relative_1e-8": int(knot_identified.size),
            "dimension": 4,
            "condition_on_identified_subspace": (
                float(knot_identified[0] / knot_identified[-1])
                if knot_identified.size
                else None
            ),
            "singular_values_per_complex": knot_singular.tolist(),
        }
    return S3FitResult(
        candidate=candidate,
        parameters=params,
        objective=float(value),
        chi2=float(chi2),
        prior=float(prior),
        regularization=float(regularization),
        projected_gradient_inf=pg_per_complex,
        projected_gradient_inf_raw=pg_raw,
        projected_gradient_normalization="raw_projected_gradient_divided_by_n_complex",
        nfev=int(opt.nfev),
        njev=int(opt.njev),
        nit=int(opt.nit),
        success=bool(opt.success),
        message=str(opt.message),
        boundary_parameters=tuple(boundary),
        hessian=hessian,
    ), np.asarray(optimum, dtype=np.float64)


__all__ = [
    "CANDIDATES",
    "EmissivityBasis",
    "S3FitResult",
    "build_positive_emissivity_basis",
    "supported_knot_radii",
    "initial_chart",
    "chart_bounds",
    "unpack_chart",
    "fit_s3_candidate",
]
