"""Target-neutral smooth projected-velocity and dispersion profiles.

The common profile fits projected circular speed ``u = V_rot sin(i)``.  Its
rotation component is a direct cubic spline in log angular speed: ``u(R) =
R exp(g(R))``.  There is no fitted turnover/carrier radius.  Fixed boundary
terms give a finite central slope and a C2 join to the constant outer
extension.  The two-mode log-dispersion profile reduces exactly to a constant
when both deviations are zero.

All radii are angular.  This module contains no target identifiers, likelihood,
distance conversion, or mass model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from kinuv.xp import numpy_or_jax


UNIFIED_PROFILE_SCHEMA = "kinuv-unified-radial-support-v1"
ROTATION_SHAPE_DIM = 4
DISPERSION_SHAPE_DIM = 2
ROTATION_CURVATURE_STRENGTH = 1.0
ROTATION_NULLSPACE_STRENGTH = 0.05
DISPERSION_LOG_SCALE = 0.25


@dataclass(frozen=True)
class UnifiedRadialSupport:
    """Fixed beam/emission-derived radial design shared by all targets."""

    bmaj_arcsec: float
    cell_arcsec: float | None
    emission_r20_arcsec: float
    emission_r50_arcsec: float
    emission_r70_arcsec: float
    emission_r80_arcsec: float
    emission_r95_arcsec: float
    reference_radius_arcsec: float
    outer_radius_arcsec: float
    rotation_knots_scaled: np.ndarray
    rotation_modes: np.ndarray
    rotation_curvature_precision: np.ndarray
    rotation_nullspace_precision: np.ndarray
    dispersion_central_scale_arcsec: float
    dispersion_transition_arcsec: float
    dispersion_transition_width_arcsec: float

    def metadata(self) -> dict:
        rotation_nodes = np.unique(self.rotation_knots_scaled)
        return {
            "schema_version": UNIFIED_PROFILE_SCHEMA,
            "algorithm": "beam-emission-quantile-cubic-v1",
            "dimensions": {
                "rotation_shape": ROTATION_SHAPE_DIM,
                "dispersion_shape": DISPERSION_SHAPE_DIM,
            },
            "bmaj_arcsec": self.bmaj_arcsec,
            "cell_arcsec": self.cell_arcsec,
            "emission_flux_quantiles_arcsec": {
                "emission_r20": self.emission_r20_arcsec,
                "emission_r50": self.emission_r50_arcsec,
                "emission_r70": self.emission_r70_arcsec,
                "emission_r80": self.emission_r80_arcsec,
                "emission_r95": self.emission_r95_arcsec,
            },
            "reference_radius_arcsec": self.reference_radius_arcsec,
            "outer_radius_arcsec": self.outer_radius_arcsec,
            "rotation_nodes_arcsec": (
                rotation_nodes * self.outer_radius_arcsec
            ).tolist(),
            "subbeam_rotation_node_count": int(
                np.sum(
                    (rotation_nodes > 0.0)
                    & (
                        rotation_nodes * self.outer_radius_arcsec
                        < self.bmaj_arcsec
                    )
                )
            ),
            "beam_is_scale_not_cutoff": True,
            "rotation_prior": {
                "coordinate": "dimensionless_radius_R_over_Router",
                "curvature": "integrated_squared_second_derivative",
                "curvature_strength": ROTATION_CURVATURE_STRENGTH,
                "nullspace_strength": ROTATION_NULLSPACE_STRENGTH,
                "proper": True,
            },
            "dispersion_prior": {
                "coordinate": "log_sigma",
                "mode_scale": DISPERSION_LOG_SCALE,
                "central_gaussian_scale_arcsec": self.dispersion_central_scale_arcsec,
                "outer_transition_arcsec": self.dispersion_transition_arcsec,
                "outer_transition_width_arcsec": self.dispersion_transition_width_arcsec,
                "zero_deviations_are_exactly_constant": True,
            },
            "outer_boundary": {
                "kind": "C2_constant_projected_speed_extension",
                "role": "computational_continuation_beyond_emission_R95",
                "supplies_plateau_or_R50_evidence": False,
            },
        }


def _weighted_quantiles(radius, weight, probabilities):
    r = np.asarray(radius, dtype=np.float64).ravel()
    w = np.asarray(weight, dtype=np.float64).ravel()
    if r.shape != w.shape:
        raise ValueError("radius and emissivity_weight must have the same shape")
    good = np.isfinite(r) & np.isfinite(w) & (r >= 0.0) & (w > 0.0)
    if not np.any(good):
        raise ValueError("positive finite radial support is empty")
    order = np.argsort(r[good])
    sorted_radius = r[good][order]
    sorted_weight = w[good][order]
    cdf = np.cumsum(sorted_weight)
    cdf /= cdf[-1]
    return np.interp(np.asarray(probabilities, dtype=np.float64), cdf, sorted_radius)


def _clamped_knots(internal):
    values = np.asarray(internal, dtype=np.float64)
    if values.ndim != 1 or np.any(np.diff(values) <= 0.0):
        raise ValueError("internal knots must be strictly increasing")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("internal knots must lie strictly inside (0, 1)")
    return np.concatenate((np.zeros(4), values, np.ones(4)))


def _basis_operator(knots, derivative_order=0, points=None):
    knot = np.asarray(knots, dtype=np.float64)
    n_basis = knot.size - 4
    if points is None:
        points = np.array([1.0])
    x = np.asarray(points, dtype=np.float64)
    rows = []
    for index in range(n_basis):
        coefficient = np.zeros(n_basis)
        coefficient[index] = 1.0
        spline = BSpline(knot, coefficient, 3)
        if derivative_order:
            spline = spline.derivative(derivative_order)
        rows.append(spline(x))
    return np.stack(rows, axis=-1)


def _constrained_modes(knots, dimension):
    n_basis = len(knots) - 4
    constraints = np.vstack(
        (
            np.ones(n_basis),
            _basis_operator(knots, 1, [1.0])[0],
            _basis_operator(knots, 2, [1.0])[0],
        )
    )
    modes = null_space(constraints)
    if modes.shape != (n_basis, dimension):
        raise RuntimeError(
            f"unexpected constrained spline dimension {modes.shape}; "
            f"expected {(n_basis, dimension)}"
        )
    return modes


def _integrated_gram(knots, derivative_order, modes):
    """Gauss-Legendre integral in dimensionless physical knot spacing."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    gram = np.zeros((modes.shape[1], modes.shape[1]), dtype=np.float64)
    for lower, upper in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        if upper <= lower:
            continue
        x = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
        operator = _basis_operator(knots, derivative_order, x) @ modes
        gram += 0.5 * (upper - lower) * (
            operator.T @ (weights[:, None] * operator)
        )
    return 0.5 * (gram + gram.T)


def build_unified_radial_support(
    radius_arcsec,
    emissivity_weight,
    *,
    bmaj_arcsec: float,
    cell_arcsec: float | None = None,
) -> UnifiedRadialSupport:
    """Construct the fixed-dimensional common support from metadata and light.

    The same deterministic rule is used for every target.  The first internal
    rotation node lies at or inside ``0.5 BMAJ``; the beam sets a scale and does
    not remove sub-beam profile support.
    """
    bmaj = float(bmaj_arcsec)
    if not np.isfinite(bmaj) or bmaj <= 0.0:
        raise ValueError("bmaj_arcsec must be positive and finite")
    cell = None if cell_arcsec is None else float(cell_arcsec)
    if cell is not None and (not np.isfinite(cell) or cell <= 0.0):
        raise ValueError("cell_arcsec must be positive and finite")
    r20, r50, r70, r80, r95 = _weighted_quantiles(
        radius_arcsec, emissivity_weight, (0.20, 0.50, 0.70, 0.80, 0.95)
    )
    if r70 <= 0.0 or r95 <= 0.0:
        raise ValueError("emission support must extend to positive radius")
    outer = max(float(r95) + bmaj, 2.0 * bmaj)
    x_subbeam = min(0.5 * bmaj / outer, 0.20)
    x_middle = np.clip(float(r50) / outer, x_subbeam + 0.10, 0.70)
    x_outer = np.clip(
        float(r80) / outer,
        max(x_middle + 0.10, 0.65),
        0.90,
    )
    rotation_knots = _clamped_knots([x_subbeam, x_middle, x_outer])
    rotation_modes = _constrained_modes(rotation_knots, ROTATION_SHAPE_DIM)
    curvature = _integrated_gram(rotation_knots, 2, rotation_modes)
    nullspace = _integrated_gram(rotation_knots, 0, rotation_modes)
    total_precision = (
        ROTATION_CURVATURE_STRENGTH * curvature
        + ROTATION_NULLSPACE_STRENGTH * nullspace
    )
    if np.min(np.linalg.eigvalsh(total_precision)) <= 0.0:
        raise RuntimeError("rotation smoothness prior is not positive definite")

    support = UnifiedRadialSupport(
        bmaj_arcsec=bmaj,
        cell_arcsec=cell,
        emission_r20_arcsec=float(r20),
        emission_r50_arcsec=float(r50),
        emission_r70_arcsec=float(r70),
        emission_r80_arcsec=float(r80),
        emission_r95_arcsec=float(r95),
        reference_radius_arcsec=float(r70),
        outer_radius_arcsec=float(outer),
        rotation_knots_scaled=rotation_knots,
        rotation_modes=rotation_modes,
        rotation_curvature_precision=curvature,
        rotation_nullspace_precision=nullspace,
        dispersion_central_scale_arcsec=bmaj,
        dispersion_transition_arcsec=float(r50),
        dispersion_transition_width_arcsec=0.5 * bmaj,
    )
    if support.metadata()["subbeam_rotation_node_count"] < 1:
        raise RuntimeError("common radial design lacks a sub-beam rotation node")
    return support


def _bspline_basis(x, knots, xp):
    """JAX-compatible clamped cubic B-spline basis."""
    values = xp.asarray(x)
    knot = xp.asarray(knots)
    bounded = xp.clip(values, 0.0, 1.0)
    basis = xp.stack(
        [
            ((bounded >= knot[index]) & (bounded < knot[index + 1])).astype(
                values.dtype
            )
            for index in range(len(knots) - 1)
        ],
        axis=-1,
    )
    for order in range(1, 4):
        next_basis = []
        for index in range(len(knots) - order - 1):
            left_denominator = float(knots[index + order] - knots[index])
            right_denominator = float(
                knots[index + order + 1] - knots[index + 1]
            )
            left = (
                xp.zeros_like(bounded)
                if left_denominator == 0.0
                else (bounded - knot[index])
                * basis[..., index]
                / left_denominator
            )
            right = (
                xp.zeros_like(bounded)
                if right_denominator == 0.0
                else (knot[index + order + 1] - bounded)
                * basis[..., index + 1]
                / right_denominator
            )
            next_basis.append(left + right)
        basis = xp.stack(next_basis, axis=-1)
    endpoint = bounded == 1.0
    basis = xp.where(endpoint[..., None], xp.zeros_like(basis), basis)
    if hasattr(basis, "at"):
        basis = basis.at[..., -1].set(
            xp.where(endpoint, 1.0, basis[..., -1])
        )
    else:
        basis = np.asarray(basis).copy()
        basis[..., -1] = np.where(endpoint, 1.0, basis[..., -1])
    return basis


def rotation_shape_features(radius_arcsec, support: UnifiedRadialSupport):
    xp = numpy_or_jax(radius_arcsec)
    radius = xp.asarray(radius_arcsec)
    x = radius / support.outer_radius_arcsec
    reference_x = support.reference_radius_arcsec / support.outer_radius_arcsec
    modes = xp.asarray(support.rotation_modes)
    values = _bspline_basis(x, support.rotation_knots_scaled, xp) @ modes
    reference = (
        _bspline_basis(xp.asarray(reference_x), support.rotation_knots_scaled, xp)
        @ modes
    )
    return values - reference


def projected_velocity(
    radius_arcsec,
    log_u_reference_kms,
    rotation_shape,
    support: UnifiedRadialSupport,
):
    """Evaluate positive projected speed with an exact regular origin.

    The fixed quadratic boundary term has first and second derivatives ``-1``
    and ``+1`` at scaled radius one.  Combined with ``u=R exp(g)`` and the
    constrained spline modes, this gives a C2 join to constant speed beyond the
    measured outer support.
    """
    xp = numpy_or_jax(radius_arcsec, log_u_reference_kms, rotation_shape)
    radius = xp.asarray(radius_arcsec)
    if xp is np and np.any(radius < 0.0):
        raise ValueError("radius_arcsec must be nonnegative")
    shape = xp.asarray(rotation_shape)
    if tuple(shape.shape) != (ROTATION_SHAPE_DIM,):
        raise ValueError(
            f"rotation_shape must have shape {(ROTATION_SHAPE_DIM,)}"
        )
    outer = support.outer_radius_arcsec
    reference = support.reference_radius_arcsec
    # The outer branch below supplies the continuation.  Keep the interior
    # coordinate unclipped here so autodiff sees the full left derivative at
    # the exact join rather than the half-gradient convention of minimum().
    x = radius / outer
    reference_x = reference / outer

    def boundary(value):
        return -2.0 * value + 0.5 * value * value

    log_angular_ratio = (
        boundary(x)
        - boundary(reference_x)
        + rotation_shape_features(radius, support) @ shape
    )
    inside = (
        xp.exp(xp.asarray(log_u_reference_kms))
        * (radius / reference)
        * xp.exp(log_angular_ratio)
    )
    outer_shape = rotation_shape_features(xp.asarray(outer), support) @ shape
    outer_value = (
        xp.exp(xp.asarray(log_u_reference_kms))
        * (outer / reference)
        * xp.exp(
            boundary(1.0) - boundary(reference_x) + outer_shape
        )
    )
    return xp.where(radius <= outer, inside, outer_value)


def dispersion_shape_features(radius_arcsec, support: UnifiedRadialSupport):
    xp = numpy_or_jax(radius_arcsec)
    radius = xp.asarray(radius_arcsec)
    reference = support.reference_radius_arcsec
    central_scale = support.dispersion_central_scale_arcsec
    transition = support.dispersion_transition_arcsec
    width = support.dispersion_transition_width_arcsec

    central = xp.exp(-0.5 * (radius / central_scale) ** 2)
    central_reference = np.exp(-0.5 * (reference / central_scale) ** 2)
    outer = 0.5 * (1.0 + xp.tanh((radius - transition) / width))
    outer_reference = 0.5 * (1.0 + np.tanh((reference - transition) / width))
    return DISPERSION_LOG_SCALE * xp.stack(
        (central - central_reference, outer - outer_reference), axis=-1
    )


def velocity_dispersion(
    radius_arcsec,
    log_sigma0_kms,
    dispersion_shape,
    support: UnifiedRadialSupport,
):
    """Evaluate the positive two-mode log-dispersion spline."""
    xp = numpy_or_jax(radius_arcsec, log_sigma0_kms, dispersion_shape)
    if xp is np and np.any(np.asarray(radius_arcsec) < 0.0):
        raise ValueError("radius_arcsec must be nonnegative")
    deviations = xp.asarray(dispersion_shape)
    if tuple(deviations.shape) != (DISPERSION_SHAPE_DIM,):
        raise ValueError(
            f"dispersion_shape must have shape {(DISPERSION_SHAPE_DIM,)}"
        )
    log_sigma = xp.asarray(log_sigma0_kms) + (
        dispersion_shape_features(radius_arcsec, support) @ deviations
    )
    return xp.exp(log_sigma)


def rotation_prior_components(rotation_shape, support: UnifiedRadialSupport):
    """Return curvature and nullspace log-prior terms separately."""
    xp = numpy_or_jax(rotation_shape)
    coefficients = xp.asarray(rotation_shape)
    if tuple(coefficients.shape) != (ROTATION_SHAPE_DIM,):
        raise ValueError(
            f"rotation_shape must have shape {(ROTATION_SHAPE_DIM,)}"
        )
    curvature = xp.asarray(support.rotation_curvature_precision)
    nullspace = xp.asarray(support.rotation_nullspace_precision)
    precision = rotation_prior_precision(support)
    _, log_determinant = np.linalg.slogdet(precision)
    return {
        "rotation_curvature": -0.5
        * ROTATION_CURVATURE_STRENGTH
        * coefficients @ curvature @ coefficients,
        "rotation_nullspace": -0.5
        * ROTATION_NULLSPACE_STRENGTH
        * coefficients @ nullspace @ coefficients,
        "rotation_normalization": 0.5 * float(log_determinant)
        - 0.5 * ROTATION_SHAPE_DIM * np.log(2.0 * np.pi),
    }


def rotation_prior_precision(support: UnifiedRadialSupport):
    return (
        ROTATION_CURVATURE_STRENGTH
        * np.asarray(support.rotation_curvature_precision)
        + ROTATION_NULLSPACE_STRENGTH
        * np.asarray(support.rotation_nullspace_precision)
    )
