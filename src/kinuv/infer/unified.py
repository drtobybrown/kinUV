"""Common inference chart for the unified smooth radial model.

The likelihood delegates to the existing visibility renderer and frozen C1
quadratic form.  Profile smoothness and all other priors are returned
separately, so likelihood-only comparisons cannot accidentally include them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kinuv.forward.sb import galaxy_r_phi
from kinuv.infer.map import predict_binned
from kinuv.infer.s2 import FitCovariance, correlated_chi2
from kinuv.profiles.unified import (
    DISPERSION_SHAPE_DIM,
    ROTATION_SHAPE_DIM,
    UnifiedRadialSupport,
    build_unified_radial_support,
    projected_velocity,
    rotation_prior_components,
    velocity_dispersion,
)
from kinuv.xp import numpy_or_jax


UNIFIED_CHART_SCHEMA = "kinuv-unified-inference-chart-v1"
UNIFIED_PARAMETER_NAMES = (
    "log_flux",
    "pa_cycle_logit",
    "vsys_channel_offset",
    "dx_over_bmaj",
    "dy_over_bmaj",
    "cos_inclination_logit",
    "log_u_reference_kms",
    "rotation_log_omega_1",
    "rotation_log_omega_2",
    "rotation_log_omega_3",
    "rotation_log_omega_4",
    "log_sigma0_kms",
    "dispersion_log_mode_1",
    "dispersion_log_mode_2",
)


@dataclass(frozen=True)
class UnifiedChartSpec:
    """Target metadata and proper-prior centres for the common 14D chart."""

    support: UnifiedRadialSupport
    pa_reference_rad: float
    vsys_reference_kms: float
    dv_kms: float
    flux_reference: float
    u_reference_kms: float
    sigma_reference_kms: float
    log_flux_scale: float = 0.7
    vsys_scale_kms: float = 50.0
    center_scale_arcsec: float = 0.5
    log_u_scale: float = 0.7
    log_sigma_scale: float = 0.5
    cos_inclination_min: float = 1.0e-3
    cos_inclination_max: float = 0.999

    def __post_init__(self):
        if not np.isfinite(self.pa_reference_rad) or not np.isfinite(
            self.vsys_reference_kms
        ):
            raise ValueError("PA and systemic-velocity references must be finite")
        positive = {
            "dv_kms": self.dv_kms,
            "flux_reference": self.flux_reference,
            "u_reference_kms": self.u_reference_kms,
            "sigma_reference_kms": self.sigma_reference_kms,
            "log_flux_scale": self.log_flux_scale,
            "vsys_scale_kms": self.vsys_scale_kms,
            "center_scale_arcsec": self.center_scale_arcsec,
            "log_u_scale": self.log_u_scale,
            "log_sigma_scale": self.log_sigma_scale,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not (
            0.0
            < self.cos_inclination_min
            < self.cos_inclination_max
            < 1.0
        ):
            raise ValueError("cos-inclination bounds must lie inside (0, 1)")

    @property
    def dimension(self):
        return len(UNIFIED_PARAMETER_NAMES)

    def metadata(self):
        return {
            "schema_version": UNIFIED_CHART_SCHEMA,
            "parameter_names": list(UNIFIED_PARAMETER_NAMES),
            "dimension": self.dimension,
            "profile_support": self.support.metadata(),
            "likelihood_prior_separated": True,
            "target_id_dispatch": False,
            "floating_rotation_radius": False,
        }


@dataclass(frozen=True)
class UnifiedLogDensity:
    log_likelihood: Callable
    log_prior: Callable
    log_posterior: Callable


def build_support_from_template(
    template,
    grid,
    *,
    pa_rad,
    i_rad,
    bmaj_arcsec,
):
    """Apply the common support rule to an empirical positive morphology."""
    image = np.maximum(np.asarray(template, dtype=np.float64), 0.0)
    radius, _ = galaxy_r_phi(grid, float(pa_rad), float(i_rad))
    return build_unified_radial_support(
        radius,
        image,
        bmaj_arcsec=float(bmaj_arcsec),
        cell_arcsec=float(grid.cell_arcsec),
    )


def _sigmoid(value, xp):
    return 0.5 * (1.0 + xp.tanh(0.5 * value))


def decode_unified_chart(z, spec: UnifiedChartSpec):
    """Map the common unconstrained chart to physical model coordinates."""
    xp = numpy_or_jax(z)
    values = xp.asarray(z)
    if tuple(values.shape) != (len(UNIFIED_PARAMETER_NAMES),):
        raise ValueError(
            f"unified chart must have shape {(len(UNIFIED_PARAMETER_NAMES),)}"
        )
    pa_probability = _sigmoid(values[1], xp)
    pa_rad = spec.pa_reference_rad - np.pi + 2.0 * np.pi * pa_probability
    inclination_probability = _sigmoid(values[5], xp)
    cos_i = spec.cos_inclination_min + (
        spec.cos_inclination_max - spec.cos_inclination_min
    ) * inclination_probability
    return {
        "flux": xp.exp(values[0]),
        "pa_rad": pa_rad,
        "vsys_kms": spec.vsys_reference_kms + values[2] * spec.dv_kms,
        "dx_arcsec": values[3] * spec.support.bmaj_arcsec,
        "dy_arcsec": values[4] * spec.support.bmaj_arcsec,
        "cos_inclination": cos_i,
        "i_rad": xp.arccos(cos_i),
        "u_reference_kms": xp.exp(values[6]),
        "rotation_shape": values[7:11],
        "sigma0_kms": xp.exp(values[11]),
        "dispersion_shape": values[12:14],
    }


def encode_unified_chart(parameters, spec: UnifiedChartSpec):
    """Encode physical model coordinates into the common chart."""
    pa_low = spec.pa_reference_rad - np.pi
    pa_probability = ((float(parameters["pa_rad"]) - pa_low) % (2.0 * np.pi)) / (
        2.0 * np.pi
    )
    pa_probability = np.clip(pa_probability, 1.0e-12, 1.0 - 1.0e-12)
    cos_i = float(parameters["cos_inclination"])
    if not spec.cos_inclination_min < cos_i < spec.cos_inclination_max:
        raise ValueError("cos_inclination lies outside the chart support")
    for name in ("flux", "u_reference_kms", "sigma0_kms"):
        if not np.isfinite(parameters[name]) or float(parameters[name]) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    rotation_shape = np.asarray(parameters["rotation_shape"], dtype=np.float64)
    dispersion_shape = np.asarray(parameters["dispersion_shape"], dtype=np.float64)
    if rotation_shape.shape != (ROTATION_SHAPE_DIM,):
        raise ValueError(f"rotation_shape must have shape {(ROTATION_SHAPE_DIM,)}")
    if dispersion_shape.shape != (DISPERSION_SHAPE_DIM,):
        raise ValueError(
            f"dispersion_shape must have shape {(DISPERSION_SHAPE_DIM,)}"
        )
    cos_probability = (cos_i - spec.cos_inclination_min) / (
        spec.cos_inclination_max - spec.cos_inclination_min
    )
    cos_probability = np.clip(cos_probability, 1.0e-12, 1.0 - 1.0e-12)
    logit = lambda probability: np.log(probability) - np.log1p(-probability)
    return np.array(
        [
            np.log(float(parameters["flux"])),
            logit(pa_probability),
            (float(parameters["vsys_kms"]) - spec.vsys_reference_kms)
            / spec.dv_kms,
            float(parameters["dx_arcsec"]) / spec.support.bmaj_arcsec,
            float(parameters["dy_arcsec"]) / spec.support.bmaj_arcsec,
            logit(cos_probability),
            np.log(float(parameters["u_reference_kms"])),
            *rotation_shape,
            np.log(float(parameters["sigma0_kms"])),
            *dispersion_shape,
        ],
        dtype=np.float64,
    )


def initial_unified_chart(spec: UnifiedChartSpec, *, inclination_deg: float):
    return encode_unified_chart(
        {
            "flux": spec.flux_reference,
            "pa_rad": spec.pa_reference_rad,
            "vsys_kms": spec.vsys_reference_kms,
            "dx_arcsec": 0.0,
            "dy_arcsec": 0.0,
            "cos_inclination": np.cos(np.radians(float(inclination_deg))),
            "u_reference_kms": spec.u_reference_kms,
            "rotation_shape": np.zeros(ROTATION_SHAPE_DIM),
            "sigma0_kms": spec.sigma_reference_kms,
            "dispersion_shape": np.zeros(DISPERSION_SHAPE_DIM),
        },
        spec,
    )


def unified_profile_callables(z, spec: UnifiedChartSpec):
    """Return callables accepted by the existing visibility renderer."""
    physical = decode_unified_chart(z, spec)
    sin_i = numpy_or_jax(z).maximum(
        numpy_or_jax(z).sin(physical["i_rad"]), 1.0e-6
    )

    def velocity_profile(radius_arcsec):
        u = projected_velocity(
            radius_arcsec,
            numpy_or_jax(z).log(physical["u_reference_kms"]),
            physical["rotation_shape"],
            spec.support,
        )
        return u / sin_i

    def dispersion_profile(radius_arcsec):
        return velocity_dispersion(
            radius_arcsec,
            numpy_or_jax(z).log(physical["sigma0_kms"]),
            physical["dispersion_shape"],
            spec.support,
        )

    return velocity_profile, dispersion_profile


def _normal_log_density(value, mean, scale, xp):
    standardized = (value - mean) / scale
    return (
        -0.5 * standardized * standardized
        - xp.log(xp.asarray(scale))
        - 0.5 * np.log(2.0 * np.pi)
    )


def unified_prior_components(z, spec: UnifiedChartSpec):
    """Return named proper-prior terms in chart coordinates."""
    xp = numpy_or_jax(z)
    values = xp.asarray(z)
    pa_probability = _sigmoid(values[1], xp)
    inclination_probability = _sigmoid(values[5], xp)
    rotation = rotation_prior_components(values[7:11], spec.support)
    return {
        "flux": _normal_log_density(
            values[0], np.log(spec.flux_reference), spec.log_flux_scale, xp
        ),
        # Uniform physical PA over exactly one 2pi cycle; normalization and
        # transform derivative cancel to the standard logistic density.
        "pa": xp.log(pa_probability) + xp.log1p(-pa_probability),
        "vsys": _normal_log_density(
            values[2], 0.0, spec.vsys_scale_kms / spec.dv_kms, xp
        ),
        "center_x": _normal_log_density(
            values[3], 0.0, spec.center_scale_arcsec / spec.support.bmaj_arcsec, xp
        ),
        "center_y": _normal_log_density(
            values[4], 0.0, spec.center_scale_arcsec / spec.support.bmaj_arcsec, xp
        ),
        # Uniform physical cos(i) inside the declared open interval.
        "inclination": xp.log(inclination_probability)
        + xp.log1p(-inclination_probability),
        "u_reference": _normal_log_density(
            values[6], np.log(spec.u_reference_kms), spec.log_u_scale, xp
        ),
        **rotation,
        "sigma0": _normal_log_density(
            values[11], np.log(spec.sigma_reference_kms), spec.log_sigma_scale, xp
        ),
        "dispersion": -0.5 * xp.sum(values[12:14] ** 2)
        - 0.5 * DISPERSION_SHAPE_DIM * np.log(2.0 * np.pi),
    }


def unified_log_prior(z, spec: UnifiedChartSpec):
    components = unified_prior_components(z, spec)
    return sum(components.values())


def build_unified_log_density(
    data,
    template,
    grid,
    covariance: FitCovariance,
    spec: UnifiedChartSpec,
):
    """Build the exact C1 visibility density on the common chart."""
    template_array = np.asarray(template)

    def log_likelihood(z):
        xp = numpy_or_jax(z)
        physical = decode_unified_chart(z, spec)
        velocity_profile, dispersion_profile = unified_profile_callables(z, spec)
        model_parameters = {
            "flux": physical["flux"],
            "pa_deg": physical["pa_rad"] * (180.0 / np.pi),
            "vsys_kms": physical["vsys_kms"],
            "gas_sigma_kms": physical["sigma0_kms"],
            "dx_arcsec": physical["dx_arcsec"],
            "dy_arcsec": physical["dy_arcsec"],
            # These compatibility values are ignored by the custom profiles.
            "v0_kms": physical["u_reference_kms"]
            / xp.maximum(xp.sin(physical["i_rad"]), 1.0e-6),
            "r_t_arcsec": spec.support.outer_radius_arcsec,
        }
        model = predict_binned(
            data,
            model_parameters,
            template_array,
            grid,
            i_rad=physical["i_rad"],
            xla=True,
            velocity_profile=velocity_profile,
            dispersion_profile=dispersion_profile,
        )
        chi2 = correlated_chi2(
            xp.asarray(data.vis), xp.asarray(model), xp.asarray(data.weights), covariance
        )
        return -0.5 * chi2

    def log_prior(z):
        return unified_log_prior(z, spec)

    def log_posterior(z):
        return log_likelihood(z) + log_prior(z)

    return UnifiedLogDensity(log_likelihood, log_prior, log_posterior)
