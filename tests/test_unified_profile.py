from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from kinuv.forward.model import los_velocity
from kinuv.infer.unified import (
    UNIFIED_PARAMETER_NAMES,
    UnifiedChartSpec,
    build_unified_log_density,
    decode_unified_chart,
    encode_unified_chart,
    initial_unified_chart,
    unified_log_prior,
    unified_prior_components,
    unified_profile_callables,
)
from kinuv.infer.s2 import FitCovariance, correlated_chi2
from kinuv.profiles.unified import (
    DISPERSION_SHAPE_DIM,
    ROTATION_SHAPE_DIM,
    build_unified_radial_support,
    projected_velocity,
    rotation_prior_components,
    rotation_prior_precision,
    velocity_dispersion,
)


def _support():
    radius = np.linspace(0.0, 12.0, 1201)
    emissivity = radius * np.exp(-radius / 2.0)
    emissivity[0] = emissivity[1]
    return build_unified_radial_support(
        radius,
        emissivity,
        bmaj_arcsec=1.2,
        cell_arcsec=0.2,
    )


def _spec():
    return UnifiedChartSpec(
        support=_support(),
        pa_reference_rad=0.7,
        vsys_reference_kms=8100.0,
        dv_kms=5.0,
        flux_reference=40.0,
        u_reference_kms=170.0,
        sigma_reference_kms=9.0,
    )


def test_common_support_has_fixed_dimensions_and_subbeam_node():
    support = _support()
    metadata = support.metadata()
    nodes = np.asarray(metadata["rotation_nodes_arcsec"])
    assert support.rotation_modes.shape == (7, ROTATION_SHAPE_DIM)
    assert metadata["dimensions"] == {
        "rotation_shape": ROTATION_SHAPE_DIM,
        "dispersion_shape": DISPERSION_SHAPE_DIM,
    }
    assert np.any((nodes > 0.0) & (nodes < support.bmaj_arcsec))
    assert support.outer_radius_arcsec >= (
        support.emission_r95_arcsec + support.bmaj_arcsec
    )
    assert metadata["outer_boundary"]["supplies_plateau_or_R50_evidence"] is False
    assert "emission_r50" in metadata["emission_flux_quantiles_arcsec"]


def test_projected_velocity_is_positive_regular_and_c2_at_outer_join():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    support = _support()
    shape = jnp.asarray([0.12, -0.08, 0.04, 0.02])
    log_u = jnp.log(160.0)

    profile = lambda radius: projected_velocity(radius, log_u, shape, support)
    assert float(profile(jnp.asarray(0.0))) == pytest.approx(0.0, abs=0.0)
    assert float(profile(jnp.asarray(support.reference_radius_arcsec))) == pytest.approx(
        160.0, rel=1.0e-12
    )
    sampled = np.asarray(profile(jnp.linspace(0.0, 1.5 * support.outer_radius_arcsec, 300)))
    assert np.all(sampled[1:] > 0.0)

    central_slope = float(jax.grad(profile)(jnp.asarray(0.0)))
    small_radius = 1.0e-7 * support.bmaj_arcsec
    small_limit = float(profile(jnp.asarray(small_radius)) / small_radius)
    assert np.isfinite(central_slope) and central_slope > 0.0
    assert central_slope == pytest.approx(small_limit, rel=2.0e-6)

    first = jax.grad(profile)
    second = jax.grad(first)
    outer = jnp.asarray(support.outer_radius_arcsec)
    assert float(first(outer)) == pytest.approx(0.0, abs=1.0e-7)
    assert float(second(outer)) == pytest.approx(0.0, abs=1.0e-6)
    assert float(profile(1.2 * outer)) == pytest.approx(float(profile(outer)), rel=1.0e-12)


def test_rotation_curvature_prior_is_spacing_aware_proper_gaussian():
    support = _support()
    precision = rotation_prior_precision(support)
    eigenvalues = np.linalg.eigvalsh(precision)
    assert precision.shape == (ROTATION_SHAPE_DIM, ROTATION_SHAPE_DIM)
    assert np.all(eigenvalues > 0.0)
    zero = rotation_prior_components(np.zeros(ROTATION_SHAPE_DIM), support)
    nonzero = rotation_prior_components(
        np.array([0.2, -0.1, 0.3, 0.05]), support
    )
    assert zero["rotation_curvature"] == pytest.approx(0.0)
    assert zero["rotation_nullspace"] == pytest.approx(0.0)
    assert sum(nonzero.values()) < sum(zero.values())
    assert support.metadata()["rotation_prior"]["coordinate"] == (
        "dimensionless_radius_R_over_Router"
    )
    assert support.metadata()["rotation_prior"]["proper"] is True


def test_two_mode_log_dispersion_has_exact_constant_limit_and_fixed_scales():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    support = _support()
    radius = jnp.linspace(0.0, 2.0 * support.outer_radius_arcsec, 500)
    constant = velocity_dispersion(radius, jnp.log(11.0), jnp.zeros(2), support)
    np.testing.assert_allclose(np.asarray(constant), 11.0, rtol=1.0e-12, atol=0.0)
    varying = lambda value: velocity_dispersion(
        value, jnp.log(11.0), jnp.asarray([0.4, -0.3]), support
    )
    values = np.asarray(jax.vmap(varying)(radius))
    assert np.all(np.isfinite(values)) and np.all(values > 0.0)
    assert np.isfinite(float(jax.grad(varying)(jnp.asarray(support.bmaj_arcsec))))
    metadata = support.metadata()["dispersion_prior"]
    assert metadata["central_gaussian_scale_arcsec"] == pytest.approx(
        support.bmaj_arcsec
    )
    assert metadata["outer_transition_arcsec"] == pytest.approx(
        support.emission_r50_arcsec
    )
    assert metadata["zero_deviations_are_exactly_constant"] is True


def test_common_chart_round_trip_and_renderer_profile_contract():
    spec = _spec()
    z = initial_unified_chart(spec, inclination_deg=52.0)
    assert len(UNIFIED_PARAMETER_NAMES) == 14
    assert z.shape == (14,)
    physical = decode_unified_chart(z, spec)
    assert float(physical["pa_rad"]) == pytest.approx(spec.pa_reference_rad)
    assert float(physical["u_reference_kms"]) == pytest.approx(
        spec.u_reference_kms
    )
    np.testing.assert_allclose(encode_unified_chart(physical, spec), z, atol=1.0e-12)
    assert np.isfinite(float(unified_log_prior(z, spec)))
    components = unified_prior_components(z, spec)
    assert "rotation_curvature" in components
    assert "rotation_nullspace" in components

    velocity_profile, dispersion_profile = unified_profile_callables(z, spec)
    radius = 2.0
    expected_u = float(
        projected_velocity(
            radius,
            np.log(float(physical["u_reference_kms"])),
            physical["rotation_shape"],
            spec.support,
        )
    )
    # PA=0 puts the receding major axis on +north.  Use an otherwise identical
    # spec so the production LOS projection should add exactly projected u.
    zero_pa = UnifiedChartSpec(
        **{**spec.__dict__, "pa_reference_rad": 0.0}
    )
    z_zero_pa = initial_unified_chart(zero_pa, inclination_deg=52.0)
    physical_zero_pa = decode_unified_chart(z_zero_pa, zero_pa)
    velocity_zero_pa, dispersion_zero_pa = unified_profile_callables(
        z_zero_pa, zero_pa
    )
    got = los_velocity(
        0.0,
        radius,
        physical_zero_pa["pa_rad"],
        physical_zero_pa["i_rad"],
        1000.0,
        velocity_profile=velocity_zero_pa,
    )
    expected_zero_pa = projected_velocity(
        radius,
        np.log(zero_pa.u_reference_kms),
        np.zeros(ROTATION_SHAPE_DIM),
        zero_pa.support,
    )
    assert float(got) == pytest.approx(1000.0 + float(expected_zero_pa))
    assert float(dispersion_profile(radius)) > 0.0
    assert float(dispersion_zero_pa(radius)) == pytest.approx(
        zero_pa.sigma_reference_kms
    )


def test_chart_metadata_has_no_target_or_floating_radius_dispatch():
    metadata = _spec().metadata()
    assert metadata["parameter_names"] == list(UNIFIED_PARAMETER_NAMES)
    assert metadata["target_id_dispatch"] is False
    assert metadata["floating_rotation_radius"] is False


def test_log_density_keeps_c1_likelihood_and_prior_separate(monkeypatch):
    import kinuv.infer.unified as module

    spec = _spec()
    z = initial_unified_chart(spec, inclination_deg=52.0)
    data = SimpleNamespace(
        vis=np.array([[1.0 + 2.0j, 0.5 - 0.25j, -0.4 + 0.1j]]),
        weights=np.array([[2.0, 1.0, 3.0]]),
    )
    model = np.array([[0.2 + 0.1j, 0.0 + 0.0j, -0.1 + 0.0j]])
    covariance = FitCovariance(1.4, 0.2, 1.0, 0.2, 1)
    monkeypatch.setattr(module, "predict_binned", lambda *args, **kwargs: model)
    density = build_unified_log_density(
        data, np.ones((2, 2)), object(), covariance, spec
    )
    expected = -0.5 * correlated_chi2(
        data.vis, model, data.weights, covariance
    )
    likelihood = float(density.log_likelihood(z))
    prior = float(density.log_prior(z))
    assert likelihood == pytest.approx(float(expected))
    assert np.isfinite(prior)
    assert float(density.log_posterior(z)) == pytest.approx(likelihood + prior)
