from __future__ import annotations

import numpy as np
import pytest

from kinuv.infer.s3 import (
    build_positive_emissivity_basis,
    chart_bounds,
    initial_chart,
    supported_knot_radii,
    unpack_chart,
)
from kinuv.transforms.grid import ImageGrid


def _template(grid):
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    east, north = np.meshgrid(x, x, indexing="xy")
    image = np.exp(-np.hypot(east, north) / 1.2)
    image[0, 0] = -1.0e-4
    image /= image.sum() * grid.cell_arcsec**2
    return image


def test_positive_emissivity_basis_is_normalized_and_reconstructs_clipped_template():
    grid = ImageGrid(nx=31, ny=31, cell_arcsec=0.2)
    template = _template(grid)
    basis = build_positive_emissivity_basis(template, grid, 0.4, 0.7)
    assert basis.images.shape == (3, grid.ny, grid.nx)
    assert np.all(basis.images >= 0.0)
    np.testing.assert_allclose(
        basis.images.sum(axis=(1, 2)) * grid.cell_arcsec**2,
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert basis.natural_weights.sum() == pytest.approx(1.0)
    reconstructed = np.sum(
        basis.natural_weights[:, None, None] * basis.images, axis=0
    )
    clipped = np.maximum(template, 0.0)
    clipped /= clipped.sum() * grid.cell_arcsec**2
    np.testing.assert_allclose(reconstructed, clipped, rtol=0.0, atol=1.0e-14)


def test_supported_knots_are_beam_aware_and_inside_emission_support():
    grid = ImageGrid(nx=41, ny=41, cell_arcsec=0.2)
    knots, r95 = supported_knot_radii(_template(grid), grid, 0.2, 0.6, 1.0)
    assert knots.shape == (4,)
    assert knots[0] >= 0.5
    assert np.all(np.diff(knots) >= 0.2 - 1.0e-12)
    assert knots[-1] <= r95 + 0.5


def test_s3_chart_round_trip_and_candidate_activity():
    s2 = {
        "flux": 8.0,
        "pa_deg": 201.0,
        "vsys_kms": 8098.0,
        "gas_sigma_kms": 11.0,
        "dx_arcsec": 0.1,
        "dy_arcsec": -0.2,
        "u_kms": 180.0,
        "inclination_deg": 45.0,
        "r_t_arcsec": 0.4,
    }
    radii = np.array([0.5, 1.0, 2.0, 3.0])
    natural = np.array([0.2, 0.3, 0.5])
    z = initial_chart(
        s2, radii, natural, vsys_seed_kms=8090.0, dv_kms=5.0, bmaj_arcsec=1.0
    )
    back = unpack_chart(
        z,
        vsys_seed_kms=8090.0,
        dv_kms=5.0,
        bmaj_arcsec=1.0,
        natural_weights=natural,
    )
    assert back["flux"] == pytest.approx(8.0)
    assert back["arctan_u_kms"] == pytest.approx(180.0)
    assert back["sigma_inner_kms"] == pytest.approx(11.0)
    assert back["sigma_outer_kms"] == pytest.approx(11.0)
    np.testing.assert_allclose(back["emissivity_weights"], natural)
    _, active_baseline = chart_bounds(200.0, 5.0, 1.0, "baseline_arctan")
    _, active_rings = chart_bounds(200.0, 5.0, 1.0, "supported_rings")
    _, active_dispersion = chart_bounds(200.0, 5.0, 1.0, "two_zone_dispersion")
    assert active_baseline == tuple(range(9))
    assert 7 not in active_rings and 8 not in active_rings
    assert set(range(9, 15)).issubset(active_rings)
    assert 15 in active_dispersion
