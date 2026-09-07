import numpy as np

from kinuv.infer.collaborator import campaign_transform
from kinuv.infer.s3 import chart_bounds


def test_campaign_transform_is_bounded_and_preserves_physical_measure():
    bounds, active = chart_bounds(200.0, 10.0, 1.0, "two_zone_dispersion", two_zone_uses_rings=False)
    transform = campaign_transform(bounds, active, uses_rings=False)
    full = np.array([(lo + hi) / 2.0 for lo, hi in bounds])
    full[7] = 0.3
    full[8] = 2.0
    y = transform.active_initial_unconstrained(full)
    recovered, log_jac = transform.unconstrained_to_full_jax(y, full)
    np.testing.assert_allclose(np.asarray(recovered), full, rtol=0.0, atol=1e-10)
    assert np.isfinite(float(log_jac))
    assert {0, 3, 7, 8, 15}.issubset(transform.physical_log_indices)


def test_ring_transform_excludes_frozen_emissivity():
    bounds, active = chart_bounds(150.0, 10.0, 1.2, "supported_rings")
    transform = campaign_transform(bounds, active, uses_rings=True)
    assert 13 not in transform.active and 14 not in transform.active
    assert set(range(9, 13)).issubset(transform.log_kinematic_indices)
