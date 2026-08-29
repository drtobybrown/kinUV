"""S2 posterior helpers (no npz, no FINUFFT)."""

from __future__ import annotations

import numpy as np
import pytest

from kinuv.infer.posterior import (
    SAMPLER_NAME,
    PARAM_NAMES,
    Z68,
    Z95,
    ess_bulk,
    fd_hessian,
    gaussian_interval,
    in_interval,
    laplace_cov,
    mh_sample,
    n_vis_of,
    split_rhat,
    t_dof,
    t_nvis,
)


def test_sampler_name_is_laplace_mh():
    assert SAMPLER_NAME == "laplace_mh"
    assert SAMPLER_NAME != "NUTS"
    assert len(PARAM_NAMES) == 8


def test_t_dof_matches_official_chi2():
    n_vis = 881 * 95
    chi2_map = 168675.59555208942
    assert t_nvis(chi2_map, n_vis) == pytest.approx(2.016, rel=1e-3)
    assert t_dof(chi2_map, n_vis) == pytest.approx(1.008, rel=1e-3)
    assert t_dof(2.0 * n_vis, n_vis) == pytest.approx(1.0)


def test_n_vis_of_shape():
    class _D:
        vis = np.zeros((4, 5))

    assert n_vis_of(_D()) == 20


def test_gaussian_interval_and_hit():
    lo, hi = gaussian_interval(0.0, 1.0, Z68)
    assert lo == pytest.approx(-Z68)
    assert hi == pytest.approx(Z68)
    assert in_interval(0.0, lo, hi)
    assert not in_interval(3.0, lo, hi)
    lo95, hi95 = gaussian_interval(10.0, 4.0, Z95)
    assert lo95 == pytest.approx(10.0 - 2 * Z95)
    assert hi95 == pytest.approx(10.0 + 2 * Z95)


def test_fd_hessian_quadratic():
    # f(x) = 3 x0^2 + 5 x1^2 + 2 x0 x1
    def fun(x):
        return 3.0 * x[0] ** 2 + 5.0 * x[1] ** 2 + 2.0 * x[0] * x[1]

    h = fd_hessian(fun, np.array([0.3, -0.2]), step=1e-4)
    assert h[0, 0] == pytest.approx(6.0, rel=1e-4)
    assert h[1, 1] == pytest.approx(10.0, rel=1e-4)
    assert h[0, 1] == pytest.approx(2.0, rel=1e-4)


def test_laplace_cov_and_rhat_ess():
    hess = np.diag([4.0, 16.0])
    cov = laplace_cov(hess, t=1.0)
    # cov = 2 * inv(H)
    assert cov[0, 0] == pytest.approx(0.5)
    assert cov[1, 1] == pytest.approx(0.125)
    cov2 = laplace_cov(hess, t=4.0)
    assert cov2[0, 0] == pytest.approx(2.0)

    rng = np.random.default_rng(2)
    # 4 well-mixed N(0,1) chains
    chains = rng.standard_normal((4, 400, 2))
    rh = split_rhat(chains)
    assert rh.shape == (2,)
    assert np.all(rh < 1.05)
    ess = ess_bulk(chains)
    assert ess.shape == (2,)
    assert np.all(ess > 200)


def test_mh_independence_recovers_1d_gaussian():
    rng = np.random.default_rng(7)

    def logp(x):
        return -0.5 * float(x[0] ** 2)

    rec = mh_sample(
        logp,
        np.array([0.0]),
        np.array([[1.0]]),
        n_chain=2,
        n_warmup=80,
        n_draw=400,
        rng=rng,
    )
    assert rec.sampler == "laplace_mh"
    flat = rec.samples.reshape(-1)
    assert rec.accept > 0.2
    assert abs(float(np.mean(flat))) < 0.25
