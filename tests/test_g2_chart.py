"""G2 unconstrained Stage A chart. JAX tests skip on missing jax, not jax-finufft."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.infer.chart import (
    PARAM_NAMES,
    inv_softplus,
    log_abs_det_jacobian,
    log_abs_det_terms,
    log_prob_unconstrained,
    params_to_unconstrained,
    physical_to_unconstrained,
    softplus,
    unconstrained_to_params,
    unconstrained_to_physical,
)
from kinuv.infer.posterior import SAMPLER_NAME, log_prob, params_to_vec
from kinuv.xp import is_jax

REPO = Path(__file__).resolve().parents[1]
SRC_CHART = REPO / "src/kinuv/infer/chart.py"
MAP_DIR = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/"
    "kinuv-KGAS066-uvsign-map"
)
CANFAR_NPZ = Path(
    "/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz"
)
CANFAR_ICO = Path(
    "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
    "KGAS66_Ico_K_kms-1.fits"
)

OFFICIAL = {
    "flux": 70.45946914656797,
    "pa_deg": 199.72980072503037,
    "vsys_kms": 8098.773150512066,
    "gas_sigma_kms": 12.050032182953927,
    "dx_arcsec": 0.09104737371760792,
    "dy_arcsec": 0.018566961155444102,
    "v0_kms": 267.6703121989014,
    "r_t_arcsec": 0.5,
}

_I_FLUX, _I_PA, _I_GS, _I_DX, _I_V0, _I_RT = 0, 1, 3, 4, 6, 7
V0_ROUNDTRIP = (1.0e-4, 1.0e-3, 1.0, 20.0, 21.0, 267.6703121989014, 1.0e3)
FD_H = 1.0e-5


def _require_jax():
    pytest.importorskip("jax")


def _tiny_data():
    from kinuv.io.vis import VisData

    rng = np.random.default_rng(7)
    n_row, n_native, n_guard, n_bin = 6, 10, 1, 2
    u = rng.uniform(8.0, 40.0, n_row)
    v = rng.uniform(-20.0, 20.0, n_row)
    freqs_n = 230.538e9 + np.arange(n_native) * 7.8e6
    n_trim = n_native - 2 * n_guard
    n_fit = n_trim // n_bin
    vis = rng.normal(size=(n_row, n_fit)) + 1j * rng.normal(size=(n_row, n_fit))
    w = np.ones((n_row, n_fit), dtype=np.float64)
    w_n = np.ones((n_row, n_trim), dtype=np.float64)
    vel_n = np.linspace(8200.0, 8400.0, n_native)
    vel = vel_n[n_guard:-n_guard].reshape(n_fit, n_bin).mean(axis=1)
    freqs = freqs_n[n_guard:-n_guard].reshape(n_fit, n_bin).mean(axis=1)
    return VisData(
        u_m=u,
        v_m=v,
        vis=vis,
        weights=w,
        freqs=freqs,
        vel=vel,
        freqs_native=freqs_n,
        vel_native=vel_n,
        n_bin=n_bin,
        dv_kms=float(np.median(np.abs(np.diff(vel)))),
        s=0.5136098555284736,
        phase_dir_rad=np.zeros(2),
        line_free_mask=np.ones(n_fit, dtype=bool),
        n_guard=n_guard,
        weights_native=w_n,
    )


def _tiny_template(grid):
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.cell_arcsec
    xe, yn = np.meshgrid(x, y, indexing="xy")
    r2 = xe**2 + yn**2
    img = np.exp(-0.5 * r2 / 4.0)
    return img / img.sum()


def _fd_dtheta_dzi(z, i, h=FD_H):
    zp = np.array(z, dtype=np.float64)
    zm = zp.copy()
    zp[i] += h
    zm[i] -= h
    tp = np.asarray(unconstrained_to_physical(zp), dtype=np.float64)
    tm = np.asarray(unconstrained_to_physical(zm), dtype=np.float64)
    return (tp[i] - tm[i]) / (2.0 * h)


def test_chart_source_has_no_box_chart_and_no_numpyro():
    text = SRC_CHART.read_text()
    assert "logit" not in text
    assert "RT_BOUNDS" not in text
    assert "GAS_SIGMA_BOUNDS" not in text
    assert "FLUX_BOUNDS" not in text
    assert "numpyro" not in text.lower()
    assert "sampler: nuts" not in text
    assert SAMPLER_NAME == "laplace_mh"


def test_param_names_are_the_live_eight():
    from kinuv.infer.map import PARAM_NAMES as MAP_NAMES
    from kinuv.infer.posterior import PARAM_NAMES as POST_NAMES

    assert PARAM_NAMES == POST_NAMES == MAP_NAMES
    assert PARAM_NAMES == (
        "flux",
        "pa_deg",
        "vsys_kms",
        "gas_sigma_kms",
        "dx_arcsec",
        "dy_arcsec",
        "v0_kms",
        "r_t_arcsec",
    )


def test_official_rt_maps_to_finite_z_and_roundtrips():
    z = params_to_unconstrained(OFFICIAL)
    assert np.all(np.isfinite(z))
    assert z[_I_RT] == pytest.approx(np.log(0.5), rel=0, abs=1e-15)
    back = unconstrained_to_params(z)
    for name in PARAM_NAMES:
        assert back[name] == pytest.approx(OFFICIAL[name], rel=1e-12, abs=1e-12)


def test_v0_zero_is_negative_infinity():
    theta = params_to_vec(OFFICIAL)
    theta[_I_V0] = 0.0
    z = physical_to_unconstrained(theta)
    assert z[_I_V0] == -np.inf


@pytest.mark.parametrize("v0", V0_ROUNDTRIP)
def test_v0_softplus_roundtrip_numpy(v0):
    theta = params_to_vec(OFFICIAL)
    theta[_I_V0] = float(v0)
    z = physical_to_unconstrained(theta)
    assert np.isfinite(z[_I_V0])
    back = np.asarray(unconstrained_to_physical(z), dtype=np.float64)
    rel = abs(back[_I_V0] - v0) / v0
    assert rel < 1.0e-10


def test_jitted_softplus_finite_at_1000_and_v0_roundtrip():
    _require_jax()
    import jax
    import jax.numpy as jnp

    sp = jax.jit(softplus)
    out = sp(jnp.asarray(1000.0))
    assert is_jax(out)
    assert np.isfinite(float(out))

    inv = jax.jit(inv_softplus)
    fwd = jax.jit(softplus)

    def roundtrip(v):
        return fwd(inv(v))

    rt = jax.jit(roundtrip)
    vals = jnp.asarray(V0_ROUNDTRIP, dtype=jnp.float64)
    back = rt(vals)
    assert is_jax(back)
    rel = np.abs(np.asarray(back) - np.asarray(V0_ROUNDTRIP)) / np.asarray(
        V0_ROUNDTRIP
    )
    assert np.max(rel) < 1.0e-10

    g_sp = jax.jit(jax.grad(lambda x: jnp.sum(softplus(x))))
    g_inv = jax.jit(jax.grad(lambda y: jnp.sum(inv_softplus(y))))
    probe = jnp.asarray([1.0e-4, 21.0, 1.0e3], dtype=jnp.float64)
    gs = g_sp(probe)
    gi = g_inv(probe)
    assert is_jax(gs) and is_jax(gi)
    assert np.all(np.isfinite(np.asarray(gs)))
    assert np.all(np.isfinite(np.asarray(gi)))


def test_per_axis_jacobian_fd_signed():
    z = params_to_unconstrained(OFFICIAL)
    # r_t log at 0.5: ln|J| = z < 0
    assert z[_I_RT] < 0.0
    terms = np.asarray(log_abs_det_terms(z), dtype=np.float64)
    assert terms[_I_RT] == pytest.approx(z[_I_RT])
    assert terms[_I_FLUX] == pytest.approx(z[_I_FLUX])
    assert terms[_I_GS] == pytest.approx(z[_I_GS])
    assert terms[_I_PA] == pytest.approx(0.0)
    assert terms[_I_DX] == pytest.approx(0.0)

    for i, name in enumerate(PARAM_NAMES):
        if name in ("flux", "gas_sigma_kms", "r_t_arcsec"):
            fd = _fd_dtheta_dzi(z, i, h=1.0e-5)
            analytic_d = float(np.exp(z[i]))
            assert fd == pytest.approx(analytic_d, rel=1e-6, abs=1e-8)
            ln_j = np.log(abs(fd))
            assert ln_j == pytest.approx(z[i], rel=1e-6, abs=1e-8)
        elif name == "v0_kms":
            # official V0 is on the identity softplus arm; σ≈1, ln|J|≈0
            assert abs(terms[i]) < 1e-8
        else:
            fd = _fd_dtheta_dzi(z, i, h=1.0e-4)
            assert abs(fd - 1.0) < 1e-8, name
            assert terms[i] == pytest.approx(0.0, abs=1e-15)

    z_v0 = z.copy()
    z_v0[_I_V0] = 0.0
    fd0 = _fd_dtheta_dzi(z_v0, _I_V0)
    terms0 = np.asarray(log_abs_det_terms(z_v0), dtype=np.float64)
    assert fd0 == pytest.approx(0.5, rel=1e-6, abs=1e-8)
    assert terms0[_I_V0] == pytest.approx(-np.log(2.0), rel=1e-12, abs=1e-12)

    total = float(log_abs_det_jacobian(z))
    assert total == pytest.approx(float(np.sum(terms)), rel=0, abs=1e-14)


def test_jit_type_preservation_eight_vector():
    _require_jax()
    import jax
    import jax.numpy as jnp

    z = jnp.ones(8, dtype=jnp.float64)
    theta = jax.jit(unconstrained_to_physical)(z)
    jac = jax.jit(log_abs_det_jacobian)(z)
    z2 = jax.jit(physical_to_unconstrained)(theta)
    assert is_jax(theta)
    assert is_jax(jac)
    assert is_jax(z2)
    assert theta.shape == (8,)


def test_log_prob_unconstrained_is_host_sum():
    from kinuv.transforms.grid import image_grid_from_uv, max_baseline_lambda

    data = _tiny_data()
    mb = max_baseline_lambda(data.u_m, data.v_m, data.freqs_native)
    grid = image_grid_from_uv(mb, 8.0)
    tmpl = _tiny_template(grid)
    z = params_to_unconstrained(OFFICIAL)
    params = unconstrained_to_params(z)
    expected = float(log_prob(data, params, tmpl, grid)) + float(
        log_abs_det_jacobian(z)
    )
    got = log_prob_unconstrained(data, z, tmpl, grid)
    assert got == pytest.approx(expected, rel=0, abs=1e-9)


def test_official_chi2_after_roundtrip():
    from kinuv.io.vis import DEFAULT_NPZ

    npz = CANFAR_NPZ if CANFAR_NPZ.is_file() else DEFAULT_NPZ
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    stage = MAP_DIR / "stage_a_map.json"
    if not npz.is_file() or not stage.is_file():
        pytest.skip("official 066 npz or MAP json missing")
    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import image_grid_for_vis, predict_binned
    from kinuv.io.vis import load_kgas066
    from kinuv.likelihood.chi2 import chi2

    rec = json.loads(stage.read_text())
    cube30 = Path(
        "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
        "KGAS66_clipped_cube.fits"
    )
    data = load_kgas066(npz, cube_path=cube30 if cube30.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)
    params = {n: rec[n] for n in PARAM_NAMES}
    z = params_to_unconstrained(params)
    assert np.isfinite(z[_I_RT])
    back = unconstrained_to_params(z)
    vis = predict_binned(data, back, tmpl, grid, xla=False)
    c = chi2(data.vis, vis, data.weights, data.s)
    assert abs(float(c) - 168675.6) < 1.0
    assert abs(float(data.s) - 0.5136098555284736) < 1e-6


def test_jax_grad_of_chart_maps():
    """Chart autodiff only. ``predict_binned`` still host-converts; that is G3."""
    _require_jax()
    import jax
    import jax.numpy as jnp

    z0 = jnp.asarray(params_to_unconstrained(OFFICIAL))
    g_theta = jax.grad(lambda z: unconstrained_to_physical(z).sum())(z0)
    g_jac = jax.grad(log_abs_det_jacobian)(z0)
    assert is_jax(g_theta) and is_jax(g_jac)
    assert np.all(np.isfinite(np.asarray(g_theta)))
    assert np.all(np.isfinite(np.asarray(g_jac)))
