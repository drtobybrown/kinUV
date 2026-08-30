"""G3 autodiff potential + CPU NUTS. Skip NUFFT tests unless jax-finufft."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.constants import velocity_to_freq_hz
from kinuv.infer.chart import (
    PARAM_NAMES,
    log_prob_unconstrained,
    params_to_unconstrained,
    unconstrained_to_physical,
)
from kinuv.infer.nuts import (
    DX_IDX,
    DY_IDX,
    FROZEN_NAMES,
    NUTS_SAMPLER,
    SAMPLED_IDX,
    SAMPLED_NAMES,
    make_potential,
    mixing_sampled,
    physical_sampled_from_z6,
    potential_unconstrained,
    sampled_z_from_physical,
    stitch_draws_8col,
    stitch_z8,
)
from kinuv.infer.posterior import SAMPLER_NAME, params_to_vec
from kinuv.io.vis import VisData
from kinuv.transforms.grid import ImageGrid, image_grid_from_uv, max_baseline_lambda
from kinuv.transforms.nufft import BACKEND, nufft_backend
from kinuv.xp import is_jax

REPO = Path(__file__).resolve().parents[1]
SRC_CHART = REPO / "src/kinuv/infer/chart.py"
SRC_NUTS = REPO / "src/kinuv/infer/nuts.py"
SRC_ROT = REPO / "src/kinuv/profiles/rotation.py"
SRC_MODEL = REPO / "src/kinuv/forward/model.py"
SRC_MAP = REPO / "src/kinuv/infer/map.py"
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
FD_H = 1.0e-5


def _require_jax_finufft():
    pytest.importorskip("jax")
    if BACKEND != "jax-finufft":
        pytest.skip("G3 autodiff tests require jax-finufft")
    assert nufft_backend() == "jax-finufft"


def _require_numpyro():
    pytest.importorskip("numpyro")


def _tiny_data():
    rng = np.random.default_rng(7)
    n_row, n_native, n_guard, n_bin = 12, 16, 1, 2
    u = rng.uniform(8.0, 80.0, n_row)
    v = rng.uniform(-60.0, 60.0, n_row)
    vsys = 8300.0
    dv_nat = 8.0
    vel_n = vsys + (np.arange(n_native) - n_native // 2) * dv_nat
    freqs_n = velocity_to_freq_hz(vel_n)
    n_trim = n_native - 2 * n_guard
    n_fit = n_trim // n_bin
    vis = rng.normal(size=(n_row, n_fit)) + 1j * rng.normal(size=(n_row, n_fit))
    w = np.ones((n_row, n_fit), dtype=np.float64)
    w_n = np.ones((n_row, n_trim), dtype=np.float64)
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


def _tiny_template(grid: ImageGrid):
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.cell_arcsec
    xe, yn = np.meshgrid(x, y, indexing="xy")
    r2 = xe**2 + yn**2
    img = np.exp(-0.5 * r2 / 4.0)
    return img / img.sum()


def _setup_tiny():
    data = _tiny_data()
    mb = max_baseline_lambda(data.u_m, data.v_m, data.freqs_native)
    grid = image_grid_from_uv(mb, 8.0)
    tmpl = _tiny_template(grid)
    return data, tmpl, grid


def _tiny_inject():
    from dataclasses import replace

    from kinuv.infer.map import predict_binned

    data, tmpl, grid = _setup_tiny()
    dx, dy = 0.1, -0.05
    params = {
        "flux": 2.0,
        "pa_deg": 200.0,
        "vsys_kms": 8300.0,
        "gas_sigma_kms": 12.0,
        "dx_arcsec": dx,
        "dy_arcsec": dy,
        "v0_kms": 200.0,
        "r_t_arcsec": 2.0,
    }
    model = np.asarray(predict_binned(data, params, tmpl, grid, xla=True))
    rms = float(np.sqrt(np.mean(np.abs(model) ** 2)))
    sig = 1.0e-3 * max(rms, 1e-6)
    rng = np.random.default_rng(11)
    vis = model + sig * (
        rng.standard_normal(model.shape) + 1j * rng.standard_normal(model.shape)
    )
    w = np.full_like(data.weights, 1.0e4)
    w_n = np.full_like(data.weights_native, 1.0e4)
    return replace(data, vis=vis, weights=w, weights_native=w_n), tmpl, grid, params


def test_chart_and_nuts_source_gates():
    chart = SRC_CHART.read_text()
    nuts = SRC_NUTS.read_text()
    assert "logit" not in chart
    assert "RT_BOUNDS" not in chart
    assert "numpyro" not in chart.lower()
    assert "def log_prob_unconstrained" not in nuts
    assert "unconstrained_to_params" not in nuts
    assert "chi2_and_prior" not in nuts
    assert SAMPLER_NAME == "laplace_mh"
    assert NUTS_SAMPLER == "nuts"
    assert SAMPLED_NAMES == (
        "flux",
        "pa_deg",
        "vsys_kms",
        "gas_sigma_kms",
        "v0_kms",
        "r_t_arcsec",
    )
    assert FROZEN_NAMES == ("dx_arcsec", "dy_arcsec")
    arctan = SRC_ROT.read_text().split("def rings_from_arctan", 1)[0]
    assert "def arctan_vc" in arctan
    assert "float(" not in arctan.split("def arctan_vc", 1)[1]
    los = SRC_MODEL.read_text().split("def _gaussian_pdf", 1)[0]
    assert "float(vsys" not in los
    gauss = SRC_MODEL.read_text().split("def _gaussian_pdf", 1)[1].split(
        "def sky_cube", 1
    )[0]
    assert "float(" not in gauss
    assert 'pa_rad = xp.asarray(params["pa_deg"]) * (np.pi / 180.0)' in SRC_MAP.read_text()
    u_src = nuts.split("def make_potential", 1)[1].split("return U", 1)[0]
    assert "0.5 * (c + prior)" in u_src
    assert "2.0 * U" not in u_src and "2 * U" not in u_src


def test_plot_posterior_corner_still_refuses_laplace_mh():
    from kinuv.diagnostics.figures import plot_posterior_corner

    with pytest.raises(ValueError, match="nuts"):
        plot_posterior_corner({"sampler": "laplace_mh", "draws": np.zeros((4, 8))}, Path("/tmp/x.png"))


def test_stitch_8col_freeze_constant():
    rng = np.random.default_rng(1)
    d6 = rng.normal(size=(2, 5, 6))
    dx, dy = OFFICIAL["dx_arcsec"], OFFICIAL["dy_arcsec"]
    d8 = stitch_draws_8col(d6, dx, dy)
    assert d8.shape == (2, 5, 8)
    assert np.allclose(d8[..., DX_IDX], dx)
    assert np.allclose(d8[..., DY_IDX], dy)
    mix = mixing_sampled(d8)
    assert set(mix) == set(SAMPLED_NAMES)
    assert "dx_arcsec" not in mix
    assert "ess_tail" in next(iter(mix.values()))


def test_jax_u_matches_host_log_prob_unconstrained():
    _require_jax_finufft()
    import jax
    import jax.numpy as jnp

    data, tmpl, grid = _setup_tiny()
    dx, dy = OFFICIAL["dx_arcsec"], OFFICIAL["dy_arcsec"]
    z8 = params_to_unconstrained(OFFICIAL)
    z6 = z8[list(SAMPLED_IDX)]
    U = make_potential(data, tmpl, grid, dx, dy)
    u_jit = jax.jit(U)
    u_val = u_jit(jnp.asarray(z6))
    assert is_jax(u_val)
    host = float(log_prob_unconstrained(data, z8, tmpl, grid))
    assert float(u_val) + host == pytest.approx(0.0, abs=1e-6)
    u_eager = potential_unconstrained(jnp.asarray(z6), data, tmpl, grid, dx, dy)
    assert float(u_eager) == pytest.approx(float(u_val), abs=1e-6)


def test_numpyro_pin_keeps_jax_finufft():
    _require_jax_finufft()
    _require_numpyro()
    import jax
    import numpyro

    assert jax.__version__.startswith("0.11")
    assert numpyro.__version__.startswith("0.21")
    assert BACKEND == "jax-finufft"


def test_jax_grad_u_six_sampled_vs_fd():
    _require_jax_finufft()
    import jax
    import jax.numpy as jnp

    data, tmpl, grid = _setup_tiny()
    dx, dy = OFFICIAL["dx_arcsec"], OFFICIAL["dy_arcsec"]
    z8 = params_to_unconstrained(OFFICIAL)
    z6 = np.asarray(z8[list(SAMPLED_IDX)], dtype=np.float64)
    U = make_potential(data, tmpl, grid, dx, dy)
    g = np.asarray(jax.grad(U)(jnp.asarray(z6)), dtype=np.float64)
    assert g.shape == (6,)
    assert np.all(np.isfinite(g))
    fd = np.empty(6, dtype=np.float64)
    for i in range(6):
        zp = z6.copy()
        zm = z6.copy()
        zp[i] += FD_H
        zm[i] -= FD_H
        fd[i] = (float(U(jnp.asarray(zp))) - float(U(jnp.asarray(zm)))) / (2.0 * FD_H)
    np.testing.assert_allclose(g, fd, rtol=1e-3, atol=1e-4)


def test_official_chi2_after_chart_xla():
    _require_jax_finufft()
    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import image_grid_for_vis, predict_binned
    from kinuv.io.vis import DEFAULT_NPZ, load_kgas066
    from kinuv.likelihood.chi2 import chi2

    npz = CANFAR_NPZ if CANFAR_NPZ.is_file() else DEFAULT_NPZ
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    stage = MAP_DIR / "stage_a_map.json"
    if not npz.is_file() or not stage.is_file():
        pytest.skip("official 066 npz or MAP json missing")
    rec = json.loads(stage.read_text())
    cube30 = Path(
        "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
        "KGAS66_clipped_cube.fits"
    )
    data = load_kgas066(npz, cube_path=cube30 if cube30.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)
    params = {n: rec[n] for n in PARAM_NAMES}
    z8 = params_to_unconstrained(params)
    back = {n: float(v) for n, v in zip(PARAM_NAMES, np.asarray(unconstrained_to_physical(z8)))}
    vis = predict_binned(data, back, tmpl, grid, xla=True)
    c = chi2(data.vis, vis, data.weights, data.s)
    assert abs(float(c) - 168675.6) < 1.0
    assert abs(float(data.s) - 0.5136098555284736) < 1e-6


def test_tiny_mock_nuts_mixing():
    _require_jax_finufft()
    _require_numpyro()
    import jax
    import jax.numpy as jnp

    from kinuv.infer.nuts import mixing_ok, run_nuts_z6

    data, tmpl, grid, params = _tiny_inject()
    dx, dy = params["dx_arcsec"], params["dy_arcsec"]
    z6 = sampled_z_from_physical(params_to_vec(params))
    U = make_potential(data, tmpl, grid, dx, dy)
    u_jit = jax.jit(U)
    _ = float(u_jit(jnp.asarray(z6)))
    z_draws, mean_steps, _ = run_nuts_z6(
        u_jit, z6, rng_seed=3, num_warmup=32, num_samples=32, num_chains=4, jitter=0.05
    )
    assert z_draws.shape == (4, 32, 6)
    assert np.all(np.isfinite(z_draws))
    phys8 = physical_sampled_from_z6(z_draws, dx, dy)
    assert phys8.shape == (4, 32, 8)
    assert np.all(np.isfinite(phys8))
    mix = mixing_sampled(phys8)
    # 32-draw smoke: finite mixing stats. Card R_hat<1.01 / ESS>200 is the artifact run.
    assert mixing_ok(mix, rhat_max=1.2, ess_min=15.0)
    assert np.all(np.isfinite([v["rhat"] for v in mix.values()]))
    assert np.isfinite(mean_steps)
    from kinuv.diagnostics.figures import plot_posterior_corner

    rec = {
        "sampler": NUTS_SAMPLER,
        "draws": phys8,
    }
    out = Path("/tmp/g3_tiny_corner.png")
    plot_posterior_corner(rec, out)
    assert out.is_file()
    with pytest.raises(ValueError, match="8"):
        plot_posterior_corner(
            {"sampler": "nuts", "draws": np.zeros((4, 6))}, Path("/tmp/g3_six.png")
        )
