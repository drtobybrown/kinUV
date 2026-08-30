"""G1 CPU JAX predict_binned. Skip unless jax-finufft."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.infer.map import predict_binned
from kinuv.io.vis import DEFAULT_NPZ, VisData
from kinuv.likelihood.chi2 import chi2
from kinuv.transforms.grid import ImageGrid, image_grid_from_uv, max_baseline_lambda
from kinuv.transforms.nufft import BACKEND, nufft_backend
from kinuv.xp import is_jax

REPO = Path(__file__).resolve().parents[1]
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
SRC_MAP = REPO / "src/kinuv/infer/map.py"


def _require_jax_finufft():
    pytest.importorskip("jax")
    if BACKEND != "jax-finufft":
        pytest.skip("G1 XLA tests require jax-finufft")
    assert nufft_backend() == "jax-finufft"


def _tiny_data():
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


def _tiny_template(grid: ImageGrid):
    x = (np.arange(grid.nx) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny) - grid.ny // 2) * grid.cell_arcsec
    xe, yn = np.meshgrid(x, y, indexing="xy")
    r2 = xe**2 + yn**2
    img = np.exp(-0.5 * r2 / 4.0)
    return img / img.sum()


def _tiny_params():
    return {
        "flux": 2.0,
        "pa_deg": 200.0,
        "vsys_kms": 8300.0,
        "gas_sigma_kms": 12.0,
        "dx_arcsec": 0.1,
        "dy_arcsec": -0.05,
        "v0_kms": 200.0,
        "r_t_arcsec": 2.0,
    }


def test_xla_path_calls_spectral_hann_then_bin():
    text = SRC_MAP.read_text()
    assert "from kinuv.response.spectral import hann_then_bin" in text
    assert "native_diagonal" not in text
    assert "xla=False" in text


def test_tiny_numpy_vs_jax_vis_and_grad():
    _require_jax_finufft()
    import jax
    import jax.numpy as jnp

    data = _tiny_data()
    mb = max_baseline_lambda(data.u_m, data.v_m, data.freqs_native)
    grid = image_grid_from_uv(mb, 8.0)
    tmpl = _tiny_template(grid)
    params = _tiny_params()
    vis_np = predict_binned(data, params, tmpl, grid, xla=False)
    vis_jax = predict_binned(data, params, tmpl, grid, xla=True)
    assert is_jax(vis_jax)
    err = float(np.max(np.abs(np.asarray(vis_jax) - vis_np)))
    assert err < 1e-8
    c_np = chi2(data.vis, vis_np, data.weights, data.s)
    c_j = chi2(data.vis, vis_jax, jnp.asarray(data.weights), data.s)
    assert is_jax(c_j)
    assert abs(float(c_j) - float(c_np)) < 1e-8

    def loss(flux):
        p = dict(params)
        p["flux"] = flux
        model = predict_binned(data, p, tmpl, grid, xla=True)
        return chi2(data.vis, model, jnp.asarray(data.weights), data.s)

    g = float(jax.grad(loss)(params["flux"]))
    assert np.isfinite(g)
    fd = (
        float(loss(params["flux"] + 1.0e-3)) - float(loss(params["flux"] - 1.0e-3))
    ) / 2.0e-3
    assert g == pytest.approx(fd, rel=1e-3, abs=1e-4)


def test_official_066_chi2_identity():
    _require_jax_finufft()
    npz = CANFAR_NPZ if CANFAR_NPZ.is_file() else DEFAULT_NPZ
    ico = CANFAR_ICO if CANFAR_ICO.is_file() else None
    stage = MAP_DIR / "stage_a_map.json"
    if not npz.is_file() or not stage.is_file():
        pytest.skip("official 066 npz or MAP json missing")
    from kinuv.forward.sb import load_sb_template
    from kinuv.infer.map import image_grid_for_vis
    from kinuv.io.vis import load_kgas066

    rec = json.loads(stage.read_text())
    cube30 = Path(
        "/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/30kms/"
        "KGAS66_clipped_cube.fits"
    )
    data = load_kgas066(npz, cube_path=cube30 if cube30.is_file() else None)
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ico)
    params = {
        "flux": rec["flux"],
        "pa_deg": rec["pa_deg"],
        "vsys_kms": rec["vsys_kms"],
        "gas_sigma_kms": rec["gas_sigma_kms"],
        "dx_arcsec": rec["dx_arcsec"],
        "dy_arcsec": rec["dy_arcsec"],
        "v0_kms": rec["v0_kms"],
        "r_t_arcsec": rec["r_t_arcsec"],
    }
    vis_np = predict_binned(data, params, tmpl, grid, xla=False)
    vis_j = predict_binned(data, params, tmpl, grid, xla=True)
    assert abs(float(chi2(data.vis, vis_np, data.weights, data.s)) - 168675.6) < 1.0
    assert float(np.max(np.abs(np.asarray(vis_j) - vis_np))) < 1e-6
    c_j = chi2(data.vis, vis_j, data.weights, data.s)
    assert abs(float(c_j) - 168675.6) < 1.0
    assert abs(float(data.s) - 0.5136098555284736) < 1e-6
