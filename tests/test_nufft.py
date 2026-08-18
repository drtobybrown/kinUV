"""FINUFFT T2 vs 066-1 DFT on a Gaussian; Nyquist vs 305 kλ (DEC-066-GRID)."""

import time
from pathlib import Path

import numpy as np
import pytest

from kinuv.transforms.dft import dft_numpy
from kinuv.transforms.grid import (
    KGAS066_MAX_BASELINE_LAMBDA,
    cell_arcsec_from_max_baseline,
    fov_co_plus_pb_arcsec,
    image_grid_from_uv,
    max_baseline_lambda,
    nyquist_assert,
    nyquist_u_max_lambda,
)
from kinuv.transforms.nufft import BACKEND, nufft2_degrid, nufft3_degrid, nufft_backend

NPZ = Path("/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz")


def gaussian_on_grid(grid, sx_arcsec, sy_arcsec, flux):
    L, M = grid.pixel_lm_rad()
    from kinuv.constants import ARCSEC_TO_RAD

    img = np.exp(
        -0.5
        * (
            (L / (sx_arcsec * ARCSEC_TO_RAD)) ** 2
            + (M / (sy_arcsec * ARCSEC_TO_RAD)) ** 2
        )
    )
    return img * (flux / img.sum())


def test_backend_is_finufft_not_dft():
    assert nufft_backend() in {"jax-finufft", "finufft"}
    assert "dft_numpy" not in nufft2_degrid.__code__.co_names


def test_nyquist_assert_fires_for_0p4_vs_305klambda():
    with pytest.raises(ValueError, match="Nyquist"):
        nyquist_assert(0.4, KGAS066_MAX_BASELINE_LAMBDA)
    assert nyquist_u_max_lambda(0.4) < KGAS066_MAX_BASELINE_LAMBDA


def test_nyquist_assert_passes_legal_cell():
    nyquist_assert(0.1, KGAS066_MAX_BASELINE_LAMBDA)
    cell = cell_arcsec_from_max_baseline(KGAS066_MAX_BASELINE_LAMBDA)
    assert cell < 0.4
    nyquist_assert(cell, KGAS066_MAX_BASELINE_LAMBDA)


def test_grid_from_uv_not_header_cdelt_or_blind_256():
    grid = image_grid_from_uv(KGAS066_MAX_BASELINE_LAMBDA, fov_co_plus_pb_arcsec())
    assert grid.cell_arcsec != 0.4
    assert not (grid.nx == 256 and abs(grid.cell_arcsec - 0.1) < 1e-12)
    assert grid.fov_arcsec + 1e-12 >= fov_co_plus_pb_arcsec()
    assert grid.fov_arcsec < fov_co_plus_pb_arcsec() + 2.0 * grid.cell_arcsec


def test_nufft2_matches_dft_gaussian(uv_sampling, freqs):
    u_m, v_m = uv_sampling
    mb = max_baseline_lambda(u_m, v_m, freqs)
    grid = image_grid_from_uv(mb, fov_arcsec=16.0)
    sx, sy, flux = 1.2, 0.8, 5.0
    img = gaussian_on_grid(grid, sx, sy, flux)
    L, M = grid.pixel_lm_rad()
    strengths = np.repeat(img.ravel()[:, None], len(freqs), axis=1)
    want = dft_numpy(L.ravel(), M.ravel(), strengths, u_m, v_m, freqs)
    got = nufft2_degrid(grid, img, u_m, v_m, freqs, eps=1e-9)
    err = float(np.abs(got - want).max() / flux)
    assert err < 1e-7, f"T2 vs DFT Gaussian error {err:.3e} backend={BACKEND}"
    # stash for the session report
    test_nufft2_matches_dft_gaussian.max_err = err


def test_nufft3_not_production():
    with pytest.raises(NotImplementedError, match="type-3"):
        nufft3_degrid()


@pytest.mark.skipif(not NPZ.is_file(), reason="KILOGAS066.npz not on this machine")
def test_dummy_eval_under_half_second_on_066_uv():
    z = np.load(NPZ)
    u_m, v_m, freqs = z["u_m"], z["v_m"], z["freqs"]
    mb = max_baseline_lambda(u_m, v_m, freqs)
    grid = image_grid_from_uv(mb, fov_co_plus_pb_arcsec())
    img = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    img[grid.ny // 2, grid.nx // 2] = 1.0
    nufft2_degrid(grid, img, u_m, v_m, freqs[:1], eps=1e-6)
    t0 = time.perf_counter()
    nufft2_degrid(grid, img, u_m, v_m, freqs[:1], eps=1e-6)
    dt = time.perf_counter() - t0
    test_dummy_eval_under_half_second_on_066_uv.dt = dt
    test_dummy_eval_under_half_second_on_066_uv.grid = grid
    test_dummy_eval_under_half_second_on_066_uv.mb = mb
    assert dt < 0.5, f"dummy T2 eval {dt:.3f}s; grid {grid.nx}²@{grid.cell_arcsec:.3f}\""
