"""Shared native-cube measurement operator for kinUV and comparators.

External packages may render an intrinsic ``(ny, nx, n_native)`` cube, but
they do not own the ALMA measurement equation. This module applies kinUV's
primary beam, visibility sampling, and native-Hann/software-bin response once.
It deliberately has no KinMS or CASA import.
"""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires
from kinuv.response.primary_beam import primary_beam
from kinuv.response.spectral import hann_then_bin
from kinuv.transforms.nufft import nufft2_degrid

from .sb import image_grid_xy_arcsec


def _validate_intrinsic_cube(cube, grid, freqs_hz):
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(cube)
    arr = xp.asarray(cube)
    expected = (int(grid.ny), int(grid.nx), int(np.asarray(freqs_hz).size))
    if tuple(arr.shape) != expected:
        raise ValueError(f"intrinsic cube shape {arr.shape} != {expected}")
    return arr


@requires("DEC-066-PB")
def attenuate_intrinsic_cube(cube, grid, freqs_hz):
    """Apply the frozen phase-centred primary beam to an intrinsic cube.

    The frequency convention exactly matches the established kinUV model: one
    primary beam at the median native frequency. A comparator cube must contain
    neither a restoring beam nor primary-beam attenuation before this call.
    """
    from kinuv.xp import is_jax, numpy_or_jax

    arr = _validate_intrinsic_cube(cube, grid, freqs_hz)
    xp = numpy_or_jax(arr)
    x, y = image_grid_xy_arcsec(grid)
    if is_jax(arr):
        x = xp.asarray(x)
        y = xp.asarray(y)
    beam = primary_beam(
        x,
        y,
        float(np.median(np.asarray(freqs_hz, dtype=np.float64))),
    )
    return arr * beam[:, :, None]


@requires("DEC-066-PB", "DEC-066-GRID")
def sample_intrinsic_cube_native(
    cube,
    grid,
    u_m,
    v_m,
    freqs_hz,
    *,
    eps: float = 1e-8,
):
    """PB-attenuate and Fourier-sample an intrinsic native-channel cube."""
    attenuated = attenuate_intrinsic_cube(cube, grid, freqs_hz)
    return nufft2_degrid(grid, attenuated, u_m, v_m, freqs_hz, eps=eps)


@requires("DEC-066-PB", "DEC-066-GRID", "DEC-066-SPECRESP")
def sample_intrinsic_cube_binned(
    data, cube, grid, *, eps: float = 1e-8, spatial_assignment: str | None = None
):
    """Apply the complete shared measurement operator to an intrinsic cube.

    ``data`` supplies only frozen sampling coordinates and the spectral-response
    contract. Its observed visibilities are never inspected.
    """
    native = sample_intrinsic_cube_native(
        cube,
        grid,
        data.u_m,
        data.v_m,
        data.freqs_native,
        eps=eps,
    )
    # Cardinal cubic B-spline assignment convolves the continuous cloud field
    # with B3 at the grid spacing. Standard particle-mesh window compensation
    # removes that numerical smoothing before the observational response.
    if spatial_assignment == "cubic_b_spline":
        from kinuv.constants import C_LIGHT_M_S

        frequency = np.asarray(data.freqs_native, dtype=np.float64)
        u_lambda = np.asarray(data.u_m, dtype=np.float64)[:, None] * frequency[None, :] / C_LIGHT_M_S
        v_lambda = np.asarray(data.v_m, dtype=np.float64)[:, None] * frequency[None, :] / C_LIGHT_M_S
        window = (
            np.sinc(u_lambda * float(grid.cell_rad)) ** 4
            * np.sinc(v_lambda * float(grid.cell_rad)) ** 4
        )
        if np.min(window) <= 0.05:
            raise ValueError("cubic B-spline compensation is unstable on this grid")
        native = native / window
    elif spatial_assignment is not None:
        raise ValueError(f"unsupported spatial-assignment kernel {spatial_assignment!r}")
    n_guard = int(data.n_guard)
    model = hann_then_bin(
        native,
        int(data.n_bin),
        n_guard=n_guard,
        weights=data.weights_native,
        vel=data.vel_native[n_guard:-n_guard],
        freqs=data.freqs_native[n_guard:-n_guard],
    )
    if tuple(model.shape) != tuple(data.vis.shape):
        raise ValueError(f"binned model {model.shape} != data vis {data.vis.shape}")
    return model
