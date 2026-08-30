"""Native Hanning then software spectral bin (DEC-066-SPECRESP).

Data are already correlator-Hann'd: never Hann the visibilities. The model is
convolved with ``[0.25, 0.5, 0.25]`` on the native axis **with guard channels**,
then binned with the same weighted-mean operator as the data. Hann on an
already-binned axis is forbidden. Implicit zero-padding at native ends is
forbidden for science evaluations — guards exist so the kernel sees real
neighbours, even if the convolution routine itself pads the *guard* ends
(those samples are discarded).
"""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires

HANN_KERNEL = np.array([0.25, 0.5, 0.25], dtype=np.float64)


def s_theory(n_bin: int) -> float:
    """``3N / (8(N − 0.75))`` vs N independent Hann channels. Not a 066 weight."""
    n = int(n_bin)
    if n < 1:
        raise ValueError(f"n_bin must be >= 1, got {n_bin}")
    return 3.0 * n / (8.0 * (n - 0.75))


def rho_bin(n_bin: int) -> float:
    """Adjacent-bin correlation after equal-weight Hann-then-bin.

    ``Cov(S, S_next) = 3/8 σ²``, ``Var(S) = (N − 0.75) σ²``.
    """
    n = int(n_bin)
    if n < 1:
        raise ValueError(f"n_bin must be >= 1, got {n_bin}")
    return 0.375 / (n - 0.75)


def hann_native(arr, *, axis: int = -1):
    """Convolve the spectral axis with ``[0.25, 0.5, 0.25]``.

    Edge samples see implicit zeros. Pass ≥1 guard channel on each end and
    trim after convolution so those zeros never enter a science bin.
    """
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(arr)
    a = xp.asarray(arr)
    if a.size == 0:
        return a
    axis = int(axis) % a.ndim
    a = xp.moveaxis(a, axis, -1)
    k = xp.asarray(HANN_KERNEL)
    zeros = xp.zeros(a.shape[:-1] + (1,), dtype=a.dtype)
    left = xp.concatenate([zeros, a[..., :-1]], axis=-1)
    right = xp.concatenate([a[..., 1:], zeros], axis=-1)
    out = k[0] * left + k[1] * a + k[2] * right
    return xp.moveaxis(out, -1, axis)


def bin_channels(
    vis: np.ndarray,
    weights: np.ndarray,
    vel: np.ndarray,
    freqs: np.ndarray,
    bin_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Weighted spectral bin: ``V_b = Σ V W / Σ W``, ``W_b = Σ W``.

    ``vel`` and ``freqs`` are arithmetic means. Trailing channels that do not
    fill a bin are dropped (``n_dropped``). ``bin_factor == 1`` is a no-op.
    """
    if bin_factor < 1:
        raise ValueError(f"bin_factor must be >= 1, got {bin_factor}")
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(vis, weights)
    vis = xp.asarray(vis)
    weights = xp.asarray(weights)
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    if vis.ndim != 2:
        raise ValueError(f"vis must be 2D (n_row, n_chan); got {vis.shape}")
    n_chan = vis.shape[1]
    vel = xp.asarray(vel).ravel()
    freqs = xp.asarray(freqs).ravel()
    if vel.shape[0] != n_chan or freqs.shape[0] != n_chan:
        raise ValueError("vel and freqs must match spectral dimension of vis")

    if bin_factor == 1:
        return vis, weights, vel, freqs, 0

    n_use = (n_chan // bin_factor) * bin_factor
    n_drop = n_chan - n_use
    if n_use == 0:
        raise ValueError(
            f"After binning by {bin_factor}, no full bins remain ({n_chan} channels)"
        )

    vis = vis[:, :n_use]
    weights = weights[:, :n_use]
    vel = vel[:n_use]
    freqs = freqs[:n_use]

    nrow, n_b = vis.shape[0], n_use // bin_factor
    vis_r = vis.reshape(nrow, n_b, bin_factor)
    w_r = weights.reshape(nrow, n_b, bin_factor)
    w_sum = xp.sum(w_r, axis=2)
    numer = xp.sum(vis_r * w_r, axis=2)
    vis_b = xp.where(w_sum > 0, numer / xp.where(w_sum > 0, w_sum, 1.0), 0.0)
    weights_b = w_sum.astype(weights.dtype)
    vel_b = xp.mean(vel.reshape(n_b, bin_factor), axis=1)
    freqs_b = xp.mean(freqs.reshape(n_b, bin_factor), axis=1)
    return vis_b, weights_b, vel_b, freqs_b, n_drop


@requires("DEC-066-SPECRESP")
def hann_then_bin(
    model_native,
    n_bin: int,
    *,
    n_guard: int = 1,
    weights=None,
    vel=None,
    freqs=None,
):
    """Hann native model (with guards), trim guards, then bin ``N``.

    ``model_native`` is ``(n_row, n_trim + 2*n_guard)`` or 1-D spectral.
    ``weights``, if given, match the **trimmed** (no-guard) native axis — the
    same operator as the data npz. Returns only the binned model visibilities.
    """
    n_bin = int(n_bin)
    n_guard = int(n_guard)
    if n_bin < 1:
        raise ValueError(f"n_bin must be >= 1, got {n_bin}")
    if n_guard < 1:
        raise ValueError("n_guard must be >= 1; implicit zero-pad is forbidden")

    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(model_native, weights, vel, freqs)
    m = xp.asarray(model_native)
    one_d = m.ndim == 1
    if one_d:
        m = m[None, :]
    if m.ndim != 2:
        raise ValueError("model_native must be (n_chan,) or (n_row, n_chan)")

    n_spec = m.shape[1]
    if n_spec <= 2 * n_guard:
        raise ValueError("model_native is too short for the requested guards")

    hann = hann_native(m, axis=-1)
    core = hann[:, n_guard : n_spec - n_guard]
    n_trim = core.shape[1]

    if weights is None:
        w = xp.ones(core.shape, dtype=core.dtype)
    else:
        w = xp.asarray(weights)
        if w.ndim == 1:
            w = xp.broadcast_to(w, core.shape)
        if w.shape != core.shape:
            raise ValueError(
                f"weights shape {w.shape} must match trimmed model {core.shape}"
            )

    if vel is None:
        vel_in = xp.arange(n_trim, dtype=core.real.dtype)
    else:
        vel_in = xp.asarray(vel).ravel()
        if vel_in.shape[0] != n_trim:
            raise ValueError("vel must match the trimmed native axis")
    if freqs is None:
        freqs_in = xp.arange(n_trim, dtype=core.real.dtype)
    else:
        freqs_in = xp.asarray(freqs).ravel()
        if freqs_in.shape[0] != n_trim:
            raise ValueError("freqs must match the trimmed native axis")

    vis_b, _, _, _, _ = bin_channels(core, w, vel_in, freqs_in, n_bin)
    if one_d:
        return vis_b[0]
    return vis_b


def native_diagonal(*_args, **_kwargs):
    """Removed. The 066-7 mock fallback is not a valid SPECRESP operator."""
    raise RuntimeError(
        "native_diagonal is removed. Use kinuv.response.spectral.hann_then_bin "
        "(Hann+bin with native guards). Do not chi2 on un-Hann'd native channels."
    )
