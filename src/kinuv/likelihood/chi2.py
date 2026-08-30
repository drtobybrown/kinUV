"""Diagonal χ² with empirical weight scale s (DEC-066-WEIGHT, ZEROMODEL).

``χ² = Σ s w |d − m|²`` accumulated in float64. ``χ²_zero`` uses ``m = 0``.
``s = 2 / ⟨w|V|²⟩`` on line-free fit channels. Not YAML 0.5. Not ``12/29``.
Reduced χ² is not the gate; report ``Δχ² = χ²_zero − χ²``.
"""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires

S_SANITY = (0.3, 1.5)


def _mag2(arr) -> np.ndarray:
    a = np.asarray(arr)
    return a.real.astype(np.float64) ** 2 + a.imag.astype(np.float64) ** 2


@requires("DEC-066-SPECRESP", "DEC-066-WEIGHT")
def empirical_s(vis, weights, line_free_mask, *, sanity=S_SANITY) -> float:
    """``s = 2 / ⟨w|V|²⟩`` on line-free fit channels with ``w > 0``."""
    mask = np.asarray(line_free_mask, dtype=bool).ravel()
    vis = np.asarray(vis)
    w = np.asarray(weights, dtype=np.float64)
    if vis.shape != w.shape:
        raise ValueError("vis and weights must have the same shape")
    if vis.ndim != 2:
        raise ValueError(f"vis must be 2D (n_row, n_chan); got {vis.shape}")
    if mask.shape[0] != vis.shape[1]:
        raise ValueError("line_free_mask must match the fit spectral axis")
    if not np.any(mask):
        raise ValueError("line_free_mask is empty")

    ww = w[:, mask]
    mag = _mag2(vis[:, mask])
    good = ww > 0.0
    if not np.any(good):
        raise ValueError("no positive-weight line-free visibilities")
    mean = float(np.mean(ww[good] * mag[good], dtype=np.float64))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"⟨w|V|²⟩ is not positive finite; got {mean}")
    s = 2.0 / mean
    lo, hi = sanity
    if not (lo < s < hi):
        raise ValueError(f"empirical s={s:.4f} outside sanity ({lo}, {hi})")
    return s


def _weighted_chi2_sum(vis, weights, model):
    from kinuv.xp import is_jax, numpy_or_jax

    xp = numpy_or_jax(vis, model, weights)
    vis_c = xp.asarray(vis)
    model_c = xp.asarray(model)
    w = xp.asarray(weights)
    if vis_c.shape != w.shape or model_c.shape != w.shape:
        raise ValueError("vis, model, and weights must share a shape")
    residual = vis_c - model_c
    mag2 = residual.real**2 + residual.imag**2
    total = xp.sum(w * mag2)
    if is_jax(total):
        return total
    return float(total)


@requires("DEC-066-WEIGHT", "DEC-066-ZEROMODEL")
def chi2(vis, model, weights, s):
    """``Σ s w |d − m|²`` with float64 accumulation."""
    from kinuv.xp import is_jax

    total = _weighted_chi2_sum(vis, weights, model) * s
    if is_jax(total):
        return total
    return float(total)


@requires("DEC-066-WEIGHT", "DEC-066-ZEROMODEL")
def chi2_zero(vis, weights, s) -> float:
    """``Σ s w |d|²`` — the V=0 model (DEC-066-ZEROMODEL)."""
    return chi2(vis, np.zeros_like(np.asarray(vis)), weights, s)


@requires("DEC-066-ZEROMODEL")
def delta_chi2(chi2_val: float, chi2_zero_val: float) -> float:
    """``Δχ² = χ²_zero − χ²``. Reduced χ² is not the gate."""
    return float(chi2_zero_val) - float(chi2_val)
