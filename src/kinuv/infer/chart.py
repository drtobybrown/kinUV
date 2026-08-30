"""Unconstrained Stage A chart + Jacobian. Not a sampler.

Eight live ``PARAM_NAMES`` only. Log on flux / gas_sigma / r_t; stable
softplus on V_0; identity on PA / vsys / (dx, dy). L-BFGS boxes are not
the chart. The JIT surface is a length-8 vector; dict packing is host-only.
``log_prob_unconstrained`` is host convenience, not autodiff.
"""

from __future__ import annotations

import numpy as np

from kinuv.infer.posterior import (
    PARAM_NAMES,
    log_prob,
    params_to_vec,
    vec_to_params,
)
from kinuv.xp import numpy_or_jax

# PARAM_NAMES order: flux, pa_deg, vsys_kms, gas_sigma_kms,
# dx_arcsec, dy_arcsec, v0_kms, r_t_arcsec
_I_FLUX = 0
_I_PA = 1
_I_VSYS = 2
_I_GS = 3
_I_DX = 4
_I_DY = 5
_I_V0 = 6
_I_RT = 7
_N = 8

_SOFTPLUS_CUT = 20.0


def softplus(x):
    """Stable ``log(1+e^x)``. Both arms finite for large ``|x|``; no Python ``if``."""
    xp = numpy_or_jax(x)
    x = xp.asarray(x)
    return xp.logaddexp(xp.zeros_like(x), x)


def inv_softplus(y):
    """Inverse of :func:`softplus`. ``y=0`` is ``-inf``; do not clip the wall."""
    xp = numpy_or_jax(y)
    y = xp.asarray(y)
    # Unused where-arm must stay finite (y=0 → -inf on the used arm; do not log(0)).
    safe = xp.where(y > 0.0, -xp.expm1(-y), xp.ones_like(y))
    small = y + xp.log(safe)
    neginf = xp.full_like(y, -xp.inf)
    body = xp.where(y > 0.0, small, neginf)
    return xp.where(y > _SOFTPLUS_CUT, y, body)


def unconstrained_to_physical(z):
    """Length-8 ``z`` → length-8 physical θ in ``PARAM_NAMES`` order."""
    xp = numpy_or_jax(z)
    z = xp.asarray(z)
    parts = (
        xp.exp(z[..., _I_FLUX]),
        z[..., _I_PA],
        z[..., _I_VSYS],
        xp.exp(z[..., _I_GS]),
        z[..., _I_DX],
        z[..., _I_DY],
        softplus(z[..., _I_V0]),
        xp.exp(z[..., _I_RT]),
    )
    return xp.stack(parts, axis=-1)


def physical_to_unconstrained(theta):
    """Length-8 physical θ → length-8 ``z``."""
    xp = numpy_or_jax(theta)
    theta = xp.asarray(theta)
    parts = (
        xp.log(theta[..., _I_FLUX]),
        theta[..., _I_PA],
        theta[..., _I_VSYS],
        xp.log(theta[..., _I_GS]),
        theta[..., _I_DX],
        theta[..., _I_DY],
        inv_softplus(theta[..., _I_V0]),
        xp.log(theta[..., _I_RT]),
    )
    return xp.stack(parts, axis=-1)


def log_abs_det_terms(z):
    """Per-axis ``ln |dθ_i/dz_i|``. Log: ``z``; softplus: ``-softplus(-z)``; identity: 0."""
    xp = numpy_or_jax(z)
    z = xp.asarray(z)
    zero = xp.zeros_like(z[..., _I_PA])
    parts = (
        z[..., _I_FLUX],
        zero,
        zero,
        z[..., _I_GS],
        zero,
        zero,
        -softplus(-z[..., _I_V0]),
        z[..., _I_RT],
    )
    return xp.stack(parts, axis=-1)


def log_abs_det_jacobian(z):
    """Scalar sum of the independent 1-D log-abs-dets."""
    xp = numpy_or_jax(z)
    terms = log_abs_det_terms(z)
    return xp.sum(terms, axis=-1)


def params_to_unconstrained(params: dict) -> np.ndarray:
    """Host dict → unconstrained vector. Not a JIT surface."""
    return np.asarray(
        physical_to_unconstrained(params_to_vec(params)), dtype=np.float64
    )


def unconstrained_to_params(z) -> dict:
    """Host unconstrained vector → dict. Not a JIT surface."""
    theta = np.asarray(unconstrained_to_physical(np.asarray(z, dtype=np.float64)))
    return vec_to_params(theta)


def log_prob_unconstrained(data, z, template, grid, *, t: float = 1.0) -> float:
    """Host ``log_prob(θ(z)) + log|det J|``. Not jitted. Not autodiff."""
    z_host = np.asarray(z, dtype=np.float64)
    params = unconstrained_to_params(z_host)
    return float(log_prob(data, params, template, grid, t=t)) + float(
        np.asarray(log_abs_det_jacobian(z_host))
    )


__all__ = [
    "PARAM_NAMES",
    "inv_softplus",
    "log_abs_det_jacobian",
    "log_abs_det_terms",
    "log_prob_unconstrained",
    "params_to_unconstrained",
    "physical_to_unconstrained",
    "softplus",
    "unconstrained_to_params",
    "unconstrained_to_physical",
]
