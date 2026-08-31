"""CPU NumPyro NUTS on the G2 chart. Not host ``log_prob_unconstrained``.

Six sampled names; ``(dx, dy)`` frozen at MAP host floats (DEC-066-SHIFT).
Energy ``U(z_6) = 0.5 (chi2 + shift_prior_const) - log|det J|``.
``SAMPLER_NAME`` in ``posterior`` stays ``laplace_mh``.
"""

from __future__ import annotations

import numpy as np

from kinuv.infer.chart import (
    PARAM_NAMES,
    log_abs_det_jacobian,
    unconstrained_to_physical,
)
from kinuv.infer.map import predict_binned, shift_prior
from kinuv.infer.posterior import SAMPLER_NAME, ess_bulk, ess_tail, split_rhat
from kinuv.likelihood.chi2 import chi2
from kinuv.xp import numpy_or_jax

SAMPLED_NAMES = (
    "flux",
    "pa_deg",
    "vsys_kms",
    "gas_sigma_kms",
    "v0_kms",
    "r_t_arcsec",
)
SAMPLED_IDX = (0, 1, 2, 3, 6, 7)
DX_IDX = 4
DY_IDX = 5
FROZEN_NAMES = ("dx_arcsec", "dy_arcsec")
NUTS_SAMPLER = "nuts"
# Interactive/subagent cap only. Headless workers (DEC-067-RUNNER) ignore this.
WALL_CAP_S = 7200.0
# Identity-chart PA / vsys / V_0 dwarf log-flux in an identity metric.
Z6_SCALES = (0.3, 8.0, 25.0, 0.3, 25.0, 0.3)


def sampled_z_from_physical(theta8):
    """Length-8 physical θ → length-6 unconstrained (sampled names)."""
    from kinuv.infer.chart import physical_to_unconstrained

    z8 = physical_to_unconstrained(theta8)
    z8 = np.asarray(z8, dtype=np.float64)
    return z8[list(SAMPLED_IDX)]


def stitch_z8(z6, dx_map, dy_map):
    """Insert host MAP ``(dx, dy)`` (identity chart) into a length-8 ``z``."""
    xp = numpy_or_jax(z6)
    z = xp.asarray(z6)
    dx = xp.asarray(dx_map)
    dy = xp.asarray(dy_map)
    return xp.stack((z[0], z[1], z[2], z[3], dx, dy, z[4], z[5]))


def params_from_theta(theta, dx_map, dy_map):
    """XLA ``predict_binned`` dict: sampled slots from θ, shifts as Python floats."""
    return {
        "flux": theta[0],
        "pa_deg": theta[1],
        "vsys_kms": theta[2],
        "gas_sigma_kms": theta[3],
        "dx_arcsec": float(dx_map),
        "dy_arcsec": float(dy_map),
        "v0_kms": theta[6],
        "r_t_arcsec": theta[7],
    }


def potential_unconstrained(z6, data, template, grid, dx_map, dy_map):
    """``U = 0.5 (chi2 + shift_prior_const) - log|J|``. JAX if ``z6`` is JAX."""
    dx = float(dx_map)
    dy = float(dy_map)
    prior = float(shift_prior(dx, dy))
    z8 = stitch_z8(z6, dx, dy)
    theta = unconstrained_to_physical(z8)
    params = params_from_theta(theta, dx, dy)
    model = predict_binned(data, params, template, grid, xla=True)
    c = chi2(data.vis, model, data.weights, float(data.s))
    logj = log_abs_det_jacobian(z8)
    return 0.5 * (c + prior) - logj


def make_potential(data, template, grid, dx_map, dy_map):
    """Closed-over ``U(z6)`` for ``jax.jit`` / NumPyro ``potential_fn``."""
    import jax.numpy as jnp

    tmpl = jnp.asarray(template)
    vis = jnp.asarray(data.vis)
    wgt = jnp.asarray(data.weights)
    s = float(data.s)
    dx = float(dx_map)
    dy = float(dy_map)
    prior = float(shift_prior(dx, dy))

    def U(z6):
        z8 = stitch_z8(z6, dx, dy)
        theta = unconstrained_to_physical(z8)
        params = params_from_theta(theta, dx, dy)
        model = predict_binned(data, params, tmpl, grid, xla=True)
        c = chi2(vis, model, wgt, s)
        logj = log_abs_det_jacobian(z8)
        return 0.5 * (c + prior) - logj

    return U


def stitch_draws_8col(draws6, dx_map, dy_map):
    """``(n_chain, n_draw, 6)`` sampled physical → ``(..., 8)`` PARAM_NAMES order."""
    a = np.asarray(draws6, dtype=np.float64)
    if a.ndim == 2:
        a = a[None, ...]
    n_c, n_d, n_s = a.shape
    if n_s != 6:
        raise ValueError(f"sampled draws must have 6 trailing names; got {n_s}")
    out = np.empty((n_c, n_d, 8), dtype=np.float64)
    out[..., 0] = a[..., 0]
    out[..., 1] = a[..., 1]
    out[..., 2] = a[..., 2]
    out[..., 3] = a[..., 3]
    out[..., DX_IDX] = float(dx_map)
    out[..., DY_IDX] = float(dy_map)
    out[..., 6] = a[..., 4]
    out[..., 7] = a[..., 5]
    return out


def physical_sampled_from_z6(z6_draws, dx_map, dy_map):
    """Unconstrained ``z6`` chains → physical 8-col draws."""
    z = np.asarray(z6_draws, dtype=np.float64)
    if z.ndim == 2:
        z = z[None, ...]
    n_c, n_d, _ = z.shape
    z6 = z.reshape(-1, 6)
    z8 = np.empty((z6.shape[0], 8), dtype=np.float64)
    z8[:, 0:4] = z6[:, 0:4]
    z8[:, DX_IDX] = float(dx_map)
    z8[:, DY_IDX] = float(dy_map)
    z8[:, 6:8] = z6[:, 4:6]
    th = np.asarray(unconstrained_to_physical(z8), dtype=np.float64)
    phys6 = th[:, list(SAMPLED_IDX)].reshape(n_c, n_d, 6)
    return stitch_draws_8col(phys6, dx_map, dy_map)


def mixing_sampled(chains8):
    """R_hat, bulk ESS, and tail ESS on the six sampled columns only."""
    a = np.asarray(chains8, dtype=np.float64)
    sampled = a[..., list(SAMPLED_IDX)]
    rhat = split_rhat(sampled)
    ess = ess_bulk(sampled)
    ess_t = ess_tail(sampled)
    return {
        name: {
            "rhat": float(rhat[i]),
            "ess": float(ess[i]),
            "ess_tail": float(ess_t[i]),
        }
        for i, name in enumerate(SAMPLED_NAMES)
    }


def mixing_ok(
    mix,
    *,
    rhat_max: float = 1.01,
    ess_min: float = 200.0,
    ess_tail_min: float | None = None,
) -> bool:
    for v in mix.values():
        if not (
            np.isfinite(v["rhat"])
            and v["rhat"] < rhat_max
            and np.isfinite(v["ess"])
            and v["ess"] > ess_min
        ):
            return False
        if ess_tail_min is not None:
            tail = v.get("ess_tail", float("nan"))
            if not (np.isfinite(tail) and tail > ess_tail_min):
                return False
    return True


def run_nuts_z6(
    U,
    z6_init,
    *,
    rng_seed: int = 0,
    num_warmup: int = 64,
    num_samples: int = 64,
    num_chains: int = 4,
    jitter: float = 0.02,
    progress_bar: bool = False,
):
    """Sequential CPU NUTS on ``U(z6)``. Returns unconstrained ``(n_chain, n_draw, 6)``."""
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    z0 = np.asarray(z6_init, dtype=np.float64).reshape(6)
    rng = np.random.default_rng(int(rng_seed))
    scales = np.asarray(Z6_SCALES, dtype=np.float64)
    u0 = z0 / scales
    inits = u0 + jitter * rng.standard_normal((int(num_chains), 6))
    scales_j = jnp.asarray(scales)

    def U_unit(u):
        return U(u * scales_j)

    kernel = NUTS(
        potential_fn=U_unit,
        max_tree_depth=8,
        adapt_mass_matrix=True,
        target_accept_prob=0.8,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=int(num_warmup),
        num_samples=int(num_samples),
        num_chains=int(num_chains),
        chain_method="sequential",
        progress_bar=bool(progress_bar),
        jit_model_args=False,
    )
    mcmc.run(
        jax.random.PRNGKey(int(rng_seed)),
        init_params=jnp.asarray(inits),
        extra_fields=("num_steps",),
    )
    u_draws = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=np.float64)
    z_draws = u_draws * scales
    extras = mcmc.get_extra_fields(group_by_chain=True)
    n_steps = extras.get("num_steps")
    mean_steps = float(np.mean(np.asarray(n_steps))) if n_steps is not None else float("nan")
    return z_draws, mean_steps, mcmc


def product_record(
    *,
    draws8,
    mix,
    pa_init_deg,
    dx_map,
    dy_map,
    autodiff_ok: bool,
    mixing_pass: bool,
    leftover_chi2_structured: bool,
    r_t_at_floor: bool,
    mean_num_steps: float,
    eval_s: float,
    note: str,
):
    """JSON-able NUTS product. ``sampler`` is nuts only after autodiff (+ mixing for 066)."""
    label = NUTS_SAMPLER if (autodiff_ok and mixing_pass) else SAMPLER_NAME
    return {
        "sampler": label,
        "draws": np.asarray(draws8, dtype=np.float64).tolist(),
        "param_names": list(PARAM_NAMES),
        "sampled_names": list(SAMPLED_NAMES),
        "frozen_names": list(FROZEN_NAMES),
        "pa_init_deg": float(pa_init_deg),
        "dx_arcsec": float(dx_map),
        "dy_arcsec": float(dy_map),
        "mixing": mix,
        "intervals_calibrated": False,
        "leftover_chi2_structured": bool(leftover_chi2_structured),
        "r_t_at_floor": bool(r_t_at_floor),
        "mean_num_steps": float(mean_num_steps),
        "eval_s": float(eval_s),
        "note": note,
    }


__all__ = [
    "DX_IDX",
    "DY_IDX",
    "FROZEN_NAMES",
    "NUTS_SAMPLER",
    "PARAM_NAMES",
    "SAMPLED_IDX",
    "SAMPLED_NAMES",
    "WALL_CAP_S",
    "Z6_SCALES",
    "make_potential",
    "mixing_ok",
    "mixing_sampled",
    "physical_sampled_from_z6",
    "potential_unconstrained",
    "product_record",
    "run_nuts_z6",
    "sampled_z_from_physical",
    "stitch_draws_8col",
    "stitch_z8",
]
