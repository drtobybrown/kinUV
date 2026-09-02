"""MAP quality flags. Not part of the visibility likelihood.

JSON in, dict out. Do not call ``run_stage_a_map``. Leftover-vs-velocity
needs leftover arrays (not Stage A JSON alone). The production ``r_t``
floor is the L-BFGS box in ``kinuv.infer.seeds.RT_BOUNDS_ARCSEC``, not a
science prior; this module copies the lower edge so flags stay off the
fitter import graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Matches ``RT_BOUNDS_ARCSEC[0]`` in ``kinuv.infer.seeds`` (L-BFGS box).
RT_FLOOR_ARCSEC = 0.5
RT_FLOOR_TOL_ARCSEC = 0.01
PA_ALIAS_DEG = 21.9
PA_ALIAS_HALF_DEG = 40.0
UV_LEFTOVER_BINS = 12


def _pa_sep_deg(pa_deg, target_deg) -> float:
    return abs((float(pa_deg) - float(target_deg) + 180.0) % 360.0 - 180.0)


def _span_over_mean(y) -> float:
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    m = float(np.mean(y))
    if m <= 0.0:
        return 0.0
    return float((np.max(y) - np.min(y)) / m)


def _binned_span(x, y, n_bin: int = UV_LEFTOVER_BINS) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < n_bin:
        return _span_over_mean(y)
    lo, hi = np.percentile(x, [2, 98])
    edges = np.linspace(lo, hi, n_bin + 1)
    means = []
    for i in range(n_bin):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(sel):
            means.append(float(np.mean(y[sel])))
    return _span_over_mean(means) if means else 0.0


def leftover_velocity_structured(baseline_m, chi2_row, chi2_chan) -> bool:
    """True when leftover vs velocity is more structured than leftover vs uv.

    Per-row leftover scatter is large even when the uv trend is flat, so
    compare binned leftover vs baseline to leftover vs channel. A max/mean
    ratio on ``chi2_chan`` alone is a weak gate.
    """
    uv = _binned_span(baseline_m, chi2_row)
    vel = _span_over_mean(chi2_chan)
    return bool(vel > uv)


def _load_leftover(leftover):
    if leftover is None:
        return None
    if isinstance(leftover, (str, Path)):
        path = Path(leftover)
        if path.suffix.lower() == ".json":
            path = path.with_suffix(".npz")
        z = np.load(path)
        return z["baseline_m"], z["chi2_row"], z["chi2_chan"]
    return leftover["baseline_m"], leftover["chi2_row"], leftover["chi2_chan"]


def map_quality_flags(stage_a, leftover_npz=None) -> dict:
    """Flags for one Stage A MAP.

    ``stage_a`` is a JSON path or a dict with ``r_t_arcsec``, ``pa_deg``,
    ``delta_chi2``. ``leftover_npz`` is the leftover ``.npz`` (or sibling
    ``.json``) from ``plot_leftover_chi2``, or a dict of those arrays.
    """
    if isinstance(stage_a, (str, Path)):
        rec = json.loads(Path(stage_a).read_text())
    else:
        rec = dict(stage_a)
    rt = float(rec["r_t_arcsec"])
    pa = float(rec["pa_deg"]) % 360.0
    dchi = float(rec["delta_chi2"])
    rt_floor = bool(abs(rt - RT_FLOOR_ARCSEC) <= RT_FLOOR_TOL_ARCSEC)
    leftover_flag = False
    leftover_uv_span = None
    leftover_vel_span = None
    arrays = _load_leftover(leftover_npz)
    if arrays is not None:
        baseline_m, chi2_row, chi2_chan = arrays
        leftover_flag = leftover_velocity_structured(
            baseline_m, chi2_row, chi2_chan
        )
        leftover_uv_span = _binned_span(baseline_m, chi2_row)
        leftover_vel_span = _span_over_mean(chi2_chan)
    return {
        "r_t_at_floor": rt_floor,
        "beats_zero": bool(dchi > 0.0),
        "delta_chi2_vs_zero_fail": bool(dchi <= 0.0),
        "delta_chi2": dchi,
        "pa_alias": bool(_pa_sep_deg(pa, PA_ALIAS_DEG) < PA_ALIAS_HALF_DEG),
        "pa_deg": pa,
        "i_held_fixed": True,
        "h_z_in_model": False,
        "axisym_assumed": True,
        "leftover_chi2_structured": leftover_flag,
        "leftover_uv_span": leftover_uv_span,
        "leftover_vel_span": leftover_vel_span,
        "rings_are_not_a_warp": True,
        "nuts_absent": True,
        # Off-floor r_t is not a license to quote dV/dr while leftover
        # vs velocity is structured (or unmeasured).
        "quote_inner_slope": bool(
            (not rt_floor) and (arrays is not None) and (not leftover_flag)
        ),
    }
