"""Frozen KGAS066 sky geometry (DEC-066-INC, DEC-066-PA).

Inclination is ``arccos(ba)`` with catalogue ``ba = 0.721``, not a stored 43.9°.
PA here is the receding-side *seed* only (CO YAML 205.2° E of N); PA is fitted
later. YAML ``inc_init = 51.690`` is discarded. Optical 108.9° is not the seed.
*i* is not loaded from a moment map. Warps, PA(r), and i(r) are out of scope.

Sky offsets: +x East, +y North. ``rotate_by_pa`` puts +x on the receding major
axis. ``incline`` deprojects the minor axis by ``1/cos(i)``.
"""

from __future__ import annotations

import numpy as np

from kinuv.decisions import requires

_CATALOGUE_BA = 0.721
_INC_PRIOR_HALF_WIDTH_DEG = 5.0
_PA_SEED_DEG = 205.2  # receding, E of N; not optical 108.9


@requires("DEC-066-INC")
def catalogue_ba() -> float:
    """Catalogue *b/a* that defines the frozen inclination."""
    return _CATALOGUE_BA


@requires("DEC-066-INC")
def inclination_rad() -> float:
    """Frozen inclination [rad]: ``arccos(ba)``."""
    return float(np.arccos(catalogue_ba()))


@requires("DEC-066-INC")
def inclination_deg() -> float:
    """Frozen inclination [deg] (~43.9), computed from ``ba``."""
    return float(np.degrees(inclination_rad()))


@requires("DEC-066-INC")
def inclination_prior_half_width_deg() -> float:
    """Tight prior half-width [deg] (±5)."""
    return _INC_PRIOR_HALF_WIDTH_DEG


@requires("DEC-066-PA")
def pa_seed_deg() -> float:
    """Receding-side PA seed [deg] east of north. Fitted later."""
    return _PA_SEED_DEG


@requires("DEC-066-PA")
def pa_seed_rad() -> float:
    """Receding-side PA seed [rad] east of north. Fitted later."""
    return float(np.radians(pa_seed_deg()))


@requires("DEC-066-PA")
def rotate_by_pa(x_east, y_north, pa_rad):
    """Sky (E, N) → (major, minor). North at ``PA=0`` maps to +x."""
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(x_east, y_north)
    x_east = xp.asarray(x_east)
    y_north = xp.asarray(y_north)
    s, c = xp.sin(pa_rad), xp.cos(pa_rad)
    return x_east * s + y_north * c, x_east * c - y_north * s


@requires("DEC-066-INC")
def incline(x_maj, y_min, i_rad):
    """Deproject minor axis by ``1/cos(i)``. Identity at ``i=0``."""
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(x_maj, y_min)
    x_maj = xp.asarray(x_maj)
    y_min = xp.asarray(y_min)
    return x_maj, y_min / xp.cos(i_rad)


@requires("DEC-066-INC", "DEC-066-PA")
def sky_to_galaxy(x_east, y_north, pa_rad, i_rad):
    """Sky (E, N) → galaxy plane: rotate by PA, then incline by *i*."""
    return incline(*rotate_by_pa(x_east, y_north, pa_rad), i_rad)


@requires("DEC-066-INC", "DEC-066-PA")
def galaxy_to_sky(x_gal, y_gal, pa_rad, i_rad):
    """Galaxy plane → sky (E, N): project by *i*, then undo PA rotation."""
    x_gal = np.asarray(x_gal, dtype=np.float64)
    y_min = np.asarray(y_gal, dtype=np.float64) * np.cos(i_rad)
    s, c = np.sin(pa_rad), np.cos(pa_rad)
    return x_gal * s + y_min * c, x_gal * c - y_min * s
