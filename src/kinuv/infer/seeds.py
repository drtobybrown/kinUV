"""Stage A seeds and L-BFGS boxes (066-8).

Catalogue ``VSYS_SEED_KM_S`` is optical (cube VOPT / YAML). The fit axis is
radio (DEC-066-VIS). Convert at this boundary with ``optical_to_radio_kms``.
PA ±30° was a dispatch box, not an ADR — it boxed out the 180° start.
"""

from __future__ import annotations

from kinuv.forward.model import GAS_SIGMA_SEED_KM_S, VSYS_SEED_KM_S
from kinuv.geometry import pa_seed_deg
from kinuv.io.vis import optical_to_radio_kms
from kinuv.profiles.rotation import CALIBRATION_RT_ARCSEC, CALIBRATION_V0_KM_S

SHIFT_PRIOR_SIGMA_ARCSEC = 0.5
DX_DY_BOUND_ARCSEC = 2.0
FLUX_SEED_JY = 1.0
PA_BOUND_HALF_DEG = 180.0
PA_AMBIGUITY_DEG = 180.0
VSYS_BOUND_HALF_KM_S = 100.0
GAS_SIGMA_BOUNDS_KM_S = (2.0, 50.0)
V0_BOUNDS_KM_S = (0.0, 400.0)
RT_BOUNDS_ARCSEC = (0.5, 15.0)
FLUX_BOUNDS_JY = (1.0e-8, 100.0)

# Collapsed 066-8 blob vsys; the radio box must not include it.
BLOB_VSYS_KMS = 8229.0


def vsys_seed_radio_kms() -> float:
    """YAML 8299.563 optical → radio vs rest CO, same operator as the trim."""
    return float(optical_to_radio_kms(VSYS_SEED_KM_S))


def pa_start_degs() -> tuple[float, float]:
    """DEC-066-PA seed and the 180° receding-sign start (205.2° and 25.2°)."""
    pa = float(pa_seed_deg())
    return pa, pa - PA_AMBIGUITY_DEG


def stage_a_seeds(*, pa_deg: float | None = None) -> dict[str, float]:
    """ADR seeds on the **radio** vsys axis. ``(dx, dy) = (0, 0)`` is not a freeze."""
    return {
        "flux": FLUX_SEED_JY,
        "pa_deg": float(pa_seed_deg() if pa_deg is None else pa_deg),
        "vsys_kms": vsys_seed_radio_kms(),
        "gas_sigma_kms": GAS_SIGMA_SEED_KM_S,
        "dx_arcsec": 0.0,
        "dy_arcsec": 0.0,
        "v0_kms": CALIBRATION_V0_KM_S,
        "r_t_arcsec": CALIBRATION_RT_ARCSEC,
    }


def stage_a_bounds() -> dict[str, tuple[float, float]]:
    """L-BFGS-B box. ``(dx, dy)`` ±2″; PA seed ±180°; vsys ±100 km/s **radio**."""
    pa = pa_seed_deg()
    vsys = vsys_seed_radio_kms()
    box = float(DX_DY_BOUND_ARCSEC)
    return {
        "flux": FLUX_BOUNDS_JY,
        "pa_deg": (pa - PA_BOUND_HALF_DEG, pa + PA_BOUND_HALF_DEG),
        "vsys_kms": (vsys - VSYS_BOUND_HALF_KM_S, vsys + VSYS_BOUND_HALF_KM_S),
        "gas_sigma_kms": GAS_SIGMA_BOUNDS_KM_S,
        "dx_arcsec": (-box, box),
        "dy_arcsec": (-box, box),
        "v0_kms": V0_BOUNDS_KM_S,
        "r_t_arcsec": RT_BOUNDS_ARCSEC,
    }
