"""Stage A L-BFGS MAP on Hann+bin KGAS066 (066-8). No NUTS. No Stage B rings."""

from .map import (
    DX_DY_BOUND_ARCSEC,
    SHIFT_PRIOR_SIGMA_ARCSEC,
    MapResult,
    gate_delta_chi2,
    map_gate_scores,
    map_objective,
    run_stage_a_map,
    score_seed_delta_chi2,
    shift_prior,
    stage_a_bounds,
    stage_a_seeds,
)
from .seeds import (
    BLOB_VSYS_KMS,
    PA_BOUND_HALF_DEG,
    pa_start_degs,
    vsys_seed_radio_kms,
)

__all__ = [
    "BLOB_VSYS_KMS",
    "DX_DY_BOUND_ARCSEC",
    "PA_BOUND_HALF_DEG",
    "SHIFT_PRIOR_SIGMA_ARCSEC",
    "MapResult",
    "gate_delta_chi2",
    "map_gate_scores",
    "map_objective",
    "pa_start_degs",
    "run_stage_a_map",
    "score_seed_delta_chi2",
    "shift_prior",
    "stage_a_bounds",
    "stage_a_seeds",
    "vsys_seed_radio_kms",
]
