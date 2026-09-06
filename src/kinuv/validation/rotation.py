"""Prospective rotation-test accounting for DEC-KINUV-CROSSDOMAIN-RECOVERY."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.likelihood.chi2 import delta_chi2_blank, delta_chi2_nonrot

NONROT_DELTA_CHI2_MIN = 25.0
NONROT_BOOTSTRAP_P_MAX = 0.01
MIN_NONROT_BOOTSTRAP_TRIALS = 199
ROTATION_GATE_ID = "crossdomain-nonrot-bootstrap-v1"


@dataclass(frozen=True)
class RotationTestMetrics:
    """Likelihood-only scores and the preregistered null-bootstrap gate."""

    chi2_blank: float
    chi2_nonrot: float
    chi2_rot: float
    delta_chi2_blank: float
    delta_chi2_nonrot: float
    bootstrap_exceedances: int | None
    bootstrap_trials: int | None
    bootstrap_p: float | None
    delta_arithmetic_pass: bool
    bootstrap_arithmetic_pass: bool | None
    scientific_gate_pass: bool
    gate_id: str
    status: str


def rotation_test_metrics(
    *,
    chi2_blank: float,
    chi2_nonrot: float,
    chi2_rot: float,
    bootstrap_exceedances: int | None = None,
    bootstrap_trials: int | None = None,
) -> RotationTestMetrics:
    """Build prospective rotation metrics without Wilks approximations.

    A bootstrap result is accepted only when both integer counts are supplied,
    at least 199 null realizations were completed, and the plus-one estimate
    ``p=(k+1)/(M+1)`` is at most 0.01. Missing bootstrap evidence leaves the
    rotation gate pending even if the likelihood improvement exceeds 25.
    """
    values = np.asarray([chi2_blank, chi2_nonrot, chi2_rot], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("all chi-square values must be finite")
    d_blank = delta_chi2_blank(chi2_rot, chi2_blank)
    d_nonrot = delta_chi2_nonrot(chi2_rot, chi2_nonrot)
    delta_pass = bool(d_nonrot >= NONROT_DELTA_CHI2_MIN)

    if (bootstrap_exceedances is None) != (bootstrap_trials is None):
        raise ValueError("bootstrap_exceedances and bootstrap_trials are a pair")
    if bootstrap_trials is None:
        p_value = None
        bootstrap_pass = None
        status = "pending_bootstrap" if delta_pass else "fail_delta_chi2_nonrot"
        k = m = None
    else:
        k = int(bootstrap_exceedances)
        m = int(bootstrap_trials)
        if k != bootstrap_exceedances or m != bootstrap_trials:
            raise ValueError("bootstrap counts must be integers")
        if m < MIN_NONROT_BOOTSTRAP_TRIALS:
            raise ValueError(
                f"bootstrap_trials must be >= {MIN_NONROT_BOOTSTRAP_TRIALS}"
            )
        if not 0 <= k <= m:
            raise ValueError("bootstrap_exceedances must lie in [0, trials]")
        p_value = (k + 1.0) / (m + 1.0)
        bootstrap_pass = bool(p_value <= NONROT_BOOTSTRAP_P_MAX)
        status = (
            "arithmetic_pass_evidence_unverified"
            if delta_pass and bootstrap_pass
            else "fail_arithmetic"
        )

    return RotationTestMetrics(
        chi2_blank=float(chi2_blank),
        chi2_nonrot=float(chi2_nonrot),
        chi2_rot=float(chi2_rot),
        delta_chi2_blank=float(d_blank),
        delta_chi2_nonrot=float(d_nonrot),
        bootstrap_exceedances=k,
        bootstrap_trials=m,
        bootstrap_p=None if p_value is None else float(p_value),
        delta_arithmetic_pass=delta_pass,
        bootstrap_arithmetic_pass=bootstrap_pass,
        scientific_gate_pass=False,
        gate_id=ROTATION_GATE_ID,
        status=status,
    )
