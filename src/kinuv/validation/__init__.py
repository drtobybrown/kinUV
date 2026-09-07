"""Scientific validation gates kept separate from likelihood primitives."""

from .rotation import (
    MIN_NONROT_BOOTSTRAP_TRIALS,
    NONROT_BOOTSTRAP_P_MAX,
    NONROT_DELTA_CHI2_MIN,
    ROTATION_GATE_ID,
    RotationTestMetrics,
    rotation_test_metrics,
)
from .groups import (
    GroupedVisibilityFolds,
    VisibilityGroup,
    build_grouped_visibility_folds,
)
from .covariance import (
    CovarianceParameters,
    CovarianceSelection,
    fit_covariance_by_stratum,
    fit_covariance_stratum,
    select_grouped_covariance,
    whitened_innovations,
    whitening_diagnostics,
)

__all__ = [
    "MIN_NONROT_BOOTSTRAP_TRIALS",
    "NONROT_BOOTSTRAP_P_MAX",
    "NONROT_DELTA_CHI2_MIN",
    "ROTATION_GATE_ID",
    "RotationTestMetrics",
    "rotation_test_metrics",
    "GroupedVisibilityFolds",
    "VisibilityGroup",
    "build_grouped_visibility_folds",
    "CovarianceParameters",
    "CovarianceSelection",
    "fit_covariance_by_stratum",
    "fit_covariance_stratum",
    "select_grouped_covariance",
    "whitened_innovations",
    "whitening_diagnostics",
]
