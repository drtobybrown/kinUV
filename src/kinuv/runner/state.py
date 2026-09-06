"""Closed gate-state reduction for production runners."""

from __future__ import annotations

VERIFIED_GATE_STATES = frozenset({"pass", "diagnostic_only", "pass_for_map_baseline"})
REQUIRED_GATE_NAMES = frozenset(
    {
        "preflight",
        "analytic_closure",
        "mock_recovery",
        "blank_comparison",
        "rotation_test",
        "covariance",
        "map_stability",
        "stage_b_model_adequacy",
        "posterior",
        "promotion",
    }
)


def terminal_validation_state(gates: dict[str, dict]) -> str:
    """Return ``verified``, ``validation_pending``, or ``failed``.

    Unknown, blocked, pending, uncalibrated, and historical-only states are
    closed toward ``validation_pending``; they can never become verified by
    omission from a hard-coded failure list.
    """
    if set(gates) != REQUIRED_GATE_NAMES:
        return "validation_pending"
    statuses = [str(record.get("status", "unknown")) for record in gates.values()]
    if any(status == "fail" or status.startswith("fail_") for status in statuses):
        return "failed"
    if all(status in VERIFIED_GATE_STATES for status in statuses):
        return "verified"
    return "validation_pending"
