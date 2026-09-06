from kinuv.runner.state import REQUIRED_GATE_NAMES, terminal_validation_state


def test_gate_reduction_is_closed_against_pending_and_unknown_states():
    passing = {name: {"status": "pass"} for name in REQUIRED_GATE_NAMES}
    assert terminal_validation_state(passing) == "verified"
    assert terminal_validation_state({}) == "validation_pending"
    assert (
        terminal_validation_state({"preflight": {"status": "pass"}})
        == "validation_pending"
    )
    assert (
        terminal_validation_state(
            {**passing, "rotation_test": {"status": "pending_bootstrap"}}
        )
        == "validation_pending"
    )
    assert (
        terminal_validation_state(
            {**passing, "posterior": {"status": "blocked_no_current_draws"}}
        )
        == "validation_pending"
    )
    assert terminal_validation_state({**passing, "promotion": {}}) == "validation_pending"
    assert (
        terminal_validation_state(
            {**passing, "rotation_test": {"status": "fail_arithmetic"}}
        )
        == "failed"
    )
