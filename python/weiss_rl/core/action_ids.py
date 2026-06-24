"""Stable action-id constants shared across legal-action helpers."""

from __future__ import annotations

PASS_ACTION_ID = 51


def resolve_pass_action_id() -> int:
    """Return the contract PASS action id, validating weiss_sim when available."""
    try:
        import weiss_sim
    except Exception:
        return PASS_ACTION_ID

    try:
        simulator_pass_action_id = int(weiss_sim.PASS_ACTION_ID)
    except AttributeError as exc:
        raise RuntimeError("weiss_sim is missing PASS_ACTION_ID") from exc

    if simulator_pass_action_id != PASS_ACTION_ID:
        raise RuntimeError(
            "PASS_ACTION_ID mismatch between weiss_rl and weiss_sim: "
            f"expected {PASS_ACTION_ID}, got {simulator_pass_action_id}"
        )
    return PASS_ACTION_ID


__all__ = ["PASS_ACTION_ID", "resolve_pass_action_id"]
