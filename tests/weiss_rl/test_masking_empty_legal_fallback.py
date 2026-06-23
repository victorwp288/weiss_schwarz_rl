from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.core.masking import (
    PASS_ACTION_ID as CONTRACT_PASS_ACTION_ID,
)
from weiss_rl.core.masking import (
    MaskingAnomalyCounters,
    apply_empty_legal_action_fallback,
    empty_legal_guard,
    resolve_pass_action_id,
)


def test_empty_legal_guard_returns_pass_payload_for_empty_rows() -> None:
    legal_mask = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    empty_rows, actions, logp, entropy = empty_legal_guard(legal_mask)

    assert np.array_equal(empty_rows, np.array([False, True, True]))
    assert np.array_equal(
        actions,
        np.array([CONTRACT_PASS_ACTION_ID, CONTRACT_PASS_ACTION_ID, CONTRACT_PASS_ACTION_ID], dtype=np.int64),
    )
    assert np.array_equal(logp, np.zeros((3,), dtype=np.float32))
    assert np.array_equal(entropy, np.zeros((3,), dtype=np.float32))


def test_empty_legal_guard_increments_anomaly_counter_only_for_empty_rows() -> None:
    counters = MaskingAnomalyCounters()

    empty_legal_guard(np.array([[1, 0], [0, 0]], dtype=np.uint8), counters=counters)
    empty_legal_guard(np.array([[1, 0], [0, 1]], dtype=np.uint8), counters=counters)
    empty_legal_guard(np.array([[0, 0], [0, 0]], dtype=np.uint8), counters=counters)

    assert counters.empty_legal == 3


def test_apply_empty_legal_action_fallback_overrides_only_empty_rows() -> None:
    actions = np.array([7, 13, 19], dtype=np.uint32)
    legal_mask = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.uint8)
    counters = MaskingAnomalyCounters()

    empty_rows, adjusted_actions = apply_empty_legal_action_fallback(actions, legal_mask, counters=counters)

    assert np.array_equal(empty_rows, np.array([False, True, False]))
    assert np.array_equal(adjusted_actions, np.array([7, CONTRACT_PASS_ACTION_ID, 19], dtype=np.uint32))
    assert counters.empty_legal == 1


def test_resolve_pass_action_id_uses_contract_default_when_weiss_sim_is_missing() -> None:
    original_module = sys.modules.pop("weiss_sim", None)
    try:
        assert resolve_pass_action_id() == CONTRACT_PASS_ACTION_ID
    finally:
        if original_module is not None:
            sys.modules["weiss_sim"] = original_module


def test_resolve_pass_action_id_rejects_mismatched_weiss_sim_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "weiss_sim", SimpleNamespace(PASS_ACTION_ID=17))

    with pytest.raises(RuntimeError, match="PASS_ACTION_ID mismatch"):
        resolve_pass_action_id()
