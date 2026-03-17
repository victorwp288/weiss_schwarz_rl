from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.masking import (
    PASS_ACTION_ID,
    MaskingAnomalyCounters,
    apply_empty_legal_action_fallback,
    assert_strictly_increasing_legal_ids,
    empty_legal_guard,
    masked_log_softmax,
    resolve_pass_action_id,
)


def _load_policy_example_module():
    module_path = Path(__file__).resolve().parents[3] / "examples" / "policy_example.py"
    spec = importlib.util.spec_from_file_location("test_policy_example_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legal_ids_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError):
        assert_strictly_increasing_legal_ids(np.array([1, 1, 3], dtype=np.uint32))


def test_legal_ids_accepts_sorted_unique() -> None:
    assert_strictly_increasing_legal_ids(np.array([2, 5, 9], dtype=np.uint32))


def test_masked_log_softmax_keeps_empty_rows_at_negative_infinity() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]], dtype=np.float32)
    legal_mask = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.uint8)

    log_probs = masked_log_softmax(logits, legal_mask)

    assert np.isfinite(log_probs[0, 0])
    assert np.isfinite(log_probs[0, 2])
    assert np.isneginf(log_probs[0, 1])
    assert np.all(np.isneginf(log_probs[1]))


def test_empty_legal_guard_returns_pass_payload_for_empty_rows() -> None:
    legal_mask = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.uint8)

    empty_rows, actions, logp, entropy = empty_legal_guard(legal_mask)

    assert np.array_equal(empty_rows, np.array([False, True, True]))
    assert np.array_equal(actions, np.array([PASS_ACTION_ID, PASS_ACTION_ID, PASS_ACTION_ID], dtype=np.int64))
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
    assert np.array_equal(adjusted_actions, np.array([7, PASS_ACTION_ID, 19], dtype=np.uint32))
    assert counters.empty_legal == 1


def test_resolve_pass_action_id_uses_contract_default_when_weiss_sim_is_missing() -> None:
    original_module = sys.modules.pop("weiss_sim", None)
    try:
        assert resolve_pass_action_id() == PASS_ACTION_ID
    finally:
        if original_module is not None:
            sys.modules["weiss_sim"] = original_module


def test_resolve_pass_action_id_rejects_mismatched_weiss_sim_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "weiss_sim", SimpleNamespace(PASS_ACTION_ID=17))

    with pytest.raises(RuntimeError, match="PASS_ACTION_ID mismatch"):
        resolve_pass_action_id()


def test_sample_actions_for_policy_applies_empty_legal_pass_fallback() -> None:
    module = _load_policy_example_module()

    class FakeLegalActions:
        def __init__(self) -> None:
            self.mask = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=np.uint8)
            self.seed: int | None = None

        def sample_uniform(self, *, seed: int) -> np.ndarray:
            self.seed = seed
            return np.array([2, 999, 1], dtype=np.uint32)

    legal_actions = FakeLegalActions()
    counters = MaskingAnomalyCounters()

    actions = module.sample_actions_for_policy(
        policy_name="uniform_legal",
        legal_actions=legal_actions,
        base_seed=10,
        step_index=3,
        counters=counters,
    )

    assert legal_actions.seed == 13
    assert np.array_equal(actions, np.array([2, PASS_ACTION_ID, 1], dtype=np.uint32))
    assert counters.empty_legal == 1
