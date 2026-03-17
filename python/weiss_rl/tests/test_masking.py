from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from weiss_rl.actors.actor_worker import actor_behavior_logp_from_legal_ids
from weiss_rl.eval.harness import eval_sampler_logp_from_mask
from weiss_rl.learners.impala_learner import learner_logp_from_mask
from weiss_rl.masking import (
    PASS_ACTION_ID as CONTRACT_PASS_ACTION_ID,
    MaskingAnomalyCounters,
    apply_empty_legal_action_fallback,
    assert_strictly_increasing_legal_ids,
    empty_legal_guard,
    masked_log_softmax,
    masked_logp_from_legal_ids,
    masked_logp_from_mask,
    resolve_pass_action_id,
)

TOY_PASS_ACTION_ID = 4


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


def test_masked_log_softmax_avoids_nan_for_fully_masked_rows() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 0.0, -1.0],
            [3.0, -5.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    log_probs = masked_log_softmax(logits, legal_mask)

    expected_row0 = np.array(
        [1.0 - math.log(math.exp(1.0) + math.exp(0.0)), -np.inf, -math.log(math.exp(1.0) + math.exp(0.0)), -np.inf],
        dtype=np.float32,
    )
    assert np.allclose(log_probs[0], expected_row0, atol=1e-6, equal_nan=False)
    assert np.all(np.isneginf(log_probs[1]))
    assert not np.any(np.isnan(log_probs[1]))
    assert np.allclose(log_probs[2], np.array([-np.inf, 0.0, -np.inf, -np.inf], dtype=np.float32), atol=1e-6)


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
    assert np.array_equal(actions, np.array([2, CONTRACT_PASS_ACTION_ID, 1], dtype=np.uint32))
    assert counters.empty_legal == 1


def test_masked_logp_from_mask_supports_pass_fallback_and_single_legal_action() -> None:
    logits = np.array(
        [
            [1.0, 0.0, -1.0, 2.0, 3.0],
            [0.0, -2.0, 0.5, 1.5, -3.0],
            [-1.0, 4.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    actions = np.array([0, TOY_PASS_ACTION_ID, 1], dtype=np.int64)

    logp = masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=TOY_PASS_ACTION_ID)

    expected = np.array(
        [
            1.0 - math.log(math.exp(1.0) + math.exp(-1.0)),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(logp, expected, atol=1e-6)


def test_masked_logp_from_mask_rejects_illegal_action() -> None:
    logits = np.array([[2.0, 1.0, 0.0, -3.0]], dtype=np.float32)
    legal_mask = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    actions = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="illegal action 1 for row 0"):
        masked_logp_from_mask(logits, legal_mask, actions)


def test_masked_logp_from_mask_requires_pass_action_id_for_empty_rows() -> None:
    logits = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    legal_mask = np.zeros((1, 5), dtype=np.uint8)
    actions = np.array([TOY_PASS_ACTION_ID], dtype=np.int64)

    with pytest.raises(ValueError, match="pass_action_id is required"):
        masked_logp_from_mask(logits, legal_mask, actions)


def test_masked_logp_from_legal_ids_rejects_illegal_action() -> None:
    logits = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    legal_ids = np.array([0, 2, 4], dtype=np.uint32)
    legal_offsets = np.array([0, 3], dtype=np.int32)
    actions = np.array([3], dtype=np.int64)

    with pytest.raises(ValueError, match="illegal action 3 for row 0"):
        masked_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions)


def test_masked_logp_from_legal_ids_rejects_non_finite_legal_logits() -> None:
    logits = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]], dtype=np.float32)
    legal_ids = np.array([1, 3], dtype=np.uint32)
    legal_offsets = np.array([0, 2], dtype=np.int32)
    actions = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="legal logits must be finite for row 0"):
        masked_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions)


def test_masked_logp_from_legal_ids_rejects_malformed_offsets() -> None:
    logits = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    legal_ids = np.array([0, 2], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 1], dtype=np.int32)
    actions = np.array([0, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="legal_offsets must be nondecreasing"):
        masked_logp_from_legal_ids(logits, legal_ids, legal_offsets, actions)


def test_masked_logp_from_legal_ids_supports_pass_fallback() -> None:
    logits = np.array(
        [
            [0.5, -1.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [3.0, 0.0, -2.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )
    legal_ids = np.array([0, 2, 3], dtype=np.uint32)
    legal_offsets = np.array([0, 2, 2, 3], dtype=np.int32)
    actions = np.array([2, TOY_PASS_ACTION_ID, 3], dtype=np.int64)

    logp = masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=TOY_PASS_ACTION_ID,
    )

    expected = np.array(
        [
            2.0 - math.log(math.exp(0.5) + math.exp(2.0)),
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    assert np.allclose(logp, expected, atol=1e-6)


def test_mask_and_packed_legal_ids_produce_matching_logp() -> None:
    logits = np.array(
        [
            [2.0, -1.0, 0.0, 3.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.5, 1.0, -4.0, 0.0, 2.0],
            [-3.0, -1.0, 5.0, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    actions = np.array([3, TOY_PASS_ACTION_ID, 1, 2], dtype=np.int64)
    legal_ids, legal_offsets = _packed_legal_ids_from_mask(legal_mask)

    from_mask = masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=TOY_PASS_ACTION_ID)
    from_ids = masked_logp_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        actions,
        pass_action_id=TOY_PASS_ACTION_ID,
    )

    assert np.allclose(from_mask, from_ids, atol=1e-6)


def test_masked_logp_from_legal_ids_requires_pass_action_on_empty_slice() -> None:
    logits = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    legal_ids = np.array([], dtype=np.uint32)
    legal_offsets = np.array([0, 0], dtype=np.int32)
    actions = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError, match="expected pass action"):
        masked_logp_from_legal_ids(
            logits,
            legal_ids,
            legal_offsets,
            actions,
            pass_action_id=TOY_PASS_ACTION_ID,
        )


def test_masking_core_is_reused_by_actor_eval_and_learner_hooks() -> None:
    logits = np.array(
        [
            [2.0, -1.0, 0.0, 3.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    actions = np.array([3, TOY_PASS_ACTION_ID], dtype=np.int64)
    legal_ids, legal_offsets = _packed_legal_ids_from_mask(legal_mask)

    expected = masked_logp_from_mask(logits, legal_mask, actions, pass_action_id=TOY_PASS_ACTION_ID)
    assert np.allclose(
        eval_sampler_logp_from_mask(logits, legal_mask, actions, pass_action_id=TOY_PASS_ACTION_ID),
        expected,
    )
    assert np.allclose(
        learner_logp_from_mask(logits, legal_mask, actions, pass_action_id=TOY_PASS_ACTION_ID),
        expected,
    )
    assert np.allclose(
        actor_behavior_logp_from_legal_ids(
            logits,
            legal_ids,
            legal_offsets,
            actions,
            pass_action_id=TOY_PASS_ACTION_ID,
        ),
        expected,
    )


def _packed_legal_ids_from_mask(legal_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    packed_ids: list[int] = []
    offsets = [0]
    for row in np.asarray(legal_mask, dtype=bool):
        packed_ids.extend(np.flatnonzero(row).tolist())
        offsets.append(len(packed_ids))
    return np.asarray(packed_ids, dtype=np.uint32), np.asarray(offsets, dtype=np.int32)
