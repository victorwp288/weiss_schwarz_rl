from __future__ import annotations

import numpy as np
from weiss_rl.core.masking import (
    sample_actions_from_legal_ids,
    sample_actions_from_mask,
    select_argmax_from_legal_ids,
    select_argmax_from_mask,
)

TOY_PASS_ACTION_ID = 4


def test_sample_actions_from_legal_ids_temperature_sharpens_behavior_logp() -> None:
    logits = np.array([[0.0, 2.0]], dtype=np.float32)
    legal_ids = np.array([0, 1], dtype=np.uint32)
    legal_offsets = np.array([0, 2], dtype=np.uint32)

    actions_hot, logp_hot, entropy_hot = sample_actions_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        rng=np.random.default_rng(1),
    )
    actions_cold, logp_cold, entropy_cold = sample_actions_from_legal_ids(
        logits,
        legal_ids,
        legal_offsets,
        rng=np.random.default_rng(1),
        temperature=0.5,
    )

    assert actions_hot.tolist() == [1]
    assert actions_cold.tolist() == [1]
    assert float(logp_cold[0]) > float(logp_hot[0])
    assert float(entropy_cold[0]) < float(entropy_hot[0])


def test_sample_actions_from_mask_temperature_sharpens_behavior_logp() -> None:
    logits = np.array([[0.0, 2.0]], dtype=np.float32)
    legal_mask = np.array([[True, True]], dtype=np.bool_)

    actions_hot, logp_hot, entropy_hot = sample_actions_from_mask(
        logits,
        legal_mask,
        rng=np.random.default_rng(1),
    )
    actions_cold, logp_cold, entropy_cold = sample_actions_from_mask(
        logits,
        legal_mask,
        rng=np.random.default_rng(1),
        temperature=0.5,
    )

    assert actions_hot.tolist() == [1]
    assert actions_cold.tolist() == [1]
    assert float(logp_cold[0]) > float(logp_hot[0])
    assert float(entropy_cold[0]) < float(entropy_hot[0])


def test_select_argmax_from_legal_ids_uses_legal_order_tiebreak_and_pass_fallback() -> None:
    logits = np.array(
        [
            [0.0, 2.0, 2.0, 4.0, 0.0],
            [9.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_ids = np.array([1, 2, 3], dtype=np.uint32)
    legal_offsets = np.array([0, 3, 3], dtype=np.uint32)

    actions = select_argmax_from_legal_ids(logits, legal_ids, legal_offsets, pass_action_id=TOY_PASS_ACTION_ID)

    assert actions.tolist() == [3, TOY_PASS_ACTION_ID]


def test_select_argmax_from_mask_uses_legal_order_tiebreak_and_pass_fallback() -> None:
    logits = np.array(
        [
            [0.0, 4.0, 4.0, 1.0, 0.0],
            [9.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    actions = select_argmax_from_mask(logits, legal_mask, pass_action_id=TOY_PASS_ACTION_ID)

    assert actions.tolist() == [1, TOY_PASS_ACTION_ID]
