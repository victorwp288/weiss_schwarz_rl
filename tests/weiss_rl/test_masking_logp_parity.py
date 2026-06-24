from __future__ import annotations

import numpy as np
from weiss_rl.actors.actor_worker import actor_behavior_logp_from_legal_ids
from weiss_rl.core.masking import masked_logp_from_legal_ids, masked_logp_from_mask
from weiss_rl.eval.simulator.harness import eval_sampler_logp_from_mask
from weiss_rl.learners.action_logp import learner_logp_from_mask

TOY_PASS_ACTION_ID = 4


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
