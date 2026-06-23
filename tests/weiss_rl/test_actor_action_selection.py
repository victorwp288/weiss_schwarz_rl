from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from weiss_rl.actors.action_selection import select_legal_ids_actions, select_mask_actions
from weiss_rl.core.masking import MaskingAnomalyCounters, resolve_pass_action_id


def test_select_legal_ids_actions_prepares_unroll_and_replay_legality() -> None:
    batch = SimpleNamespace(
        ids_offsets=SimpleNamespace(
            legal_ids=np.asarray([1, 2, 3, 4, 99], dtype=np.int32),
            offsets=np.asarray([0, 2, 4], dtype=np.uint32),
        )
    )
    logits = np.zeros((2, 100), dtype=np.float32)

    selection = select_legal_ids_actions(
        batch=batch,
        logits=logits,
        rng=np.random.default_rng(0),
        counters=MaskingAnomalyCounters(),
        pass_action_id=resolve_pass_action_id(),
        offset_base=5,
    )

    assert selection.legal_ids is not None
    assert selection.legal_offsets is not None
    assert selection.unroll_legal_ids is not None
    assert selection.unroll_legal_offsets is not None
    assert selection.legal_ids.tolist() == [1, 2, 3, 4, 99]
    assert selection.unroll_legal_ids.tolist() == [1, 2, 3, 4]
    assert selection.unroll_legal_offsets.tolist() == [7, 9]
    assert [legal_slice.tolist() for legal_slice in selection.replay_legal_slices] == [[1, 2], [3, 4]]
    assert selection.actions.shape == (2,)
    assert selection.logp.shape == (2,)
    assert selection.entropy.shape == (2,)


def test_select_mask_actions_returns_mask_snapshot_and_samples() -> None:
    legal_mask = np.asarray([[True, False, True], [False, False, False]], dtype=bool)
    batch = SimpleNamespace(legal_mask=legal_mask)
    logits = np.zeros((2, 3), dtype=np.float32)
    counters = MaskingAnomalyCounters()

    selection = select_mask_actions(
        batch=batch,
        logits=logits,
        action_space=3,
        rng=np.random.default_rng(1),
        counters=counters,
        pass_action_id=resolve_pass_action_id(),
    )

    assert selection.legal_mask is not None
    assert np.array_equal(selection.legal_mask, legal_mask)
    assert counters.empty_legal == 1
    assert selection.actions.shape == (2,)
    assert selection.logp.shape == (2,)
    assert selection.entropy.shape == (2,)


def test_select_mask_actions_rejects_wrong_action_width() -> None:
    batch = SimpleNamespace(legal_mask=np.ones((2, 2), dtype=bool))
    logits = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=r"expected legal_mask shape"):
        select_mask_actions(
            batch=batch,
            logits=logits,
            action_space=3,
            rng=np.random.default_rng(2),
            counters=MaskingAnomalyCounters(),
            pass_action_id=resolve_pass_action_id(),
        )
