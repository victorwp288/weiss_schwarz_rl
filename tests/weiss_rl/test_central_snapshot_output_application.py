from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from weiss_rl.runtime.components.opponents.central_snapshot_opponents import (
    CentralSnapshotModelOutputs,
    apply_central_snapshot_outputs,
)

from tests.weiss_rl.central_snapshot_opponents_test_support import make_central_opponent_entry


def test_apply_central_snapshot_outputs_sample_writes_raw_logits_values_and_hidden() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((3, 2), dtype=torch.float32))
    logits_out = np.zeros((3, 4), dtype=np.float32)
    values_out = np.zeros((3,), dtype=np.float32)
    entry = make_central_opponent_entry(
        actor=actor,
        batch=SimpleNamespace(ids_offsets=None, mask=np.ones((3, 4), dtype=np.bool_)),
        row_indices=[2, 0],
        obs_step=np.zeros((3, 2), dtype=np.float32),
        actor_step=np.zeros((3,), dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )
    outputs = CentralSnapshotModelOutputs(
        logits=np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        values=np.asarray([9.0, 10.0], dtype=np.float32),
        next_hidden=torch.tensor([[11.0, 12.0], [13.0, 14.0]], dtype=torch.float32),
    )

    apply_central_snapshot_outputs(
        entries=[entry],
        outputs=outputs,
        action_selection="sample",
        pass_action_id=0,
        action_dim=4,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert logits_out[2].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert logits_out[0].tolist() == [5.0, 6.0, 7.0, 8.0]
    assert values_out.tolist() == [10.0, 0.0, 9.0]
    assert actor.opponent_hidden[2].tolist() == [11.0, 12.0]
    assert actor.opponent_hidden[0].tolist() == [13.0, 14.0]


def test_apply_central_snapshot_outputs_argmax_rewrites_packed_logits() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((3, 2), dtype=torch.float32))
    logits_out = np.full((3, 5), -3.0, dtype=np.float32)
    values_out = np.zeros((3,), dtype=np.float32)
    entry = make_central_opponent_entry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=(
                np.asarray([1, 3, 4, 0, 2], dtype=np.uint32),
                np.asarray([0, 2, 3, 5], dtype=np.uint32),
            ),
            legal_action_meta=None,
            mask=None,
        ),
        row_indices=[0, 2],
        obs_step=np.zeros((3, 2), dtype=np.float32),
        actor_step=np.zeros((3,), dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )
    outputs = CentralSnapshotModelOutputs(
        logits=np.asarray(
            [
                [0.0, 3.0, 1.0, 2.0, 4.0],
                [4.0, 0.0, 5.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        values=np.asarray([6.0, 7.0], dtype=np.float32),
        next_hidden=torch.ones((2, 2), dtype=torch.float32),
    )

    apply_central_snapshot_outputs(
        entries=[entry],
        outputs=outputs,
        action_selection="argmax",
        pass_action_id=0,
        action_dim=5,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert logits_out[0, 1] == pytest.approx(0.0)
    assert logits_out[0, 3] == pytest.approx(-100.0)
    assert logits_out[0, 2] < -1.0e8
    assert logits_out[2, 2] == pytest.approx(0.0)
    assert logits_out[2, 0] == pytest.approx(-100.0)
    assert values_out.tolist() == [6.0, 0.0, 7.0]


def test_apply_central_snapshot_outputs_argmax_rewrites_mask_logits() -> None:
    actor = SimpleNamespace(opponent_hidden=torch.zeros((2, 2), dtype=torch.float32))
    logits_out = np.full((2, 5), -3.0, dtype=np.float32)
    values_out = np.zeros((2,), dtype=np.float32)
    entry = make_central_opponent_entry(
        actor=actor,
        batch=SimpleNamespace(
            ids_offsets=None,
            mask=np.asarray(
                [
                    [True, False, True, False, False],
                    [False, True, False, True, False],
                ],
                dtype=np.bool_,
            ),
        ),
        row_indices=[1],
        obs_step=np.zeros((2, 2), dtype=np.float32),
        actor_step=np.zeros((2,), dtype=np.int64),
        logits_out=logits_out,
        values_out=values_out,
    )
    outputs = CentralSnapshotModelOutputs(
        logits=np.asarray([[0.0, 2.0, 1.0, 3.0, 4.0]], dtype=np.float32),
        values=np.asarray([8.0], dtype=np.float32),
        next_hidden=torch.ones((1, 2), dtype=torch.float32),
    )

    apply_central_snapshot_outputs(
        entries=[entry],
        outputs=outputs,
        action_selection="argmax",
        pass_action_id=0,
        action_dim=5,
        ensure_legal_action_meta=lambda _ids, meta: meta,
    )

    assert logits_out[1, 3] == pytest.approx(0.0)
    assert logits_out[1, 1] == pytest.approx(-100.0)
    assert logits_out[1, 4] < -1.0e8
    assert values_out.tolist() == [0.0, 8.0]
