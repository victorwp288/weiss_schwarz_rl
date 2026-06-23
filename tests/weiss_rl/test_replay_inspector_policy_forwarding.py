from __future__ import annotations

import numpy as np
import pytest
import torch
from weiss_rl.replay.inspection_policy_execution import forward_policy
from weiss_rl.replay.inspection_policy_loading import LoadedReplayPolicy, opponent_context_index_for_policy

from .replay_inspector_test_support import _ids_batch


def test_replay_forward_policy_uses_packed_candidate_scoring_for_structured_models() -> None:
    class PackedOnlyModel:
        supports_legal_candidate_scoring = True
        supports_factorized_legal_policy = False

        def forward_packed_seat_aware(
            self,
            _obs,
            _seat,
            hidden,
            *,
            legal_actions,
            scoring_mode,
            opponent_context_index=None,
        ):
            assert scoring_mode == "learner"
            assert opponent_context_index is None
            assert np.asarray(legal_actions.ids).tolist() == [2, 5]
            return torch.tensor([4.0, 9.0]), torch.tensor([0.0]), hidden + 1.0

        def forward_seat_aware(self, *_args, **_kwargs):
            raise AssertionError("replay inspector must not use dense scoring for structured models")

    policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=PackedOnlyModel(),  # type: ignore[arg-type]
    )
    batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([2, 5], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    logits, next_hidden = forward_policy(
        policy=policy,
        batch=batch,
        seat_hidden=torch.zeros((1,)),
        legal_ids=np.array([2, 5], dtype=np.uint32),
    )

    assert logits[2] == pytest.approx(4.0)
    assert logits[5] == pytest.approx(9.0)
    assert next_hidden is not None
    assert float(next_hidden.item()) == pytest.approx(1.0)


def test_replay_forward_policy_passes_opponent_context_index_to_packed_scoring() -> None:
    class PackedOnlyModel:
        supports_legal_candidate_scoring = True
        supports_factorized_legal_policy = False

        def forward_packed_seat_aware(
            self,
            _obs,
            _seat,
            hidden,
            *,
            legal_actions,
            scoring_mode,
            opponent_context_index=None,
        ):
            assert scoring_mode == "learner"
            assert opponent_context_index is not None
            assert opponent_context_index.detach().cpu().tolist() == [3]
            assert np.asarray(legal_actions.ids).tolist() == [2, 5]
            return torch.tensor([4.0, 9.0]), torch.tensor([0.0]), hidden + 1.0

        def forward_seat_aware(self, *_args, **_kwargs):
            raise AssertionError("replay inspector must not use dense scoring for structured models")

    policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=PackedOnlyModel(),  # type: ignore[arg-type]
    )
    batch = _ids_batch(
        decision_id=10,
        actor=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        engine_status=0,
        legal_ids=np.array([2, 5], dtype=np.uint16),
        episode_seed=44,
        episode_key=555,
    )
    logits, next_hidden = forward_policy(
        policy=policy,
        batch=batch,
        seat_hidden=torch.zeros((1,)),
        legal_ids=np.array([2, 5], dtype=np.uint32),
        opponent_context_index=3,
    )

    assert logits[2] == pytest.approx(4.0)
    assert logits[5] == pytest.approx(9.0)
    assert next_hidden is not None
    assert float(next_hidden.item()) == pytest.approx(1.0)


def test_opponent_context_index_for_policy_can_require_nonzero() -> None:
    class FakeModel:
        def opponent_context_indices_for_policy_ids(self, policy_ids):
            return [7 if str(policy_ids[0]) == "known_policy" else 0]

    policy = LoadedReplayPolicy(
        spec="policy_a",
        label="policy_a",
        kind="model",
        weights_path=None,
        model=FakeModel(),  # type: ignore[arg-type]
    )

    assert (
        opponent_context_index_for_policy(
            policy=policy,
            opponent_context_policy_id="known_policy",
            require_nonzero=True,
        )
        == 7
    )
    with pytest.raises(RuntimeError, match="has no opponent-context index"):
        opponent_context_index_for_policy(
            policy=policy,
            opponent_context_policy_id="missing_policy",
            require_nonzero=True,
        )
