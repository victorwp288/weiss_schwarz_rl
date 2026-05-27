from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from weiss_rl.eval.model_sampling import model_eval_logits_for_legal_ids


class _PackedOnlyEvalModel:
    supports_legal_candidate_scoring = True
    supports_factorized_legal_policy = False
    action_catalog = None

    def forward_packed_seat_aware(
        self,
        obs,
        acting_seat,
        seat_hidden,
        *,
        legal_actions,
        scoring_mode,
        opponent_context_index=None,
    ):
        assert scoring_mode == "learner"
        assert opponent_context_index is None or opponent_context_index.tolist() == [2]
        assert tuple(obs.shape) == (1, 4)
        assert acting_seat.tolist() == [1]
        assert legal_actions.ids.tolist() == [1, 3]
        assert legal_actions.offsets.tolist() == [0, 2]
        return torch.tensor([0.25, 2.5], dtype=torch.float32), torch.zeros((1,)), seat_hidden + 1.0

    def forward_seat_aware(self, *_args, **_kwargs):
        raise AssertionError("structured eval must use packed legal-candidate scoring")


class _FactorizedEvalModel(_PackedOnlyEvalModel):
    supports_factorized_legal_policy = True

    def factorized_packed_action_log_probs_seat_aware(
        self, obs, acting_seat, seat_hidden, *, legal_actions, scoring_mode, opponent_context_index=None
    ):
        assert scoring_mode == "learner"
        assert opponent_context_index is None or opponent_context_index.tolist() == [2]
        assert tuple(obs.shape) == (1, 4)
        assert acting_seat.tolist() == [0]
        assert legal_actions.ids.tolist() == [1, 3]
        assert legal_actions.offsets.tolist() == [0, 2]
        return torch.log(torch.tensor([0.75, 0.25], dtype=torch.float32)), torch.zeros((1,)), seat_hidden + 3.0

    def forward_packed_seat_aware(self, *_args, **_kwargs):
        raise AssertionError("factorized eval must use factorized packed action log-probs")


class _DenseEvalModel:
    supports_legal_candidate_scoring = False

    def forward_seat_aware(self, obs, acting_seat, seat_hidden, *, scoring_mode, opponent_context_index=None):
        assert scoring_mode == "learner"
        assert opponent_context_index is None
        assert tuple(obs.shape) == (1, 4)
        assert acting_seat.tolist() == [0]
        return torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32), torch.zeros((1,)), seat_hidden + 2.0


def test_model_eval_logits_for_legal_ids_uses_packed_structured_surface() -> None:
    batch = SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.float32),
        ids_offsets=(np.asarray([1, 3], dtype=np.uint32), np.asarray([0, 2], dtype=np.uint32)),
        legal_action_meta=np.asarray([[10, 0, 0, 0], [20, 0, 0, 0]], dtype=np.uint16),
    )
    hidden = torch.zeros((1, 2, 4), dtype=torch.float32)

    logits, next_hidden = model_eval_logits_for_legal_ids(
        model=_PackedOnlyEvalModel(),
        batch=batch,
        current_seat=1,
        seat_hidden=hidden,
        legal_ids=np.asarray([1, 3], dtype=np.uint32),
        action_dim=5,
        device=torch.device("cpu"),
        opponent_context_index=2,
    )

    assert logits.tolist() == [0.0, 0.25, 0.0, 2.5, 0.0]
    torch.testing.assert_close(next_hidden, hidden + 1.0)


def test_model_eval_logits_for_legal_ids_keeps_dense_models_on_dense_surface() -> None:
    batch = SimpleNamespace(obs=np.zeros((1, 4), dtype=np.float32), ids_offsets=None, legal_action_meta=None)
    hidden = torch.zeros((1, 2, 4), dtype=torch.float32)

    logits, next_hidden = model_eval_logits_for_legal_ids(
        model=_DenseEvalModel(),
        batch=batch,
        current_seat=0,
        seat_hidden=hidden,
        legal_ids=np.asarray([0, 2], dtype=np.uint32),
        action_dim=4,
        device=torch.device("cpu"),
    )

    assert logits.tolist() == [0.0, 1.0, 2.0, 3.0]
    torch.testing.assert_close(next_hidden, hidden + 2.0)


def test_model_eval_logits_for_legal_ids_supports_factorized_policy_surface() -> None:
    batch = SimpleNamespace(
        obs=np.zeros((1, 4), dtype=np.float32),
        ids_offsets=(np.asarray([1, 3], dtype=np.uint32), np.asarray([0, 2], dtype=np.uint32)),
        legal_action_meta=None,
    )
    hidden = torch.zeros((1, 2, 4), dtype=torch.float32)

    logits, next_hidden = model_eval_logits_for_legal_ids(
        model=_FactorizedEvalModel(),
        batch=batch,
        current_seat=0,
        seat_hidden=hidden,
        legal_ids=np.asarray([1, 3], dtype=np.uint32),
        action_dim=5,
        device=torch.device("cpu"),
        opponent_context_index=2,
    )

    assert np.isneginf(logits[0])
    np.testing.assert_allclose(
        logits[[1, 3]],
        np.log(np.asarray([0.75, 0.25], dtype=np.float32)),
        atol=1e-7,
        rtol=1e-7,
    )
    assert np.isneginf(logits[[2, 4]]).all()
    torch.testing.assert_close(next_hidden, hidden + 3.0)
