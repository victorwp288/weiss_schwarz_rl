from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, replay_trajectory_bc_batch
from weiss_rl.training.auxiliary_replay_metrics import AuxiliaryReplayMetricAccumulator
from weiss_rl.training.auxiliary_replay_runner import (
    AuxiliaryReplayBatchContext,
    AuxiliaryReplayRunResult,
    auxiliary_replay_sampler_is_due,
    auxiliary_replay_update_is_due,
    emit_auxiliary_replay_aux_metrics,
    emit_auxiliary_replay_run_metrics,
    emit_auxiliary_replay_sampled_metrics,
    run_auxiliary_replay_updates,
)
from weiss_rl.training.paired_auxiliary_replay import (
    emit_paired_auxiliary_replay_metrics,
    run_due_paired_auxiliary_replay,
)
from weiss_rl.training.paired_outcome_preference_replay import (
    PairedOutcomePreferenceReplayState,
    maybe_run_paired_outcome_preference_replay,
)
from weiss_rl.training.paired_swing_replay import PairedSwingReplayState, maybe_run_paired_swing_replay
from weiss_rl.training.trajectory_bc_sampling import TrajectoryBcReplayState


def test_auxiliary_replay_due_preserves_positive_modulo_cadence() -> None:
    assert not auxiliary_replay_update_is_due(every_updates=2, update_count=0)
    assert not auxiliary_replay_update_is_due(every_updates=2, update_count=1)
    assert auxiliary_replay_update_is_due(every_updates=2, update_count=2)
    assert auxiliary_replay_update_is_due(every_updates=2, update_count=4)
    sampler = _sampler(["opponent_a"], batch_episodes=1, aux_updates=1, every_updates=2)
    assert not auxiliary_replay_sampler_is_due(sampler, update_count=1)
    assert auxiliary_replay_sampler_is_due(sampler, update_count=2)


def test_auxiliary_replay_emission_helpers_preserve_common_aux_and_precedence() -> None:
    sampler = _sampler(["opponent_a", "opponent_b"], batch_episodes=2, aux_updates=1, every_updates=1)
    sampled_metrics = AuxiliaryReplayMetricAccumulator(sampler)
    sampled_metrics.record_sampled_episodes([0, 1])
    sampled_metrics.record_context_indices(np.asarray([0, 5], dtype=np.int64))
    replay_result = AuxiliaryReplayRunResult(
        aux_metrics={"loss": 0.25, "coef": 0.9, "nan_loss": float("nan"), "label": "ignored"},
        sampled_metrics=sampled_metrics,
    )
    latest = {"example_replay_coef": 0.1}

    emit_auxiliary_replay_sampled_metrics(
        latest,
        prefix="example_replay",
        replay_result=replay_result,
        include_context=True,
    )
    assert latest["example_replay_aux_updates"] == 1.0
    assert latest["example_replay_batch_episodes"] == 2.0
    assert latest["example_replay_opponent_context_episodes"] == 1.0
    assert latest["example_replay_coef"] == 0.1

    latest["example_replay_coef"] = 0.7
    emit_auxiliary_replay_aux_metrics(latest, prefix="example_replay", replay_result=replay_result)
    assert latest["example_replay_loss"] == 0.25
    assert latest["example_replay_coef"] == 0.9
    assert "example_replay_nan_loss" not in latest
    assert "example_replay_label" not in latest

    combined_latest: dict[str, float] = {}
    emit_auxiliary_replay_run_metrics(
        combined_latest,
        prefix="combined_replay",
        replay_result=replay_result,
        include_context=True,
    )
    assert combined_latest["combined_replay_loss"] == 0.25
    assert combined_latest["combined_replay_opponent_context_episodes"] == 1.0


def test_auxiliary_replay_runner_samples_context_batches_and_metrics() -> None:
    dataset = _dataset(["opponent_a", "opponent_b", "opponent_c", "opponent_d"])
    state = TrajectoryBcReplayState(
        dataset=dataset,
        rng=np.random.default_rng(5),
        batch_episodes=2,
        aux_updates=2,
        every_updates=1,
        order=np.asarray([0, 1, 2, 3], dtype=np.int64),
    )
    model = _ContextModel({"opponent_a": 7, "opponent_b": 8, "opponent_c": 9, "opponent_d": 10})
    learner = _Learner(model=model)
    contexts: list[AuxiliaryReplayBatchContext] = []

    def update_batch(batch: dict[str, Any], context: AuxiliaryReplayBatchContext) -> dict[str, float]:
        contexts.append(context)
        assert context.opponent_context_indices is not None
        assert np.asarray(batch["initial_hidden_state"]).shape == (2, 3)
        assert np.asarray(batch["opponent_context_index"]).tolist() == [context.opponent_context_indices.tolist()]
        return {"loss": 0.25}

    result = run_auxiliary_replay_updates(
        sampler=state,
        learner=learner,
        device=torch.device("cpu"),
        update_batch=update_batch,
        use_opponent_context=True,
    )
    latest: dict[str, float] = {}
    result.sampled_metrics.emit_common_metrics(latest, prefix="aux", include_context=True)

    assert [context.episode_indices for context in contexts] == [[0, 1], [2, 3]]
    assert [
        context.opponent_context_indices.tolist()
        for context in contexts
        if context.opponent_context_indices is not None
    ] == [
        [7, 8],
        [9, 10],
    ]
    assert model.initial_contexts == [[7, 8], [9, 10]]
    assert result.aux_metrics == {"loss": 0.25}
    assert latest["aux_aux_updates"] == 2.0
    assert latest["aux_batch_episodes"] == 4.0
    assert latest["aux_opponent_context_episodes"] == 4.0


def test_auxiliary_replay_runner_keeps_plain_replay_batches_context_free() -> None:
    dataset = _dataset(["opponent_a", "opponent_b"])
    state = TrajectoryBcReplayState(
        dataset=dataset,
        rng=np.random.default_rng(5),
        batch_episodes=2,
        aux_updates=1,
        every_updates=1,
        order=np.asarray([0, 1], dtype=np.int64),
    )
    model = _PlainModel()

    def batch_factory(
        dataset: ReplayTrajectoryDataset,
        *,
        episode_indices: list[int],
        initial_hidden_state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        batch = replay_trajectory_bc_batch(
            dataset,
            episode_indices=episode_indices,
            initial_hidden_state=initial_hidden_state,
        )
        assert "opponent_context_index" not in batch
        return batch

    def update_batch(batch: dict[str, Any], context: AuxiliaryReplayBatchContext) -> dict[str, float]:
        assert context.episode_indices == [0, 1]
        assert context.opponent_context_indices is None
        assert np.asarray(batch["initial_hidden_state"]).shape == (2, 2)
        return {"acc": 1.0}

    result = run_auxiliary_replay_updates(
        sampler=state,
        learner=_Learner(model=model),
        device=torch.device("cpu"),
        update_batch=update_batch,
        batch_factory=batch_factory,
        use_opponent_context=False,
    )

    assert model.initial_batch_sizes == [2]
    assert result.aux_metrics == {"acc": 1.0}


def test_run_due_paired_auxiliary_replay_skips_before_due_without_resolving_updater() -> None:
    state = SimpleNamespace(sampler=_sampler(["opponent_a"], batch_episodes=1, aux_updates=1, every_updates=2))

    result = run_due_paired_auxiliary_replay(
        state=state,
        learner=_Learner(model=_PlainModel()),
        device=torch.device("cpu"),
        update_count=1,
        updater_method_name="missing_update",
        updater_error_message="missing updater",
        make_update_batch=lambda _updater: (_ for _ in ()).throw(AssertionError("should not build updater")),
    )

    assert result is None


def test_emit_paired_auxiliary_replay_metrics_preserves_static_then_aux_precedence() -> None:
    sampler = _sampler(["opponent_a"], batch_episodes=1, aux_updates=1, every_updates=1)
    sampled_metrics = AuxiliaryReplayMetricAccumulator(sampler)
    sampled_metrics.record_sampled_episodes([0])
    replay_result = AuxiliaryReplayRunResult(
        aux_metrics={"loss": 0.25, "coef": 0.9},
        sampled_metrics=sampled_metrics,
    )
    latest: dict[str, float] = {}

    emit_paired_auxiliary_replay_metrics(
        latest,
        prefix="paired",
        replay_result=replay_result,
        static_metrics={"paired_coef": 0.1, "paired_static": 2.0},
        include_context=False,
    )

    assert latest["paired_aux_updates"] == 1.0
    assert latest["paired_batch_episodes"] == 1.0
    assert latest["paired_static"] == 2.0
    assert latest["paired_coef"] == 0.9
    assert latest["paired_loss"] == 0.25


def test_paired_swing_replay_preserves_updater_kwargs_context_batch_and_metrics() -> None:
    sampler = _sampler(["opponent_a", "opponent_b"], batch_episodes=2, aux_updates=1, every_updates=2)
    state = PairedSwingReplayState(
        sampler=sampler,
        margin=0.4,
        coef=0.2,
        positive_action_source="teacher_action",
        negative_action_source="actions",
        distinct_train_rows=2,
        loss_scope="label_mean",
        compare_to="top_other",
    )
    learner = _PairedSwingLearner(
        model=_ContextModel({"opponent_a": 3, "opponent_b": 4}),
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )
    assert learner.calls == []
    assert latest_metrics == {}

    maybe_run_paired_swing_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=2,
        latest_metrics=latest_metrics,
    )

    assert len(learner.calls) == 1
    call = learner.calls[0]
    assert call["kwargs"] == {
        "margin": 0.4,
        "coef": 0.2,
        "positive_action_source": "teacher_action",
        "negative_action_source": "actions",
        "loss_scope": "label_mean",
        "compare_to": "top_other",
    }
    assert np.asarray(call["batch"]["opponent_context_index"]).tolist() == [[3, 4]]
    assert latest_metrics["paired_swing_replay_aux_updates"] == 1.0
    assert latest_metrics["paired_swing_replay_batch_episodes"] == 2.0
    assert latest_metrics["paired_swing_replay_opponent_context_episodes"] == 2.0
    assert latest_metrics["paired_swing_replay_dataset_distinct_train_rows"] == 2.0
    assert latest_metrics["paired_swing_replay_margin"] == 0.4
    assert latest_metrics["paired_swing_replay_coef"] == 0.2
    assert latest_metrics["paired_swing_replay_loss_scope_label_mean"] == 1.0
    assert latest_metrics["paired_swing_replay_compare_to_top_other"] == 1.0
    assert latest_metrics["paired_swing_replay_positive_source_teacher"] == 1.0
    assert latest_metrics["paired_swing_replay_paired_swing_loss"] == 0.5


def test_paired_outcome_preference_replay_preserves_groups_kwargs_context_and_metrics() -> None:
    sampler = _sampler(["opponent_b", "opponent_a"], batch_episodes=2, aux_updates=1, every_updates=1)
    state = PairedOutcomePreferenceReplayState(
        sampler=sampler,
        beta=0.3,
        coef=0.7,
        aggregation="sum",
        group_balance=True,
        complete_pair_count=1,
    )
    learner = _PairedOutcomePreferenceLearner(
        model=_ContextModel({"opponent_a": 6, "opponent_b": 5}),
    )
    latest_metrics: dict[str, float] = {}

    maybe_run_paired_outcome_preference_replay(
        state=state,
        learner=learner,
        device=torch.device("cpu"),
        update_count=1,
        latest_metrics=latest_metrics,
    )

    assert len(learner.calls) == 1
    call = learner.calls[0]
    assert call["kwargs"] == {
        "beta": 0.3,
        "coef": 0.7,
        "aggregation": "sum",
        "group_balance": True,
    }
    assert np.asarray(call["batch"]["opponent_context_index"]).tolist() == [[5, 6]]
    assert np.asarray(call["batch"]["preference_group_id"]).tolist() == [[0, 1]]
    assert latest_metrics["paired_outcome_preference_replay_aux_updates"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_batch_episodes"] == 2.0
    assert latest_metrics["paired_outcome_preference_replay_opponent_context_episodes"] == 2.0
    assert latest_metrics["paired_outcome_preference_replay_complete_pair_count"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_beta"] == 0.3
    assert latest_metrics["paired_outcome_preference_replay_coef"] == 0.7
    assert latest_metrics["paired_outcome_preference_replay_aggregation_sum"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_group_balance"] == 1.0
    assert latest_metrics["paired_outcome_preference_replay_preference_loss"] == 0.75


class _ContextModel:
    def __init__(self, policy_index_by_id: dict[str, int]) -> None:
        self.policy_index_by_id = policy_index_by_id
        self.initial_contexts: list[list[int]] = []

    def opponent_context_indices_for_policy_ids(self, policy_ids: list[str]) -> np.ndarray:
        return np.asarray([self.policy_index_by_id[policy_id] for policy_id in policy_ids], dtype=np.int64)

    def initial_seat_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device,
        opponent_context_indices: np.ndarray | None = None,
    ) -> torch.Tensor:
        assert device == torch.device("cpu")
        assert opponent_context_indices is not None
        self.initial_contexts.append(opponent_context_indices.tolist())
        return torch.zeros((int(batch_size), 3), dtype=torch.float32, device=device)


class _PlainModel:
    def __init__(self) -> None:
        self.initial_batch_sizes: list[int] = []

    def initial_seat_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        self.initial_batch_sizes.append(int(batch_size))
        return torch.zeros((int(batch_size), 2), dtype=torch.float32, device=device)


class _Learner:
    def __init__(self, *, model: object) -> None:
        self.model = model


class _PairedSwingLearner:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def paired_swing_update(self, batch: dict[str, Any], **kwargs: object) -> dict[str, float]:
        self.calls.append({"batch": batch, "kwargs": dict(kwargs)})
        return {"paired_swing_loss": 0.5}


class _PairedOutcomePreferenceLearner:
    def __init__(self, *, model: object) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def paired_outcome_preference_update(self, batch: dict[str, Any], **kwargs: object) -> dict[str, float]:
        self.calls.append({"batch": batch, "kwargs": dict(kwargs)})
        return {"preference_loss": 0.75}


def _sampler(
    opponent_policy_ids: list[str],
    *,
    batch_episodes: int,
    aux_updates: int,
    every_updates: int,
) -> TrajectoryBcReplayState:
    return TrajectoryBcReplayState(
        dataset=_dataset(opponent_policy_ids),
        rng=np.random.default_rng(5),
        batch_episodes=batch_episodes,
        aux_updates=aux_updates,
        every_updates=every_updates,
        order=np.arange(len(opponent_policy_ids), dtype=np.int64),
    )


def _dataset(opponent_policy_ids: list[str]) -> ReplayTrajectoryDataset:
    time_steps = 1
    episode_count = len(opponent_policy_ids)
    obs = np.zeros((time_steps, episode_count, 4), dtype=np.float32)
    actor = np.zeros((time_steps, episode_count), dtype=np.int64)
    to_play_seat = np.zeros((time_steps, episode_count), dtype=np.int64)
    actions = np.ones((time_steps, episode_count), dtype=np.int64)
    teacher = np.ones((time_steps, episode_count), dtype=np.int32)
    valid = np.ones((time_steps, episode_count), dtype=np.bool_)
    legal_ids = np.tile(np.asarray([0, 1], dtype=np.uint32), episode_count)
    metadata = {
        "format": "weiss_rl_replay_trajectory_bc_v1",
        "train_rows": episode_count,
        "selected_bundles": [
            {
                "source_dataset_label": f"episode_{index}",
                "source_opponent_policy_id": policy_id,
            }
            for index, policy_id in enumerate(opponent_policy_ids)
        ],
    }
    return ReplayTrajectoryDataset(
        obs=obs,
        actor=actor,
        to_play_seat=to_play_seat,
        actions=actions,
        legal_ids=legal_ids,
        legal_offsets=np.arange(0, (episode_count + 1) * 2, 2, dtype=np.uint32),
        legal_action_meta=np.zeros((episode_count * 2, 3), dtype=np.uint16),
        teacher_family=teacher,
        teacher_slot=teacher,
        teacher_move_source=teacher,
        teacher_attack_type=teacher,
        teacher_action=teacher,
        teacher_valid=valid,
        policy_train_mask=valid,
        reset_before_step=np.zeros((time_steps, episode_count), dtype=np.bool_),
        metadata=metadata,
    )
