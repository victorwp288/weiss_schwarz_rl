from __future__ import annotations

import numpy as np
import torch
from weiss_rl.eval.search.simulator_god_search_outcomes import build_prefix_replay_failure_detail
from weiss_rl.eval.search.simulator_god_search_rollouts import (
    clone_seat_hidden_map,
    copy_action_sequence_state,
    god_search_rollout_rng_seed,
    god_search_rollout_sampling_algorithm,
)
from weiss_rl.eval.search.simulator_god_search_selection import (
    build_god_search_trace,
    mean_candidate_scores,
    root_logit_by_action,
    select_god_search_candidate,
)
from weiss_rl.eval.simulator.harness import ScheduledGame


def _scheduled_game() -> ScheduledGame:
    return ScheduledGame(
        pair_index=2,
        swap_index=1,
        episode_index=5,
        episode_seed=12345,
        focal_policy_id="policy_a",
        opponent_policy_id="policy_b",
        seat0_policy_id="policy_a",
        seat1_policy_id="policy_b",
        focal_seat=0,
    )


def test_god_search_rollout_sampling_algorithm_maps_policy_names() -> None:
    assert god_search_rollout_sampling_algorithm(
        rollout_policy="argmax",
        eval_sampling_algorithm="pinned_cdf_pcg_v1",
    ) == "model_argmax_pinned_v1"
    assert god_search_rollout_sampling_algorithm(
        rollout_policy="sample",
        eval_sampling_algorithm="model_argmax_pinned_v1",
    ) == "pinned_cdf_pcg_v1"
    assert god_search_rollout_sampling_algorithm(
        rollout_policy="eval",
        eval_sampling_algorithm="model_argmax_pinned_v1",
    ) == "model_argmax_pinned_v1"


def test_god_search_rollout_rng_seed_is_stable_and_decision_sensitive() -> None:
    base = god_search_rollout_rng_seed(
        scheduled_game=_scheduled_game(),
        seat=0,
        candidate_action=7,
        rollout_index=0,
        decision_id=100,
    )
    same = god_search_rollout_rng_seed(
        scheduled_game=_scheduled_game(),
        seat=0,
        candidate_action=7,
        rollout_index=0,
        decision_id=100,
    )
    changed = god_search_rollout_rng_seed(
        scheduled_game=_scheduled_game(),
        seat=0,
        candidate_action=8,
        rollout_index=0,
        decision_id=100,
    )

    assert same == base
    assert changed != base


def test_god_search_candidate_selection_tie_breaks_by_root_logit() -> None:
    candidates = [2, 4]
    root_logits = root_logit_by_action(root_logits=np.asarray([0.0, 0.0, 0.1, 0.0, 0.8]), candidates=candidates)
    mean_scores = mean_candidate_scores({2: [0.5, 0.5], 4: [0.25, 0.75]})

    selected = select_god_search_candidate(
        candidates=candidates,
        mean_scores=mean_scores,
        root_logits=root_logits,
    )

    assert selected == 4


def test_build_god_search_trace_preserves_decision_context() -> None:
    trace = build_god_search_trace(
        scheduled_game=_scheduled_game(),
        decision_id=9,
        current_seat=1,
        current_policy_id="policy_b",
        opponent_policy_id="policy_a",
        base_action=2,
        selected_action=4,
        candidates=[2, 4],
        mean_scores={2: 0.1, 4: 0.2},
        root_logits={2: 0.3, 4: 0.4},
        rollout_details={2: [{"score": 0.1}], 4: [{"score": 0.2}]},
    )

    assert trace["pair_index"] == 2
    assert trace["swap_index"] == 1
    assert trace["actor_seat"] == 1
    assert trace["base_action"] == 2
    assert trace["selected_action"] == 4
    assert trace["candidates"][1]["rollouts"] == [{"score": 0.2}]


def test_prefix_replay_failure_detail_preserves_root_context() -> None:
    detail = build_prefix_replay_failure_detail(
        reason="prefix_root_mismatch",
        scheduled_game=_scheduled_game(),
        root_decision_id=77,
        extra={"observed_decision_id": 78},
    )

    assert detail["status"] == "prefix_replay_failed"
    assert detail["reason"] == "prefix_root_mismatch"
    assert detail["pair_index"] == 2
    assert detail["episode_seed"] == 12345
    assert detail["root_decision_id"] == 77
    assert detail["observed_decision_id"] == 78


def test_clone_seat_hidden_map_replaces_current_seat_without_aliasing() -> None:
    seat0 = torch.ones((1, 2))
    seat1 = torch.full((1, 2), 2.0)
    root_next = torch.full((1, 2), 3.0)

    cloned = clone_seat_hidden_map({0: seat0, 1: seat1}, current_seat=1, root_next_seat_hidden=root_next)

    assert cloned[0] is not seat0
    assert cloned[1] is not root_next
    torch.testing.assert_close(cloned[0], seat0)
    torch.testing.assert_close(cloned[1], root_next)


def test_copy_action_sequence_state_preserves_main_move_counter() -> None:
    source = copy_action_sequence_state(None)
    source.consecutive_main_moves_by_env[0] = 4

    copied = copy_action_sequence_state(source)

    assert copied is not source
    assert int(copied.consecutive_main_moves_by_env[0]) == 4
