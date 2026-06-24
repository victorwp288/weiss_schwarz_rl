from __future__ import annotations

from weiss_rl.artifacts.reproducibility import key256_to_hex, resolve_episode_key256, stable_hash64
from weiss_rl.eval.simulator.harness import GameResult, ScheduledGame, resolve_eval_episode_key

from tests.weiss_rl.eval_harness_test_support import _RUN_ID256


def test_episode_key_prefers_simulator_value_and_falls_back_deterministically() -> None:
    scheduled_game = ScheduledGame(
        pair_index=0,
        swap_index=1,
        episode_index=3,
        episode_seed=123,
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        seat0_policy_id="baseline",
        seat1_policy_id="champion",
        focal_seat=1,
    )

    with_simulator_key = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=GameResult(
            episode_seed=123,
            terminated=True,
            truncated=False,
            winner_seat=1,
            simulator_episode_key=777,
        ),
        run_id256=_RUN_ID256,
    )
    expected = key256_to_hex(
        resolve_episode_key256(
            simulator_episode_key=777,
            run_id256=bytes.fromhex(_RUN_ID256),
            actor_id=stable_hash64(b"champion") & ((1 << 32) - 1),
            env_id=stable_hash64(b"champion\0baseline") & ((1 << 32) - 1),
            episode_index=3,
            episode_seed64=123,
        )
    )
    assert with_simulator_key == expected

    without_simulator_key_a = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=GameResult(
            episode_seed=123,
            terminated=True,
            truncated=False,
            winner_seat=1,
            simulator_episode_key=None,
        ),
        run_id256=_RUN_ID256,
    )
    without_simulator_key_b = resolve_eval_episode_key(
        scheduled_game=scheduled_game,
        result=GameResult(
            episode_seed=123,
            terminated=True,
            truncated=False,
            winner_seat=1,
            simulator_episode_key=None,
        ),
        run_id256=_RUN_ID256,
    )
    assert without_simulator_key_a == without_simulator_key_b
    assert without_simulator_key_a != with_simulator_key


def test_episode_key_fallback_includes_explicit_decks() -> None:
    common = {
        "pair_index": 0,
        "swap_index": 0,
        "episode_index": 0,
        "episode_seed": 123,
        "focal_policy_id": "policy_000021",
        "opponent_policy_id": "B3 HeuristicPublicAggro",
        "seat0_policy_id": "policy_000021",
        "seat1_policy_id": "B3 HeuristicPublicAggro",
        "focal_seat": 0,
    }
    main_vs_aggro = ScheduledGame(
        **common,
        seat0_deck="preset:main_deck_5hy_yotsuba_v1",
        seat1_deck="preset:aggro_deck_5hy_nino_v1",
    )
    main_vs_main = ScheduledGame(
        **common,
        seat0_deck="preset:main_deck_5hy_yotsuba_v1",
        seat1_deck="preset:main_deck_5hy_yotsuba_v1",
    )
    result = GameResult(
        episode_seed=123,
        terminated=True,
        truncated=False,
        winner_seat=0,
        simulator_episode_key=None,
    )

    assert resolve_eval_episode_key(
        scheduled_game=main_vs_aggro, result=result, run_id256=_RUN_ID256
    ) != resolve_eval_episode_key(scheduled_game=main_vs_main, result=result, run_id256=_RUN_ID256)
