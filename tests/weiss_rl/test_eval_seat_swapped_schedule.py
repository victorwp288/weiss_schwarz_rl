from __future__ import annotations

from weiss_rl.eval.harness import build_seat_swapped_schedule


def test_build_seat_swapped_schedule_uses_fixed_seed_pair_order() -> None:
    schedule = build_seat_swapped_schedule(
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        paired_seeds=[11, 22],
    )

    assert [(game.pair_index, game.swap_index, game.episode_seed) for game in schedule] == [
        (0, 0, 11),
        (0, 1, 11),
        (1, 0, 22),
        (1, 1, 22),
    ]
    assert [(game.seat0_policy_id, game.seat1_policy_id, game.focal_seat) for game in schedule] == [
        ("champion", "baseline", 0),
        ("baseline", "champion", 1),
        ("champion", "baseline", 0),
        ("baseline", "champion", 1),
    ]
    assert [(game.seat0_deck, game.seat1_deck) for game in schedule] == [
        ("preset:main_deck_5hy_yotsuba_v1", "preset:main_deck_5hy_yotsuba_v1"),
        ("preset:main_deck_5hy_yotsuba_v1", "preset:main_deck_5hy_yotsuba_v1"),
        ("preset:main_deck_5hy_yotsuba_v1", "preset:main_deck_5hy_yotsuba_v1"),
        ("preset:main_deck_5hy_yotsuba_v1", "preset:main_deck_5hy_yotsuba_v1"),
    ]


def test_build_seat_swapped_schedule_assigns_profile_decks_to_heuristics() -> None:
    schedule = build_seat_swapped_schedule(
        focal_policy_id="policy_000021",
        opponent_policy_id="B3 HeuristicPublicAggro",
        paired_seeds=[11],
    )

    assert [(game.seat0_policy_id, game.seat0_deck, game.seat1_policy_id, game.seat1_deck) for game in schedule] == [
        (
            "policy_000021",
            "preset:main_deck_5hy_yotsuba_v1",
            "B3 HeuristicPublicAggro",
            "preset:aggro_deck_5hy_nino_v1",
        ),
        (
            "B3 HeuristicPublicAggro",
            "preset:aggro_deck_5hy_nino_v1",
            "policy_000021",
            "preset:main_deck_5hy_yotsuba_v1",
        ),
    ]
