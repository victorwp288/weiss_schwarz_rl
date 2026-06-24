"""Named public-heuristic scoring profiles shared by eval, learners, and models."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class HeuristicPublicScoringProfile:
    name: str
    attack_priority: int = 900
    encore_pay_priority: int = 700
    play_priority: int = 650
    climax_priority: int = 550
    clock_priority: int = 500
    event_priority: int = 320
    choice_select_priority: int = 300
    level_up_priority: int = 290
    trigger_order_priority: int = 280
    mulligan_confirm_priority: int = 260
    move_priority: int = 120
    pager_priority: int = 170
    pass_priority: int = 160
    mulligan_select_priority: int = 120
    encore_decline_priority: int = 110
    attack_direct_open_bonus: int = 60
    attack_direct_blocked_bonus: int = 15
    attack_frontal_win_bonus: int = 45
    attack_frontal_loss_bonus: int = 25
    attack_side_allowed_bonus: int = 40
    attack_side_blocked_bonus: int = 5
    attack_soul_scale: int = 4
    play_front_bonus: int = 40
    play_back_bonus: int = 20
    move_back_to_front_bonus: int = 30
    move_center_bonus: int = 15
    climax_attacker_scale: int = 10
    climax_defender_scale: int = 4
    climax_active_bonus: int = 10
    climax_inactive_bonus: int = -20
    early_clock_score: int = 40
    late_clock_score: int = 10


_BASE_HEURISTIC_PUBLIC_SCORING_PROFILE = HeuristicPublicScoringProfile(name="base")
_HEURISTIC_PUBLIC_SCORING_PROFILES = {
    "base": _BASE_HEURISTIC_PUBLIC_SCORING_PROFILE,
    "aggressive": replace(
        _BASE_HEURISTIC_PUBLIC_SCORING_PROFILE,
        name="aggressive",
        attack_priority=940,
        climax_priority=610,
        move_priority=210,
        pass_priority=115,
        attack_direct_open_bonus=85,
        attack_direct_blocked_bonus=42,
        attack_frontal_win_bonus=40,
        attack_frontal_loss_bonus=12,
        attack_side_allowed_bonus=18,
        attack_side_blocked_bonus=-10,
        attack_soul_scale=7,
        play_front_bonus=60,
        play_back_bonus=6,
        move_back_to_front_bonus=48,
        move_center_bonus=28,
        climax_attacker_scale=16,
        climax_defender_scale=8,
        climax_active_bonus=18,
        climax_inactive_bonus=-32,
        early_clock_score=18,
        late_clock_score=4,
    ),
    "control": replace(
        _BASE_HEURISTIC_PUBLIC_SCORING_PROFILE,
        name="control",
        attack_priority=870,
        play_priority=680,
        climax_priority=505,
        move_priority=195,
        pass_priority=185,
        attack_direct_open_bonus=38,
        attack_direct_blocked_bonus=0,
        attack_frontal_win_bonus=58,
        attack_frontal_loss_bonus=35,
        attack_side_allowed_bonus=52,
        attack_side_blocked_bonus=0,
        attack_soul_scale=2,
        play_front_bonus=22,
        play_back_bonus=38,
        move_back_to_front_bonus=18,
        move_center_bonus=6,
        climax_attacker_scale=6,
        climax_defender_scale=2,
        climax_active_bonus=6,
        climax_inactive_bonus=-8,
        early_clock_score=48,
        late_clock_score=14,
    ),
}

SUPPORTED_PUBLIC_HEURISTIC_PROFILES = frozenset(_HEURISTIC_PUBLIC_SCORING_PROFILES)


def heuristic_public_scoring_profile(name: str) -> HeuristicPublicScoringProfile:
    profile = _HEURISTIC_PUBLIC_SCORING_PROFILES.get(str(name).strip().lower())
    if profile is None:
        supported = ", ".join(sorted(_HEURISTIC_PUBLIC_SCORING_PROFILES))
        raise ValueError(f"unknown heuristic public profile {name!r}; expected one of: {supported}")
    return profile


__all__ = [
    "HeuristicPublicScoringProfile",
    "SUPPORTED_PUBLIC_HEURISTIC_PROFILES",
    "heuristic_public_scoring_profile",
]
