"""Scalar action scoring for the public-only heuristic policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weiss_rl.core.action_catalog import DecodedAction
from weiss_rl.eval.heuristic_public.heuristic_public_observation import PublicBoardState, StageSlotPublic
from weiss_rl.public_heuristic.profiles import HeuristicPublicScoringProfile

PUBLIC_HEURISTIC_FRONT_ROW_SLOTS = (0, 1, 2)
PUBLIC_HEURISTIC_BACK_ROW_SLOTS = (3, 4)
PUBLIC_HEURISTIC_SLOT_PREFERENCE = {
    0: 20,
    1: 30,
    2: 15,
    3: 8,
    4: 6,
}
PUBLIC_HEURISTIC_SLOT_PREFERENCE_ARRAY = np.asarray(
    [PUBLIC_HEURISTIC_SLOT_PREFERENCE[index] for index in range(5)],
    dtype=np.int64,
)


@dataclass(frozen=True, slots=True)
class HeuristicPublicActionScorer:
    """Scores decoded actions using only public board state."""

    profile: HeuristicPublicScoringProfile

    def score_action(self, action: DecodedAction, board: PublicBoardState) -> tuple[int, int, int, int]:
        family = action.family
        if family == "attack":
            return (self.profile.attack_priority, self._score_attack(action, board), 0, 0)
        if family == "encore_pay":
            return (
                self.profile.encore_pay_priority,
                self._score_slot_action(action.slot, board.self_stage),
                0,
                0,
            )
        if family == "main_play_character":
            return (
                self.profile.play_priority,
                self._score_play_character(action, board),
                _prefer_lower(action.hand_index),
                0,
            )
        if family == "climax_play":
            return (self.profile.climax_priority, self._score_climax(board), _prefer_lower(action.hand_index), 0)
        if family == "clock_from_hand":
            return (self.profile.clock_priority, self._score_clock(board), _prefer_lower(action.hand_index), 0)
        if family == "main_play_event":
            return (self.profile.event_priority, 10, _prefer_lower(action.hand_index), 0)
        if family == "choice_select":
            return (self.profile.choice_select_priority, _prefer_lower(action.index), 0, 0)
        if family == "level_up":
            return (self.profile.level_up_priority, _prefer_lower(action.index), 0, 0)
        if family == "trigger_order":
            return (self.profile.trigger_order_priority, _prefer_lower(action.index), 0, 0)
        if family == "mulligan_confirm":
            return (self.profile.mulligan_confirm_priority, 0, 0, 0)
        if family == "main_move":
            move_score = self._score_move(action, board)
            move_priority = (
                self.profile.move_priority
                if move_score > 0
                else min(self.profile.move_priority, self.profile.pass_priority - 1)
            )
            return (move_priority, move_score, 0, 0)
        if family == "choice_next_page":
            remaining = max(board.choice_total - (board.choice_page_start + 16), 0)
            return (self.profile.pager_priority, remaining, 0, 0)
        if family == "choice_prev_page":
            return (self.profile.pager_priority, max(board.choice_page_start, 0), 0, 0)
        if family == "pass":
            return (self.profile.pass_priority, 0, 0, 0)
        if family == "mulligan_select":
            return (self.profile.mulligan_select_priority, _prefer_lower(action.hand_index), 0, 0)
        if family == "encore_decline":
            return (
                self.profile.encore_decline_priority,
                self._score_slot_action(action.slot, board.self_stage),
                0,
                0,
            )
        if family == "concede":
            return (-1000, 0, 0, 0)
        raise RuntimeError(f"Unhandled B2 heuristic action family: {family!r}")

    def _score_attack(self, action: DecodedAction, board: PublicBoardState) -> int:
        slot = 0 if action.slot is None else int(action.slot)
        attacker = board.self_stage[slot]
        defender = board.opponent_stage[slot]
        if not attacker.occupied:
            return -1000
        attack_type = action.attack_type or "frontal"
        if attack_type == "direct":
            type_score = (
                self.profile.attack_direct_open_bonus
                if not defender.occupied
                else self.profile.attack_direct_blocked_bonus
            )
        elif attack_type == "frontal":
            type_score = (
                self.profile.attack_frontal_win_bonus
                if attacker.power >= defender.power
                else self.profile.attack_frontal_loss_bonus
            )
        elif attack_type == "side":
            type_score = (
                self.profile.attack_side_allowed_bonus
                if attacker.side_attack_allowed
                else self.profile.attack_side_blocked_bonus
            )
        else:
            type_score = 0
        return (
            type_score
            + PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(slot, 0)
            + max(attacker.effective_soul, 0) * self.profile.attack_soul_scale
            + max(attacker.power, 0) // 1000
        )

    def _score_play_character(self, action: DecodedAction, board: PublicBoardState) -> int:
        slot = 0 if action.stage_slot is None else int(action.stage_slot)
        stage = board.self_stage[slot]
        if stage.occupied:
            return -1000
        bonus = PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(slot, 0)
        if slot in PUBLIC_HEURISTIC_FRONT_ROW_SLOTS:
            return self.profile.play_front_bonus + bonus
        if slot in PUBLIC_HEURISTIC_BACK_ROW_SLOTS:
            return self.profile.play_back_bonus + bonus
        return bonus

    def _score_move(self, action: DecodedAction, board: PublicBoardState) -> int:
        if action.from_slot is None or action.to_slot is None:
            return -1000
        origin = board.self_stage[int(action.from_slot)]
        target = board.self_stage[int(action.to_slot)]
        if not origin.occupied or target.occupied:
            return -1000
        improvement = PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(
            int(action.to_slot),
            0,
        ) - PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(int(action.from_slot), 0)
        bonus = 0
        if (
            int(action.from_slot) in PUBLIC_HEURISTIC_BACK_ROW_SLOTS
            and int(action.to_slot) in PUBLIC_HEURISTIC_FRONT_ROW_SLOTS
        ):
            bonus += self.profile.move_back_to_front_bonus
        if int(action.to_slot) == 1 and int(action.from_slot) != 1:
            bonus += self.profile.move_center_bonus
        return improvement + bonus

    def _score_climax(self, board: PublicBoardState) -> int:
        attackers = sum(
            1
            for slot in PUBLIC_HEURISTIC_FRONT_ROW_SLOTS
            if board.self_stage[slot].occupied and not board.self_stage[slot].has_attacked
        )
        defenders = sum(1 for slot in PUBLIC_HEURISTIC_FRONT_ROW_SLOTS if board.opponent_stage[slot].occupied)
        return (
            attackers * self.profile.climax_attacker_scale
            + defenders * self.profile.climax_defender_scale
            + (self.profile.climax_active_bonus if attackers > 0 else self.profile.climax_inactive_bonus)
        )

    def _score_clock(self, board: PublicBoardState) -> int:
        if board.self_level_count <= 0 and board.self_clock_count < 6:
            return self.profile.early_clock_score - board.self_clock_count
        return self.profile.late_clock_score

    @staticmethod
    def _score_slot_action(slot: int | None, stage: tuple[StageSlotPublic, ...]) -> int:
        if slot is None:
            return 0
        slot_index = int(slot)
        slot_state = stage[slot_index]
        return PUBLIC_HEURISTIC_SLOT_PREFERENCE.get(slot_index, 0) + max(slot_state.power, 0) // 1000


def _prefer_lower(value: int | None) -> int:
    if value is None:
        return 0
    return -int(value)


__all__ = [
    "HeuristicPublicActionScorer",
    "PUBLIC_HEURISTIC_BACK_ROW_SLOTS",
    "PUBLIC_HEURISTIC_FRONT_ROW_SLOTS",
    "PUBLIC_HEURISTIC_SLOT_PREFERENCE_ARRAY",
]
