"""Vectorized packed-meta action selection for the public heuristic policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from weiss_rl.eval.heuristic_public_action_scoring import (
    PUBLIC_HEURISTIC_SLOT_PREFERENCE_ARRAY as _SLOT_PREFERENCE_ARRAY,
)
from weiss_rl.eval.heuristic_public_observation import (
    PlayerPublicObservationLayout,
    PublicObservationLayout,
)
from weiss_rl.eval.heuristic_public_profiles import HeuristicPublicScoringProfile


class HeuristicPublicMetaBatchSelector:
    """Selects public-heuristic actions from packed legal-action metadata."""

    def __init__(
        self,
        *,
        observation_layout: PublicObservationLayout,
        scoring_profile: HeuristicPublicScoringProfile,
        pass_action_id: int,
        family_index: Mapping[str, int],
        attack_type_index: Mapping[str, int],
    ) -> None:
        self._observation_layout = observation_layout
        self._scoring_profile = scoring_profile
        self._pass_action_id = int(pass_action_id)
        self._attack_family_id = int(family_index.get("attack", -1))
        self._encore_pay_family_id = int(family_index.get("encore_pay", -1))
        self._play_family_id = int(family_index.get("main_play_character", -1))
        self._climax_family_id = int(family_index.get("climax_play", -1))
        self._clock_family_id = int(family_index.get("clock_from_hand", -1))
        self._event_family_id = int(family_index.get("main_play_event", -1))
        self._choice_select_family_id = int(family_index.get("choice_select", -1))
        self._level_up_family_id = int(family_index.get("level_up", -1))
        self._trigger_order_family_id = int(family_index.get("trigger_order", -1))
        self._mulligan_confirm_family_id = int(family_index.get("mulligan_confirm", -1))
        self._move_family_id = int(family_index.get("main_move", -1))
        self._next_page_family_id = int(family_index.get("choice_next_page", -1))
        self._prev_page_family_id = int(family_index.get("choice_prev_page", -1))
        self._pass_family_id = int(family_index.get("pass", -1))
        self._mulligan_select_family_id = int(family_index.get("mulligan_select", -1))
        self._encore_decline_family_id = int(family_index.get("encore_decline", -1))
        self._direct_attack_type_id = int(attack_type_index.get("direct", -1))
        self._frontal_attack_type_id = int(attack_type_index.get("frontal", -1))
        self._side_attack_type_id = int(attack_type_index.get("side", -1))
        self._meta_unused = int(np.iinfo(np.uint16).max)

    def choose_actions(
        self,
        *,
        obs_batch: np.ndarray,
        action_ids: np.ndarray,
        offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray | None:
        if legal_action_meta is None:
            return None
        meta = np.asarray(legal_action_meta, dtype=np.uint16)
        if meta.ndim != 2 or meta.shape[0] != action_ids.shape[0] or meta.shape[1] < 3:
            return None

        board = self._parse_public_board_batch(obs_batch)
        lengths = offsets[1:] - offsets[:-1]
        if np.any(lengths < 0):
            return None
        row_ids = np.repeat(np.arange(obs_batch.shape[0], dtype=np.int64), lengths.astype(np.int64, copy=False))
        if row_ids.shape[0] != action_ids.shape[0]:
            return None

        family_ids = meta[:, 0].astype(np.int64, copy=False)
        arg0 = meta[:, 1].astype(np.int64, copy=True)
        arg1 = meta[:, 2].astype(np.int64, copy=True)
        arg0[arg0 == self._meta_unused] = -1
        arg1[arg1 == self._meta_unused] = -1

        score0 = np.full((action_ids.shape[0],), -1000, dtype=np.int64)
        score1 = np.zeros((action_ids.shape[0],), dtype=np.int64)
        score2 = np.zeros((action_ids.shape[0],), dtype=np.int64)
        score3 = np.zeros((action_ids.shape[0],), dtype=np.int64)

        self_occupied = board["self_occupied"]
        self_attacked = board["self_attacked"]
        self_power = board["self_power"]
        self_soul = board["self_soul"]
        self_side_attack_allowed = board["self_side_attack_allowed"]
        opp_occupied = board["opponent_occupied"]
        opp_power = board["opponent_power"]
        self_level_count = board["self_level_count"]
        self_clock_count = board["self_clock_count"]
        choice_page_start = board["choice_page_start"]
        choice_total = board["choice_total"]

        def _prefer_lower(values: np.ndarray) -> np.ndarray:
            return np.where(values >= 0, -values, 0).astype(np.int64, copy=False)

        def _slot_preference(values: np.ndarray) -> np.ndarray:
            out = np.zeros(values.shape, dtype=np.int64)
            valid = (values >= 0) & (values < _SLOT_PREFERENCE_ARRAY.shape[0])
            if np.any(valid):
                out[valid] = _SLOT_PREFERENCE_ARRAY[values[valid]]
            return out

        def _score_slot_action(rows: np.ndarray, slots: np.ndarray) -> np.ndarray:
            out = np.zeros(slots.shape, dtype=np.int64)
            valid = (slots >= 0) & (slots < self_power.shape[1])
            if np.any(valid):
                valid_rows = rows[valid]
                valid_slots = slots[valid]
                out[valid] = _slot_preference(valid_slots) + np.maximum(self_power[valid_rows, valid_slots], 0) // 1000
            return out

        profile = self._scoring_profile

        attack_mask = family_ids == self._attack_family_id
        if np.any(attack_mask):
            attack_rows = row_ids[attack_mask]
            slots = np.where(arg0[attack_mask] >= 0, arg0[attack_mask], 0)
            attack_types = np.where(arg1[attack_mask] >= 0, arg1[attack_mask], 0)
            type_score = np.zeros(slots.shape, dtype=np.int64)
            if self._direct_attack_type_id >= 0:
                direct = attack_types == self._direct_attack_type_id
                type_score[direct] = np.where(
                    opp_occupied[attack_rows[direct], slots[direct]],
                    profile.attack_direct_blocked_bonus,
                    profile.attack_direct_open_bonus,
                )
            if self._frontal_attack_type_id >= 0:
                frontal = attack_types == self._frontal_attack_type_id
                type_score[frontal] = np.where(
                    self_power[attack_rows[frontal], slots[frontal]] >= opp_power[attack_rows[frontal], slots[frontal]],
                    profile.attack_frontal_win_bonus,
                    profile.attack_frontal_loss_bonus,
                )
            if self._side_attack_type_id >= 0:
                side = attack_types == self._side_attack_type_id
                type_score[side] = np.where(
                    self_side_attack_allowed[attack_rows[side], slots[side]],
                    profile.attack_side_allowed_bonus,
                    profile.attack_side_blocked_bonus,
                )
            attack_score = (
                type_score
                + _slot_preference(slots)
                + np.maximum(self_soul[attack_rows, slots], 0) * profile.attack_soul_scale
                + np.maximum(self_power[attack_rows, slots], 0) // 1000
            )
            attack_score = np.where(self_occupied[attack_rows, slots], attack_score, -1000)
            score0[attack_mask] = profile.attack_priority
            score1[attack_mask] = attack_score

        encore_pay_mask = family_ids == self._encore_pay_family_id
        if np.any(encore_pay_mask):
            score0[encore_pay_mask] = profile.encore_pay_priority
            score1[encore_pay_mask] = _score_slot_action(row_ids[encore_pay_mask], arg0[encore_pay_mask])

        play_mask = family_ids == self._play_family_id
        if np.any(play_mask):
            play_rows = row_ids[play_mask]
            slots = np.where(arg1[play_mask] >= 0, arg1[play_mask], 0)
            play_score = _slot_preference(slots)
            play_score = play_score + np.where(
                slots <= 2,
                profile.play_front_bonus,
                np.where(slots <= 4, profile.play_back_bonus, 0),
            )
            play_score = np.where(self_occupied[play_rows, slots], -1000, play_score)
            score0[play_mask] = profile.play_priority
            score1[play_mask] = play_score
            score2[play_mask] = _prefer_lower(arg0[play_mask])

        climax_mask = family_ids == self._climax_family_id
        if np.any(climax_mask):
            attackers = np.count_nonzero(self_occupied[:, :3] & ~self_attacked[:, :3], axis=1).astype(
                np.int64, copy=False
            )
            defenders = np.count_nonzero(opp_occupied[:, :3], axis=1).astype(np.int64, copy=False)
            climax_rows = row_ids[climax_mask]
            score0[climax_mask] = profile.climax_priority
            score1[climax_mask] = (
                attackers[climax_rows] * profile.climax_attacker_scale
                + defenders[climax_rows] * profile.climax_defender_scale
                + np.where(
                    attackers[climax_rows] > 0,
                    profile.climax_active_bonus,
                    profile.climax_inactive_bonus,
                )
            )
            score2[climax_mask] = _prefer_lower(arg0[climax_mask])

        clock_mask = family_ids == self._clock_family_id
        if np.any(clock_mask):
            clock_rows = row_ids[clock_mask]
            score0[clock_mask] = profile.clock_priority
            score1[clock_mask] = np.where(
                (self_level_count[clock_rows] <= 0) & (self_clock_count[clock_rows] < 6),
                profile.early_clock_score - self_clock_count[clock_rows],
                profile.late_clock_score,
            )
            score2[clock_mask] = _prefer_lower(arg0[clock_mask])

        event_mask = family_ids == self._event_family_id
        if np.any(event_mask):
            score0[event_mask] = profile.event_priority
            score1[event_mask] = 10
            score2[event_mask] = _prefer_lower(arg0[event_mask])

        choice_select_mask = family_ids == self._choice_select_family_id
        if np.any(choice_select_mask):
            score0[choice_select_mask] = profile.choice_select_priority
            score1[choice_select_mask] = _prefer_lower(arg0[choice_select_mask])

        level_up_mask = family_ids == self._level_up_family_id
        if np.any(level_up_mask):
            score0[level_up_mask] = profile.level_up_priority
            score1[level_up_mask] = _prefer_lower(arg0[level_up_mask])

        trigger_order_mask = family_ids == self._trigger_order_family_id
        if np.any(trigger_order_mask):
            score0[trigger_order_mask] = profile.trigger_order_priority
            score1[trigger_order_mask] = _prefer_lower(arg0[trigger_order_mask])

        mulligan_confirm_mask = family_ids == self._mulligan_confirm_family_id
        if np.any(mulligan_confirm_mask):
            score0[mulligan_confirm_mask] = profile.mulligan_confirm_priority

        move_mask = family_ids == self._move_family_id
        if np.any(move_mask):
            move_rows = row_ids[move_mask]
            from_slots = arg0[move_mask]
            to_slots = arg1[move_mask]
            move_score = np.full(from_slots.shape, -1000, dtype=np.int64)
            valid = (
                (from_slots >= 0)
                & (from_slots < self_occupied.shape[1])
                & (to_slots >= 0)
                & (to_slots < self_occupied.shape[1])
            )
            if np.any(valid):
                valid_rows = move_rows[valid]
                valid_from = from_slots[valid]
                valid_to = to_slots[valid]
                improvement = _slot_preference(valid_to) - _slot_preference(valid_from)
                bonus = np.zeros(valid_to.shape, dtype=np.int64)
                bonus[(valid_from >= 3) & (valid_to <= 2)] += profile.move_back_to_front_bonus
                bonus[(valid_to == 1) & (valid_from != 1)] += profile.move_center_bonus
                legal = self_occupied[valid_rows, valid_from] & ~self_occupied[valid_rows, valid_to]
                move_score[valid] = np.where(legal, improvement + bonus, -1000)
            # Aggressive/control profiles intentionally like good repositioning, but a neutral
            # or bad move must not outrank pass or the policy can main_move forever.
            score0[move_mask] = np.where(
                move_score > 0,
                profile.move_priority,
                min(profile.move_priority, profile.pass_priority - 1),
            )
            score1[move_mask] = move_score

        next_page_mask = family_ids == self._next_page_family_id
        if np.any(next_page_mask):
            next_rows = row_ids[next_page_mask]
            score0[next_page_mask] = profile.pager_priority
            score1[next_page_mask] = np.maximum(choice_total[next_rows] - (choice_page_start[next_rows] + 16), 0)

        prev_page_mask = family_ids == self._prev_page_family_id
        if np.any(prev_page_mask):
            prev_rows = row_ids[prev_page_mask]
            score0[prev_page_mask] = profile.pager_priority
            score1[prev_page_mask] = np.maximum(choice_page_start[prev_rows], 0)

        pass_mask = family_ids == self._pass_family_id
        if np.any(pass_mask):
            score0[pass_mask] = profile.pass_priority

        mulligan_select_mask = family_ids == self._mulligan_select_family_id
        if np.any(mulligan_select_mask):
            score0[mulligan_select_mask] = profile.mulligan_select_priority
            score1[mulligan_select_mask] = _prefer_lower(arg0[mulligan_select_mask])

        encore_decline_mask = family_ids == self._encore_decline_family_id
        if np.any(encore_decline_mask):
            score0[encore_decline_mask] = profile.encore_decline_priority
            score1[encore_decline_mask] = _score_slot_action(row_ids[encore_decline_mask], arg0[encore_decline_mask])

        chosen_actions = np.full((obs_batch.shape[0],), self._pass_action_id, dtype=np.int64)
        for row_index in range(obs_batch.shape[0]):
            start = int(offsets[row_index])
            stop = int(offsets[row_index + 1])
            if stop <= start:
                continue
            order = np.lexsort(
                (
                    action_ids[start:stop],
                    -score3[start:stop],
                    -score2[start:stop],
                    -score1[start:stop],
                    -score0[start:stop],
                )
            )
            chosen_actions[row_index] = int(action_ids[start:stop][int(order[0])])
        return chosen_actions

    def _parse_public_board_batch(self, obs_rows: np.ndarray) -> dict[str, np.ndarray]:
        obs_batch = np.asarray(obs_rows, dtype=np.int32)
        if obs_batch.ndim != 2:
            raise ValueError("obs_rows must have shape (rows, observation)")
        if obs_batch.shape[1] < self._observation_layout.obs_len:
            raise ValueError(
                f"observation rows are too short ({obs_batch.shape[1]} < {self._observation_layout.obs_len})"
            )
        self_stage = self._stage_arrays(obs_batch, self._observation_layout.self_player)
        opponent_stage = self._stage_arrays(obs_batch, self._observation_layout.opponent_player)
        return {
            "self_level_count": obs_batch[:, self._observation_layout.self_player.level_count_index].astype(
                np.int64, copy=False
            ),
            "self_clock_count": obs_batch[:, self._observation_layout.self_player.clock_count_index].astype(
                np.int64, copy=False
            ),
            "choice_page_start": obs_batch[:, self._observation_layout.choice_page_start_index].astype(
                np.int64, copy=False
            ),
            "choice_total": obs_batch[:, self._observation_layout.choice_total_index].astype(np.int64, copy=False),
            "self_occupied": self_stage["occupied"],
            "self_attacked": self_stage["has_attacked"],
            "self_power": self_stage["power"],
            "self_soul": self_stage["effective_soul"],
            "self_side_attack_allowed": self_stage["side_attack_allowed"],
            "opponent_occupied": opponent_stage["occupied"],
            "opponent_power": opponent_stage["power"],
        }

    @staticmethod
    def _stage_arrays(obs_batch: np.ndarray, layout: PlayerPublicObservationLayout) -> dict[str, np.ndarray]:
        width = int(layout.stage_slot_width)
        count = int(layout.stage_slot_count)
        stage_values = np.asarray(
            obs_batch[:, layout.stage_base : layout.stage_base + width * count],
            dtype=np.int32,
        ).reshape(obs_batch.shape[0], count, width)

        def _stage_component(offset: int, *, dtype: np.dtype[Any]) -> np.ndarray:
            if offset >= stage_values.shape[2]:
                return np.zeros(stage_values.shape[:2], dtype=dtype)
            return stage_values[:, :, offset].astype(dtype, copy=False)

        return {
            "occupied": _stage_component(0, dtype=np.dtype(np.int32)) != 0,
            "has_attacked": _stage_component(2, dtype=np.dtype(np.int32)) != 0,
            "power": _stage_component(3, dtype=np.dtype(np.int64)),
            "effective_soul": _stage_component(5, dtype=np.dtype(np.int64)),
            "side_attack_allowed": _stage_component(6, dtype=np.dtype(np.int32)) != 0,
        }


__all__ = [
    "HeuristicPublicMetaBatchSelector",
]
