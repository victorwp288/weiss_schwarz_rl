"""Deterministic public-only heuristic policy used for the optional B2 anchor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from weiss_rl.action_catalog import ActionCatalog, DecodedAction

_FRONT_ROW_SLOTS = (0, 1, 2)
_BACK_ROW_SLOTS = (3, 4)
_SLOT_PREFERENCE = {
    0: 20,
    1: 30,
    2: 15,
    3: 8,
    4: 6,
}
_SLOT_PREFERENCE_ARRAY = np.asarray([_SLOT_PREFERENCE[index] for index in range(5)], dtype=np.int64)


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_sequence(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return list(value)


def _coerce_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"{context} must be an int-compatible value")
    return int(value)


@dataclass(frozen=True, slots=True)
class StageSlotPublic:
    occupied: bool
    has_attacked: bool
    power: int
    effective_soul: int
    side_attack_allowed: bool


@dataclass(frozen=True, slots=True)
class PublicBoardState:
    self_level_count: int
    self_clock_count: int
    self_stage: tuple[StageSlotPublic, ...]
    opponent_stage: tuple[StageSlotPublic, ...]
    choice_page_start: int
    choice_total: int


@dataclass(frozen=True, slots=True)
class _PlayerObservationLayout:
    level_count_index: int
    clock_count_index: int
    stage_base: int
    stage_slot_width: int
    stage_slot_count: int

    def parse_stage(self, obs_row: np.ndarray) -> tuple[StageSlotPublic, ...]:
        slots: list[StageSlotPublic] = []
        for slot_index in range(self.stage_slot_count):
            base = self.stage_base + slot_index * self.stage_slot_width
            card_id = int(obs_row[base])
            slots.append(
                StageSlotPublic(
                    occupied=card_id != 0,
                    has_attacked=bool(int(obs_row[base + 2])),
                    power=int(obs_row[base + 3]),
                    effective_soul=int(obs_row[base + 5]),
                    side_attack_allowed=bool(int(obs_row[base + 6])),
                )
            )
        return tuple(slots)


@dataclass(frozen=True, slots=True)
class PublicObservationLayout:
    obs_len: int
    choice_page_start_index: int
    choice_total_index: int
    self_player: _PlayerObservationLayout
    opponent_player: _PlayerObservationLayout

    @classmethod
    def from_spec_bundle(cls, spec_bundle: Mapping[str, object], *, stage_slot_count: int) -> PublicObservationLayout:
        observation_spec = _require_mapping(spec_bundle.get("observation"), context="spec_bundle.observation")
        if not bool(observation_spec.get("self_first", False)):
            raise ValueError("B2 HeuristicPublic requires a self-first observation layout")

        header_fields = _require_sequence(
            observation_spec.get("header_fields"),
            context="spec_bundle.observation.header_fields",
        )
        header_indices: dict[str, int] = {}
        for field in header_fields:
            field_mapping = _require_mapping(field, context="spec_bundle.observation.header_fields[]")
            header_indices[str(field_mapping["name"])] = _coerce_int(
                field_mapping["index"],
                context=f"spec_bundle.observation.header_fields[{field_mapping['name']!r}].index",
            )

        player_blocks = _require_sequence(
            observation_spec.get("player_blocks"),
            context="spec_bundle.observation.player_blocks",
        )
        if len(player_blocks) < 2:
            raise ValueError("B2 HeuristicPublic requires two player blocks in the observation spec")

        def parse_player_layout(item: object) -> _PlayerObservationLayout:
            block = _require_mapping(item, context="spec_bundle.observation.player_blocks[]")
            base = _coerce_int(block["base"], context="spec_bundle.observation.player_blocks[].base")
            slices = _require_sequence(
                block.get("slices"),
                context="spec_bundle.observation.player_blocks[].slices",
            )
            slices_by_name = {}
            for slice_item in slices:
                slice_mapping = _require_mapping(
                    slice_item,
                    context="spec_bundle.observation.player_blocks[].slices[]",
                )
                slices_by_name[str(slice_mapping["name"])] = slice_mapping
            stage_slice = _require_mapping(
                slices_by_name["stage"],
                context="spec_bundle.observation.player_blocks[].slices.stage",
            )
            stage_len = _coerce_int(
                stage_slice["len"],
                context="spec_bundle.observation.player_blocks[].slices.stage.len",
            )
            if stage_len % stage_slot_count != 0:
                raise ValueError(
                    f"stage slice length {stage_len} is not divisible by stage slot count {stage_slot_count}"
                )
            return _PlayerObservationLayout(
                level_count_index=base
                + _coerce_int(
                    slices_by_name["level_count"]["start"],
                    context="spec_bundle.observation.player_blocks[].slices.level_count.start",
                ),
                clock_count_index=base
                + _coerce_int(
                    slices_by_name["clock_count"]["start"],
                    context="spec_bundle.observation.player_blocks[].slices.clock_count.start",
                ),
                stage_base=base
                + _coerce_int(
                    stage_slice["start"],
                    context="spec_bundle.observation.player_blocks[].slices.stage.start",
                ),
                stage_slot_width=stage_len // stage_slot_count,
                stage_slot_count=stage_slot_count,
            )

        return cls(
            obs_len=_coerce_int(observation_spec["obs_len"], context="spec_bundle.observation.obs_len"),
            choice_page_start_index=int(header_indices["choice_page_start"]),
            choice_total_index=int(header_indices["choice_total"]),
            self_player=parse_player_layout(player_blocks[0]),
            opponent_player=parse_player_layout(player_blocks[1]),
        )

    def parse_public_board(self, obs_row: np.ndarray) -> PublicBoardState:
        flat_obs = np.asarray(obs_row, dtype=np.int32).reshape(-1)
        if flat_obs.shape[0] < self.obs_len:
            raise ValueError(f"observation row is too short ({flat_obs.shape[0]} < {self.obs_len})")
        return PublicBoardState(
            self_level_count=int(flat_obs[self.self_player.level_count_index]),
            self_clock_count=int(flat_obs[self.self_player.clock_count_index]),
            self_stage=self.self_player.parse_stage(flat_obs),
            opponent_stage=self.opponent_player.parse_stage(flat_obs),
            choice_page_start=int(flat_obs[self.choice_page_start_index]),
            choice_total=int(flat_obs[self.choice_total_index]),
        )


class HeuristicPublicPolicy:
    """Deterministic action selection that only consults public observation features."""

    def __init__(self, *, action_catalog: ActionCatalog, observation_layout: PublicObservationLayout) -> None:
        self._action_catalog = action_catalog
        self._observation_layout = observation_layout
        self._decode_cache: dict[int, DecodedAction] = {}
        self._family_index = {
            family.name: index for index, family in enumerate(self._action_catalog.families)
        }
        self._attack_type_index = {
            str(name): index for index, name in enumerate(self._action_catalog.attack_type_names)
        }
        self._family_ids = {name: int(index) for name, index in self._family_index.items()}
        self._attack_family_id = int(self._family_index.get("attack", -1))
        self._encore_pay_family_id = int(self._family_index.get("encore_pay", -1))
        self._play_family_id = int(self._family_index.get("main_play_character", -1))
        self._climax_family_id = int(self._family_index.get("climax_play", -1))
        self._clock_family_id = int(self._family_index.get("clock_from_hand", -1))
        self._event_family_id = int(self._family_index.get("main_play_event", -1))
        self._choice_select_family_id = int(self._family_index.get("choice_select", -1))
        self._level_up_family_id = int(self._family_index.get("level_up", -1))
        self._trigger_order_family_id = int(self._family_index.get("trigger_order", -1))
        self._mulligan_confirm_family_id = int(self._family_index.get("mulligan_confirm", -1))
        self._move_family_id = int(self._family_index.get("main_move", -1))
        self._next_page_family_id = int(self._family_index.get("choice_next_page", -1))
        self._prev_page_family_id = int(self._family_index.get("choice_prev_page", -1))
        self._pass_family_id = int(self._family_index.get("pass", -1))
        self._mulligan_select_family_id = int(self._family_index.get("mulligan_select", -1))
        self._encore_decline_family_id = int(self._family_index.get("encore_decline", -1))
        self._direct_attack_type_id = int(self._attack_type_index.get("direct", -1))
        self._frontal_attack_type_id = int(self._attack_type_index.get("frontal", -1))
        self._side_attack_type_id = int(self._attack_type_index.get("side", -1))
        self._meta_unused = int(np.iinfo(np.uint16).max)

    @classmethod
    def from_spec_bundle(cls, spec_bundle: Mapping[str, object]) -> HeuristicPublicPolicy:
        action_catalog = ActionCatalog.from_spec_bundle(spec_bundle)
        observation_layout = PublicObservationLayout.from_spec_bundle(
            spec_bundle,
            stage_slot_count=action_catalog.max_stage,
        )
        return cls(action_catalog=action_catalog, observation_layout=observation_layout)

    @property
    def pass_action_id(self) -> int:
        return self._action_catalog.pass_action_id

    def choose_action(self, obs_row: np.ndarray, legal_ids: np.ndarray) -> int:
        if np.asarray(legal_ids).size == 0:
            return self.pass_action_id
        board = self._observation_layout.parse_public_board(obs_row)
        best_action_id = self.pass_action_id
        best_score: tuple[int, ...] | None = None
        for action_id in np.asarray(legal_ids, dtype=np.int64).tolist():
            decoded = self._decode(int(action_id))
            candidate_score = self._score_action(decoded, board) + (-int(action_id),)
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_action_id = int(action_id)
        return best_action_id

    def choose_action_from_meta(
        self,
        obs_row: np.ndarray,
        legal_ids: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> int:
        action_ids = np.asarray(legal_ids, dtype=np.int64).reshape(-1)
        if action_ids.size == 0:
            return self.pass_action_id
        if legal_action_meta is None:
            return self.choose_action(obs_row, action_ids.astype(np.uint32, copy=False))
        meta = np.asarray(legal_action_meta, dtype=np.uint16)
        if meta.ndim != 2 or meta.shape[0] != action_ids.shape[0] or meta.shape[1] < 3:
            return self.choose_action(obs_row, action_ids.astype(np.uint32, copy=False))
        return int(
            self.choose_actions_from_meta_batch(
                np.asarray(obs_row, dtype=np.int32).reshape(1, -1),
                action_ids.astype(np.uint32, copy=False),
                np.asarray([0, action_ids.shape[0]], dtype=np.uint32),
                meta,
            )[0]
        )

    def choose_actions_from_meta_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        legal_offsets: np.ndarray,
        legal_action_meta: np.ndarray | None,
    ) -> np.ndarray:
        obs_batch = np.asarray(obs_rows, dtype=np.int32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        if obs_batch.ndim != 2:
            raise ValueError("obs_rows must have shape (rows, observation)")
        if obs_batch.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        if obs_batch.shape[1] < self._observation_layout.obs_len:
            raise ValueError(
                f"observation rows are too short ({obs_batch.shape[1]} < {self._observation_layout.obs_len})"
            )
        action_ids = np.asarray(legal_ids, dtype=np.int64).reshape(-1)
        offsets = np.asarray(legal_offsets, dtype=np.int64).reshape(-1)
        if offsets.shape != (obs_batch.shape[0] + 1,):
            return self._choose_actions_scalar_batch(obs_batch, action_ids)
        if int(offsets[0]) != 0 or int(offsets[-1]) != int(action_ids.shape[0]) or np.any(np.diff(offsets) < 0):
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)
        if legal_action_meta is None:
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)
        meta = np.asarray(legal_action_meta, dtype=np.uint16)
        if meta.ndim != 2 or meta.shape[0] != action_ids.shape[0] or meta.shape[1] < 3:
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)

        board = self._parse_public_board_batch(obs_batch)
        lengths = offsets[1:] - offsets[:-1]
        if np.any(lengths < 0):
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)
        row_ids = np.repeat(np.arange(obs_batch.shape[0], dtype=np.int64), lengths.astype(np.int64, copy=False))
        if row_ids.shape[0] != action_ids.shape[0]:
            return self._choose_actions_scalar_batch(obs_batch, action_ids, offsets=offsets)

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
                out[valid] = (
                    _slot_preference(valid_slots)
                    + np.maximum(self_power[valid_rows, valid_slots], 0) // 1000
                )
            return out

        attack_mask = family_ids == self._attack_family_id
        if np.any(attack_mask):
            attack_rows = row_ids[attack_mask]
            slots = np.where(arg0[attack_mask] >= 0, arg0[attack_mask], 0)
            attack_types = np.where(arg1[attack_mask] >= 0, arg1[attack_mask], 0)
            type_score = np.zeros(slots.shape, dtype=np.int64)
            if self._direct_attack_type_id >= 0:
                direct = attack_types == self._direct_attack_type_id
                type_score[direct] = np.where(opp_occupied[attack_rows[direct], slots[direct]], 15, 60)
            if self._frontal_attack_type_id >= 0:
                frontal = attack_types == self._frontal_attack_type_id
                type_score[frontal] = np.where(
                    self_power[attack_rows[frontal], slots[frontal]] >= opp_power[attack_rows[frontal], slots[frontal]],
                    45,
                    25,
                )
            if self._side_attack_type_id >= 0:
                side = attack_types == self._side_attack_type_id
                type_score[side] = np.where(
                    self_side_attack_allowed[attack_rows[side], slots[side]],
                    40,
                    5,
                )
            attack_score = (
                type_score
                + _slot_preference(slots)
                + np.maximum(self_soul[attack_rows, slots], 0) * 4
                + np.maximum(self_power[attack_rows, slots], 0) // 1000
            )
            attack_score = np.where(self_occupied[attack_rows, slots], attack_score, -1000)
            score0[attack_mask] = 900
            score1[attack_mask] = attack_score

        encore_pay_mask = family_ids == self._encore_pay_family_id
        if np.any(encore_pay_mask):
            score0[encore_pay_mask] = 700
            score1[encore_pay_mask] = _score_slot_action(row_ids[encore_pay_mask], arg0[encore_pay_mask])

        play_mask = family_ids == self._play_family_id
        if np.any(play_mask):
            play_rows = row_ids[play_mask]
            slots = np.where(arg1[play_mask] >= 0, arg1[play_mask], 0)
            play_score = _slot_preference(slots)
            play_score = play_score + np.where(slots <= 2, 40, np.where(slots <= 4, 20, 0))
            play_score = np.where(self_occupied[play_rows, slots], -1000, play_score)
            score0[play_mask] = 650
            score1[play_mask] = play_score
            score2[play_mask] = _prefer_lower(arg0[play_mask])

        climax_mask = family_ids == self._climax_family_id
        if np.any(climax_mask):
            attackers = np.count_nonzero(self_occupied[:, :3] & ~self_attacked[:, :3], axis=1).astype(np.int64, copy=False)
            defenders = np.count_nonzero(opp_occupied[:, :3], axis=1).astype(np.int64, copy=False)
            climax_rows = row_ids[climax_mask]
            score0[climax_mask] = 550
            score1[climax_mask] = attackers[climax_rows] * 10 + defenders[climax_rows] * 4 + np.where(
                attackers[climax_rows] > 0,
                10,
                -20,
            )
            score2[climax_mask] = _prefer_lower(arg0[climax_mask])

        clock_mask = family_ids == self._clock_family_id
        if np.any(clock_mask):
            clock_rows = row_ids[clock_mask]
            score0[clock_mask] = 500
            score1[clock_mask] = np.where(
                (self_level_count[clock_rows] <= 0) & (self_clock_count[clock_rows] < 6),
                40 - self_clock_count[clock_rows],
                10,
            )
            score2[clock_mask] = _prefer_lower(arg0[clock_mask])

        event_mask = family_ids == self._event_family_id
        if np.any(event_mask):
            score0[event_mask] = 320
            score1[event_mask] = 10
            score2[event_mask] = _prefer_lower(arg0[event_mask])

        choice_select_mask = family_ids == self._choice_select_family_id
        if np.any(choice_select_mask):
            score0[choice_select_mask] = 300
            score1[choice_select_mask] = _prefer_lower(arg0[choice_select_mask])

        level_up_mask = family_ids == self._level_up_family_id
        if np.any(level_up_mask):
            score0[level_up_mask] = 290
            score1[level_up_mask] = _prefer_lower(arg0[level_up_mask])

        trigger_order_mask = family_ids == self._trigger_order_family_id
        if np.any(trigger_order_mask):
            score0[trigger_order_mask] = 280
            score1[trigger_order_mask] = _prefer_lower(arg0[trigger_order_mask])

        mulligan_confirm_mask = family_ids == self._mulligan_confirm_family_id
        if np.any(mulligan_confirm_mask):
            score0[mulligan_confirm_mask] = 260

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
                bonus[(valid_from >= 3) & (valid_to <= 2)] += 30
                bonus[(valid_to == 1) & (valid_from != 1)] += 15
                legal = self_occupied[valid_rows, valid_from] & ~self_occupied[valid_rows, valid_to]
                move_score[valid] = np.where(legal, improvement + bonus, -1000)
            score0[move_mask] = 120
            score1[move_mask] = move_score

        next_page_mask = family_ids == self._next_page_family_id
        if np.any(next_page_mask):
            next_rows = row_ids[next_page_mask]
            score0[next_page_mask] = 170
            score1[next_page_mask] = np.maximum(choice_total[next_rows] - (choice_page_start[next_rows] + 16), 0)

        prev_page_mask = family_ids == self._prev_page_family_id
        if np.any(prev_page_mask):
            prev_rows = row_ids[prev_page_mask]
            score0[prev_page_mask] = 170
            score1[prev_page_mask] = np.maximum(choice_page_start[prev_rows], 0)

        pass_mask = family_ids == self._pass_family_id
        if np.any(pass_mask):
            score0[pass_mask] = 160

        mulligan_select_mask = family_ids == self._mulligan_select_family_id
        if np.any(mulligan_select_mask):
            score0[mulligan_select_mask] = 120
            score1[mulligan_select_mask] = _prefer_lower(arg0[mulligan_select_mask])

        encore_decline_mask = family_ids == self._encore_decline_family_id
        if np.any(encore_decline_mask):
            score0[encore_decline_mask] = 110
            score1[encore_decline_mask] = _score_slot_action(row_ids[encore_decline_mask], arg0[encore_decline_mask])

        chosen_actions = np.full((obs_batch.shape[0],), self.pass_action_id, dtype=np.int64)
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

    def _choose_actions_scalar_batch(
        self,
        obs_rows: np.ndarray,
        legal_ids: np.ndarray,
        *,
        offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        obs_batch = np.asarray(obs_rows, dtype=np.int32)
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        action_ids = np.asarray(legal_ids, dtype=np.uint32).reshape(-1)
        if offsets is None:
            offsets = np.asarray([0, action_ids.shape[0]], dtype=np.int64)
        chosen_actions = np.full((obs_batch.shape[0],), self.pass_action_id, dtype=np.int64)
        for row_index in range(obs_batch.shape[0]):
            start = int(offsets[row_index])
            stop = int(offsets[row_index + 1])
            chosen_actions[row_index] = int(self.choose_action(obs_batch[row_index], action_ids[start:stop]))
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
            "self_level_count": obs_batch[:, self._observation_layout.self_player.level_count_index].astype(np.int64, copy=False),
            "self_clock_count": obs_batch[:, self._observation_layout.self_player.clock_count_index].astype(np.int64, copy=False),
            "choice_page_start": obs_batch[:, self._observation_layout.choice_page_start_index].astype(np.int64, copy=False),
            "choice_total": obs_batch[:, self._observation_layout.choice_total_index].astype(np.int64, copy=False),
            "self_occupied": self_stage["occupied"],
            "self_attacked": self_stage["has_attacked"],
            "self_power": self_stage["power"],
            "self_soul": self_stage["effective_soul"],
            "self_side_attack_allowed": self_stage["side_attack_allowed"],
            "opponent_occupied": opponent_stage["occupied"],
            "opponent_power": opponent_stage["power"],
        }

    def _stage_arrays(
        self,
        obs_batch: np.ndarray,
        layout: _PlayerObservationLayout,
    ) -> dict[str, np.ndarray]:
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
            "occupied": _stage_component(0, dtype=np.int32) != 0,
            "has_attacked": _stage_component(2, dtype=np.int32) != 0,
            "power": _stage_component(3, dtype=np.int64),
            "effective_soul": _stage_component(5, dtype=np.int64),
            "side_attack_allowed": _stage_component(6, dtype=np.int32) != 0,
        }

    def _decode(self, action_id: int) -> DecodedAction:
        cached = self._decode_cache.get(int(action_id))
        if cached is not None:
            return cached
        decoded = self._action_catalog.decode(int(action_id))
        self._decode_cache[int(action_id)] = decoded
        return decoded

    def _score_action(self, action: DecodedAction, board: PublicBoardState) -> tuple[int, int, int, int]:
        family = action.family
        if family == "attack":
            return (900, self._score_attack(action, board), 0, 0)
        if family == "encore_pay":
            return (700, self._score_slot_action(action.slot, board.self_stage), 0, 0)
        if family == "main_play_character":
            return (650, self._score_play_character(action, board), self._prefer_lower(action.hand_index), 0)
        if family == "climax_play":
            return (550, self._score_climax(board), self._prefer_lower(action.hand_index), 0)
        if family == "clock_from_hand":
            return (500, self._score_clock(board), self._prefer_lower(action.hand_index), 0)
        if family == "main_play_event":
            return (320, 10, self._prefer_lower(action.hand_index), 0)
        if family == "choice_select":
            return (300, self._prefer_lower(action.index), 0, 0)
        if family == "level_up":
            return (290, self._prefer_lower(action.index), 0, 0)
        if family == "trigger_order":
            return (280, self._prefer_lower(action.index), 0, 0)
        if family == "mulligan_confirm":
            return (260, 0, 0, 0)
        if family == "main_move":
            return (120, self._score_move(action, board), 0, 0)
        if family == "choice_next_page":
            remaining = max(board.choice_total - (board.choice_page_start + 16), 0)
            return (170, remaining, 0, 0)
        if family == "choice_prev_page":
            return (170, max(board.choice_page_start, 0), 0, 0)
        if family == "pass":
            return (160, 0, 0, 0)
        if family == "mulligan_select":
            return (120, self._prefer_lower(action.hand_index), 0, 0)
        if family == "encore_decline":
            return (110, self._score_slot_action(action.slot, board.self_stage), 0, 0)
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
            type_score = 60 if not defender.occupied else 15
        elif attack_type == "frontal":
            type_score = 45 if attacker.power >= defender.power else 25
        elif attack_type == "side":
            type_score = 40 if attacker.side_attack_allowed else 5
        else:
            type_score = 0
        return (
            type_score
            + _SLOT_PREFERENCE.get(slot, 0)
            + max(attacker.effective_soul, 0) * 4
            + max(attacker.power, 0) // 1000
        )

    def _score_play_character(self, action: DecodedAction, board: PublicBoardState) -> int:
        slot = 0 if action.stage_slot is None else int(action.stage_slot)
        stage = board.self_stage[slot]
        if stage.occupied:
            return -1000
        bonus = _SLOT_PREFERENCE.get(slot, 0)
        if slot in _FRONT_ROW_SLOTS:
            return 40 + bonus
        if slot in _BACK_ROW_SLOTS:
            return 20 + bonus
        return bonus

    def _score_move(self, action: DecodedAction, board: PublicBoardState) -> int:
        if action.from_slot is None or action.to_slot is None:
            return -1000
        origin = board.self_stage[int(action.from_slot)]
        target = board.self_stage[int(action.to_slot)]
        if not origin.occupied or target.occupied:
            return -1000
        improvement = _SLOT_PREFERENCE.get(int(action.to_slot), 0) - _SLOT_PREFERENCE.get(int(action.from_slot), 0)
        bonus = 0
        if int(action.from_slot) in _BACK_ROW_SLOTS and int(action.to_slot) in _FRONT_ROW_SLOTS:
            bonus += 30
        if int(action.to_slot) == 1 and int(action.from_slot) != 1:
            bonus += 15
        return improvement + bonus

    def _score_climax(self, board: PublicBoardState) -> int:
        attackers = sum(
            1
            for slot in _FRONT_ROW_SLOTS
            if board.self_stage[slot].occupied and not board.self_stage[slot].has_attacked
        )
        defenders = sum(1 for slot in _FRONT_ROW_SLOTS if board.opponent_stage[slot].occupied)
        return attackers * 10 + defenders * 4 + (10 if attackers > 0 else -20)

    def _score_clock(self, board: PublicBoardState) -> int:
        if board.self_level_count <= 0 and board.self_clock_count < 6:
            return 40 - board.self_clock_count
        return 10

    def _score_slot_action(self, slot: int | None, stage: tuple[StageSlotPublic, ...]) -> int:
        if slot is None:
            return 0
        slot_index = int(slot)
        slot_state = stage[slot_index]
        return _SLOT_PREFERENCE.get(slot_index, 0) + max(slot_state.power, 0) // 1000

    @staticmethod
    def _prefer_lower(value: int | None) -> int:
        if value is None:
            return 0
        return -int(value)


__all__ = [
    "ActionCatalog",
    "DecodedAction",
    "HeuristicPublicPolicy",
    "PublicBoardState",
    "PublicObservationLayout",
    "StageSlotPublic",
]
