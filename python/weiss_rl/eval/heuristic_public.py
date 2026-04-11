"""Deterministic public-only heuristic policy used for the optional B2 anchor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

_FRONT_ROW_SLOTS = (0, 1, 2)
_BACK_ROW_SLOTS = (3, 4)
_SLOT_PREFERENCE = {
    0: 20,
    1: 30,
    2: 15,
    3: 8,
    4: 6,
}


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
class DecodedAction:
    action_id: int
    family: str
    hand_index: int | None = None
    stage_slot: int | None = None
    from_slot: int | None = None
    to_slot: int | None = None
    slot: int | None = None
    attack_type: str | None = None
    index: int | None = None


@dataclass(frozen=True, slots=True)
class _ActionFamily:
    name: str
    base: int
    count: int


@dataclass(frozen=True, slots=True)
class ActionCatalog:
    action_space_size: int
    pass_action_id: int
    max_hand: int
    max_stage: int
    attack_slot_count: int
    attack_type_names: tuple[str, ...]
    families: tuple[_ActionFamily, ...]

    @classmethod
    def from_spec_bundle(cls, spec_bundle: Mapping[str, object]) -> ActionCatalog:
        action_spec = _require_mapping(spec_bundle.get("action"), context="spec_bundle.action")
        constants_raw = _require_sequence(action_spec.get("constants"), context="spec_bundle.action.constants")
        constants: dict[str, int] = {}
        for item in constants_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                constants[str(item[0])] = int(item[1])
        families_raw = _require_sequence(action_spec.get("families"), context="spec_bundle.action.families")
        families = tuple(
            _ActionFamily(
                name=str(_require_mapping(item, context="spec_bundle.action.families[]")["name"]),
                base=_coerce_int(
                    _require_mapping(item, context="spec_bundle.action.families[]")["base"],
                    context="spec_bundle.action.families[].base",
                ),
                count=_coerce_int(
                    _require_mapping(item, context="spec_bundle.action.families[]")["count"],
                    context="spec_bundle.action.families[].count",
                ),
            )
            for item in families_raw
        )
        attack_type_encoding = _require_sequence(
            action_spec.get("attack_type_encoding"),
            context="spec_bundle.action.attack_type_encoding",
        )
        attack_type_names = tuple(str(item[0]) for item in attack_type_encoding if isinstance(item, (list, tuple)))
        if not attack_type_names:
            raise ValueError("spec_bundle.action.attack_type_encoding must contain at least one attack type")
        return cls(
            action_space_size=_coerce_int(
                action_spec["action_space_size"],
                context="spec_bundle.action.action_space_size",
            ),
            pass_action_id=_coerce_int(
                action_spec["pass_action_id"],
                context="spec_bundle.action.pass_action_id",
            ),
            max_hand=int(constants.get("MAX_HAND", 50)),
            max_stage=int(constants.get("MAX_STAGE", 5)),
            attack_slot_count=int(constants.get("ATTACK_SLOT_COUNT", 3)),
            attack_type_names=attack_type_names,
            families=tuple(sorted(families, key=lambda family: family.base)),
        )

    def decode(self, action_id: int) -> DecodedAction:
        action = int(action_id)
        if action < 0 or action >= self.action_space_size:
            raise ValueError(f"action_id {action} is outside action space {self.action_space_size}")
        family = next(
            (family for family in self.families if family.base <= action < family.base + family.count),
            None,
        )
        if family is None:
            raise ValueError(f"Could not decode action_id {action} from exported family ranges")
        offset = action - family.base

        if family.name in {"mulligan_confirm", "pass", "choice_prev_page", "choice_next_page", "concede"}:
            return DecodedAction(action_id=action, family=family.name)
        if family.name in {
            "mulligan_select",
            "clock_from_hand",
            "main_play_event",
            "climax_play",
            "level_up",
            "trigger_order",
            "choice_select",
        }:
            if family.name in {"level_up", "trigger_order", "choice_select"}:
                return DecodedAction(action_id=action, family=family.name, index=offset)
            return DecodedAction(action_id=action, family=family.name, hand_index=offset)
        if family.name == "main_play_character":
            return DecodedAction(
                action_id=action,
                family=family.name,
                hand_index=offset // self.max_stage,
                stage_slot=offset % self.max_stage,
            )
        if family.name == "main_move":
            from_slot = offset // (self.max_stage - 1)
            to_index = offset % (self.max_stage - 1)
            to_slot = to_index if to_index < from_slot else to_index + 1
            return DecodedAction(
                action_id=action,
                family=family.name,
                from_slot=from_slot,
                to_slot=to_slot,
            )
        if family.name == "attack":
            attack_type_index = offset % len(self.attack_type_names)
            return DecodedAction(
                action_id=action,
                family=family.name,
                slot=offset // len(self.attack_type_names),
                attack_type=self.attack_type_names[attack_type_index],
            )
        if family.name in {"encore_pay", "encore_decline"}:
            return DecodedAction(action_id=action, family=family.name, slot=offset)

        raise ValueError(f"Unsupported action family for B2 heuristic: {family.name!r}")


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
