"""Shared decoding helpers for exported Weiss Schwarz action spaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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


__all__ = [
    "ActionCatalog",
    "DecodedAction",
]
