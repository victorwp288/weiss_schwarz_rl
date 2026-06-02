"""Deck listing helpers for human play setup screens."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from weiss_rl.eval.policies.set import AGGRO_DECK_ID, CONTROL_DECK_ID, MAIN_DECK_ID, STARTER_DECK_ID

_PRESET_PREFIX = "preset:"
_ROLE_BY_PRESET = {
    MAIN_DECK_ID.removeprefix(_PRESET_PREFIX): "primary thesis deck",
    AGGRO_DECK_ID.removeprefix(_PRESET_PREFIX): "B3 aggro deck",
    CONTROL_DECK_ID.removeprefix(_PRESET_PREFIX): "B4 control deck",
    STARTER_DECK_ID.removeprefix(_PRESET_PREFIX): "starter/debug deck",
}
_DISPLAY_NAME_BY_PRESET = {
    MAIN_DECK_ID.removeprefix(_PRESET_PREFIX): "Yotsuba thesis deck",
    AGGRO_DECK_ID.removeprefix(_PRESET_PREFIX): "Nino aggro deck",
    CONTROL_DECK_ID.removeprefix(_PRESET_PREFIX): "JoJo control deck",
    STARTER_DECK_ID.removeprefix(_PRESET_PREFIX): "Starter deck",
}


@dataclass(frozen=True, slots=True)
class DeckSummary:
    deck_id: str
    preset_name: str
    label: str
    role: str
    card_count: int
    unique_card_count: int
    sample_cards: tuple[str, ...]
    source: str | None = None
    min_rules_profile: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "preset_name": self.preset_name,
            "label": self.label,
            "role": self.role,
            "card_count": int(self.card_count),
            "unique_card_count": int(self.unique_card_count),
            "sample_cards": list(self.sample_cards),
            "source": self.source,
            "min_rules_profile": self.min_rules_profile,
        }


def preset_id(name: str) -> str:
    normalized = str(name).strip()
    if normalized.startswith(_PRESET_PREFIX):
        return normalized
    return f"{_PRESET_PREFIX}{normalized}"


def preset_name(deck_id: str) -> str:
    normalized = str(deck_id).strip()
    if normalized.startswith(_PRESET_PREFIX):
        return normalized.removeprefix(_PRESET_PREFIX)
    return normalized


def list_deck_presets(weiss_sim: Any) -> list[DeckSummary]:
    cards = getattr(weiss_sim, "cards", None)
    if cards is None or not callable(getattr(cards, "presets", None)):
        raise RuntimeError("weiss_sim.cards.presets() is required for human-play deck selection")

    metadata_raw = {}
    preset_metadata = getattr(cards, "preset_metadata", None)
    if callable(preset_metadata):
        loaded = preset_metadata()
        if isinstance(loaded, Mapping):
            metadata_raw = dict(loaded)

    summaries: list[DeckSummary] = []
    for name in sorted(str(item) for item in cards.presets()):
        meta = metadata_raw.get(name, {})
        meta_mapping = meta if isinstance(meta, Mapping) else {}
        description = _describe_deck(cards, name)
        count_rows = description.get("counts", []) if isinstance(description, Mapping) else []
        card_rows = description.get("cards", []) if isinstance(description, Mapping) else []
        card_count = _count_cards(count_rows=count_rows, card_rows=card_rows)
        unique_card_count = _count_unique_cards(count_rows=count_rows, card_rows=card_rows)
        sample_cards = _sample_card_names(count_rows=count_rows, card_rows=card_rows)
        summaries.append(
            DeckSummary(
                deck_id=preset_id(name),
                preset_name=name,
                label=_DISPLAY_NAME_BY_PRESET.get(name, _title_from_preset(name)),
                role=_ROLE_BY_PRESET.get(name, "freeplay deck"),
                card_count=card_count,
                unique_card_count=unique_card_count,
                sample_cards=tuple(sample_cards),
                source=_optional_str(meta_mapping.get("source")),
                min_rules_profile=_optional_str(meta_mapping.get("min_rules_profile")),
            )
        )
    return summaries


def _describe_deck(cards: Any, name: str) -> Mapping[str, Any]:
    describe_deck = getattr(cards, "describe_deck", None)
    if not callable(describe_deck):
        return {}
    payload = describe_deck(name, rules_profile="approx", card_pool="all")
    return payload if isinstance(payload, Mapping) else {}


def _count_cards(*, count_rows: Any, card_rows: Any) -> int:
    if isinstance(count_rows, list) and count_rows:
        total = 0
        for row in count_rows:
            if isinstance(row, Mapping):
                try:
                    total += int(row.get("count", 1))
                except (TypeError, ValueError):
                    total += 1
        if total > 0:
            return total
    return len(card_rows) if isinstance(card_rows, list) else 0


def _count_unique_cards(*, count_rows: Any, card_rows: Any) -> int:
    rows = count_rows if isinstance(count_rows, list) and count_rows else card_rows
    seen: set[str] = set()
    if not isinstance(rows, list):
        return 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        identity = row.get("id", row.get("card_id", row.get("card_no", row.get("name"))))
        if identity is not None:
            seen.add(str(identity))
    return len(seen)


def _sample_card_names(*, count_rows: Any, card_rows: Any) -> list[str]:
    rows = count_rows if isinstance(count_rows, list) and count_rows else card_rows
    names: list[str] = []
    if not isinstance(rows, list):
        return names
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or row.get("card_no") or row.get("id") or "").strip()
        if name and name not in names:
            names.append(name)
        if len(names) >= 4:
            break
    return names


def _title_from_preset(name: str) -> str:
    words = [word for word in str(name).replace("_v1", "").split("_") if word]
    return " ".join(word.capitalize() for word in words)


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
