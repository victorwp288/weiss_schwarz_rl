"""Bundle selection and teacher override manifests for replay BC extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.replay.bundles import load_replay_bundle

_PAIR_SWAP_RE = re.compile(r"_pair(?P<pair>\d+)_swap(?P<swap>\d+)")


@dataclass(frozen=True, slots=True)
class BundleSelection:
    bundle_path: Path
    pair_index: int | None
    swap_index: int | None
    focal_seat: int
    outcome: str | None
    episode_seed: int | None


def load_teacher_action_overrides_jsonl(path: Path) -> dict[tuple[str, int], int]:
    """Load bundle/step teacher-action overrides from a JSONL manifest."""

    overrides: dict[tuple[str, int], int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"teacher-action override rows must be objects: {path}:{line_number}")
            raw_bundle_path = payload.get("bundle_path")
            raw_bundle_name = payload.get("bundle_name")
            if not isinstance(raw_bundle_path, str) and not isinstance(raw_bundle_name, str):
                raise ValueError(f"override row must include bundle_path or bundle_name: {path}:{line_number}")
            step_index = int(payload["step_index"])
            teacher_action = int(payload["teacher_action"])
            keys: list[tuple[str, int]] = []
            if isinstance(raw_bundle_path, str) and raw_bundle_path:
                bundle_path = Path(raw_bundle_path)
                keys.append((bundle_path.resolve().as_posix(), step_index))
                keys.append((bundle_path.name, step_index))
            if isinstance(raw_bundle_name, str) and raw_bundle_name:
                keys.append((raw_bundle_name, step_index))
            for key in keys:
                previous = overrides.get(key)
                if previous is not None and int(previous) != teacher_action:
                    raise ValueError(f"conflicting teacher-action override for {key}: {previous} vs {teacher_action}")
                overrides[key] = teacher_action
    if not overrides:
        raise ValueError(f"no teacher-action overrides found in {path}")
    return overrides


def load_episode_records(path: Path | None) -> dict[tuple[int, int], Mapping[str, Any]]:
    if path is None:
        return {}
    records: dict[tuple[int, int], Mapping[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"episodes_jsonl rows must be objects: {path}")
            key = (int(payload["pair_index"]), int(payload["swap_index"]))
            records[key] = payload
    return records


def select_bundles(
    bundle_paths: Sequence[Path],
    *,
    episode_records: Mapping[tuple[int, int], Mapping[str, Any]],
    include_outcomes: tuple[str, ...],
    focal_seat: int | None,
) -> list[BundleSelection]:
    allowed_outcomes = {str(item).strip().upper() for item in include_outcomes if str(item).strip()}
    selections: list[BundleSelection] = []
    for bundle_path in bundle_paths:
        pair_swap = _pair_swap_from_bundle_path(bundle_path)
        record = None if pair_swap is None else episode_records.get(pair_swap)
        outcome = None if record is None else str(record.get("outcome", "")).strip().upper()
        if allowed_outcomes and outcome and outcome not in allowed_outcomes:
            continue
        resolved_focal_seat = focal_seat
        if resolved_focal_seat is None and record is not None:
            resolved_focal_seat = int(record["focal_seat"])
        if resolved_focal_seat is None:
            raise ValueError(
                f"Could not infer focal seat for {bundle_path}; pass episodes_jsonl or explicit focal_seat"
            )
        selections.append(
            BundleSelection(
                bundle_path=bundle_path,
                pair_index=None if pair_swap is None else pair_swap[0],
                swap_index=None if pair_swap is None else pair_swap[1],
                focal_seat=int(resolved_focal_seat),
                outcome=outcome or None,
                episode_seed=None if record is None else int(record.get("episode_seed", 0)),
            )
        )
    return selections


def teacher_action_override_for(
    overrides: Mapping[tuple[str, int], int],
    *,
    bundle_path: Path,
    step_index: int,
) -> int | None:
    if not overrides:
        return None
    path = Path(bundle_path)
    keys = (
        (path.resolve().as_posix(), int(step_index)),
        (path.name, int(step_index)),
    )
    for key in keys:
        if key in overrides:
            return int(overrides[key])
    return None


def first_spec_hash(selections: Sequence[BundleSelection]) -> str | None:
    for selection in selections:
        meta, _steps, _fault = load_replay_bundle(selection.bundle_path)
        return str(meta.spec_hash256)
    return None


def _pair_swap_from_bundle_path(path: Path) -> tuple[int, int] | None:
    match = _PAIR_SWAP_RE.search(Path(path).stem)
    if match is None:
        return None
    return int(match.group("pair")), int(match.group("swap"))


__all__ = [
    "BundleSelection",
    "first_spec_hash",
    "load_episode_records",
    "load_teacher_action_overrides_jsonl",
    "select_bundles",
    "teacher_action_override_for",
]
