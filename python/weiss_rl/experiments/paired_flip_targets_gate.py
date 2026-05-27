"""Gate paired-flip target extraction before building repair datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PairedFlipTargetsGateConfig:
    target_jsons: tuple[Path, ...]
    min_total_targets: int = 1
    min_target_opponents: int = 1
    min_distinct_pair_indices: int = 1
    excluded_pair_indices: tuple[int, ...] = ()
    required_opponents: tuple[str, ...] = ()


def evaluate_paired_flip_targets_gate(config: PairedFlipTargetsGateConfig) -> dict[str, Any]:
    if not config.target_jsons:
        raise ValueError("at least one paired-flip targets JSON is required")

    payloads = [_read_json_object(path) for path in config.target_jsons]
    targets = [target for payload in payloads for target in payload.get("targets", []) if isinstance(target, Mapping)]
    target_opponents = sorted({str(target.get("opponent_policy_id") or "") for target in targets})
    target_opponents = [opponent for opponent in target_opponents if opponent]
    target_pair_indices = sorted(
        {int(target["pair_index"]) for target in targets if target.get("pair_index") is not None}
    )
    excluded_hits = sorted(set(target_pair_indices).intersection(set(config.excluded_pair_indices)))
    missing_required_opponents = [
        opponent for opponent in config.required_opponents if opponent not in set(target_opponents)
    ]

    failures: list[str] = []
    if len(targets) < int(config.min_total_targets):
        failures.append(f"target_count_below:{len(targets)}<{int(config.min_total_targets)}")
    if len(target_opponents) < int(config.min_target_opponents):
        failures.append(f"target_opponent_count_below:{len(target_opponents)}<{int(config.min_target_opponents)}")
    if len(target_pair_indices) < int(config.min_distinct_pair_indices):
        failures.append(
            f"distinct_pair_index_count_below:{len(target_pair_indices)}<{int(config.min_distinct_pair_indices)}"
        )
    if excluded_hits:
        failures.append("excluded_pair_indices_present:" + ",".join(str(index) for index in excluded_hits))
    if missing_required_opponents:
        failures.append("required_opponents_missing:" + ",".join(missing_required_opponents))

    return {
        "kind": "paired_flip_targets_gate_v1",
        "passed": not failures,
        "failures": failures,
        "target_jsons": [Path(path).as_posix() for path in config.target_jsons],
        "thresholds": {
            "min_total_targets": int(config.min_total_targets),
            "min_target_opponents": int(config.min_target_opponents),
            "min_distinct_pair_indices": int(config.min_distinct_pair_indices),
            "excluded_pair_indices": list(config.excluded_pair_indices),
            "required_opponents": list(config.required_opponents),
        },
        "summary": {
            "target_count": len(targets),
            "target_opponent_count": len(target_opponents),
            "target_opponents": target_opponents,
            "distinct_pair_index_count": len(target_pair_indices),
            "target_pair_indices": target_pair_indices,
            "excluded_pair_indices_present": excluded_hits,
            "missing_required_opponents": missing_required_opponents,
        },
    }


def write_paired_flip_targets_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _paths(paths: Iterable[Path | str]) -> tuple[Path, ...]:
    return tuple(Path(path) for path in paths)


def build_paired_flip_targets_gate_config(
    *,
    target_jsons: Sequence[Path | str],
    min_total_targets: int = 1,
    min_target_opponents: int = 1,
    min_distinct_pair_indices: int = 1,
    excluded_pair_indices: Sequence[int] = (),
    required_opponents: Sequence[str] = (),
) -> PairedFlipTargetsGateConfig:
    return PairedFlipTargetsGateConfig(
        target_jsons=_paths(target_jsons),
        min_total_targets=int(min_total_targets),
        min_target_opponents=int(min_target_opponents),
        min_distinct_pair_indices=int(min_distinct_pair_indices),
        excluded_pair_indices=tuple(int(index) for index in excluded_pair_indices),
        required_opponents=tuple(str(opponent) for opponent in required_opponents),
    )


__all__ = [
    "PairedFlipTargetsGateConfig",
    "build_paired_flip_targets_gate_config",
    "evaluate_paired_flip_targets_gate",
    "write_paired_flip_targets_gate",
]
