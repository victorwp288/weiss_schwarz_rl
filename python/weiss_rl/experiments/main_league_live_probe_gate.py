from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.league_progress_summary import slugify_opponent


@dataclass(frozen=True, slots=True)
class MainLeagueLiveProbeGateConfig:
    league_progress_summary_json: Path
    min_champion_envs: float = 1.0
    min_hard_negative_envs: float = 1.0
    min_heuristic_public_envs: float = 1.0
    min_heuristic_public_variant_envs: float = 1.0
    min_noleague_baseline_envs: float = 1.0
    min_champion_pool_size: float = 1.0
    min_hard_negative_pool_size: float = 1.0
    required_sampled_policies: tuple[str, ...] = ()
    min_required_sampled_policy_envs: float = 1.0


def evaluate_main_league_live_probe_gate(config: MainLeagueLiveProbeGateConfig) -> dict[str, Any]:
    payload = _read_json_object(config.league_progress_summary_json)
    failures: list[dict[str, Any]] = []
    exposure_totals = payload.get("exposure_totals") if isinstance(payload.get("exposure_totals"), Mapping) else {}
    pool_size_max = payload.get("pool_size_max") if isinstance(payload.get("pool_size_max"), Mapping) else {}
    policy_exposure_totals = (
        payload.get("policy_exposure_totals") if isinstance(payload.get("policy_exposure_totals"), Mapping) else {}
    )
    sampled_policy_exposure = (
        policy_exposure_totals.get("sampled") if isinstance(policy_exposure_totals.get("sampled"), Mapping) else {}
    )

    _require_min(
        failures,
        values=exposure_totals,
        key="pfsp_champion_envs",
        threshold=float(config.min_champion_envs),
        reason="champion_exposure_below_min",
    )
    _require_min(
        failures,
        values=exposure_totals,
        key="pfsp_hard_negative_envs",
        threshold=float(config.min_hard_negative_envs),
        reason="hard_negative_exposure_below_min",
    )
    _require_min(
        failures,
        values=exposure_totals,
        key="pfsp_heuristic_public_envs",
        threshold=float(config.min_heuristic_public_envs),
        reason="heuristic_public_exposure_below_min",
    )
    _require_min(
        failures,
        values=exposure_totals,
        key="pfsp_heuristic_public_variant_envs",
        threshold=float(config.min_heuristic_public_variant_envs),
        reason="heuristic_public_variant_exposure_below_min",
    )
    _require_min(
        failures,
        values=exposure_totals,
        key="pfsp_noleague_baseline_envs",
        threshold=float(config.min_noleague_baseline_envs),
        reason="noleague_baseline_exposure_below_min",
    )
    _require_min(
        failures,
        values=pool_size_max,
        key="pfsp_champion_pool_size",
        threshold=float(config.min_champion_pool_size),
        reason="champion_pool_size_below_min",
    )
    _require_min(
        failures,
        values=pool_size_max,
        key="pfsp_hard_negative_pool_size",
        threshold=float(config.min_hard_negative_pool_size),
        reason="hard_negative_pool_size_below_min",
    )

    sampled_policy_summary = _required_sampled_policy_summary(
        sampled_policy_exposure,
        required_policies=config.required_sampled_policies,
        min_envs=float(config.min_required_sampled_policy_envs),
    )
    failures.extend(sampled_policy_summary["failures"])

    return {
        "kind": "main_league_live_probe_gate_v1",
        "passed": not failures,
        "failures": failures,
        "league_progress_summary_json": Path(config.league_progress_summary_json).as_posix(),
        "thresholds": {
            "min_champion_envs": float(config.min_champion_envs),
            "min_hard_negative_envs": float(config.min_hard_negative_envs),
            "min_heuristic_public_envs": float(config.min_heuristic_public_envs),
            "min_heuristic_public_variant_envs": float(config.min_heuristic_public_variant_envs),
            "min_noleague_baseline_envs": float(config.min_noleague_baseline_envs),
            "min_champion_pool_size": float(config.min_champion_pool_size),
            "min_hard_negative_pool_size": float(config.min_hard_negative_pool_size),
            "required_sampled_policies": list(config.required_sampled_policies),
            "min_required_sampled_policy_envs": float(config.min_required_sampled_policy_envs),
        },
        "summary": {
            "records_seen": payload.get("records_seen"),
            "update_count_min": payload.get("update_count_min"),
            "update_count_max": payload.get("update_count_max"),
            "exposure_totals": {
                key: _float_or_zero(exposure_totals.get(key))
                for key in (
                    "pfsp_champion_envs",
                    "pfsp_hard_negative_envs",
                    "pfsp_heuristic_public_envs",
                    "pfsp_heuristic_public_variant_envs",
                    "pfsp_noleague_baseline_envs",
                    "pfsp_sampled_envs",
                )
            },
            "pool_size_max": {
                key: _float_or_zero(pool_size_max.get(key))
                for key in ("pfsp_champion_pool_size", "pfsp_hard_negative_pool_size", "pfsp_pool_size")
            },
            "required_sampled_policies": sampled_policy_summary["policies"],
        },
    }


def write_main_league_live_probe_gate(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _required_sampled_policy_summary(
    sampled_policy_exposure: Mapping[str, Any],
    *,
    required_policies: Sequence[str],
    min_envs: float,
) -> dict[str, Any]:
    present = {str(policy): _float_or_zero(value) for policy, value in sampled_policy_exposure.items()}
    policy_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for required in required_policies:
        required_slug = slugify_opponent(required)
        matches = [
            {"policy": policy, "envs": envs}
            for policy, envs in present.items()
            if _is_seed_wrapped_suffix_match(policy, required_slug)
        ]
        total_envs = sum(float(match["envs"]) for match in matches)
        row = {
            "requested_policy": required,
            "requested_slug": required_slug,
            "matched_policies": matches,
            "total_envs": total_envs,
        }
        policy_rows.append(row)
        if total_envs < min_envs:
            failures.append(
                {
                    "reason": "required_sampled_policy_exposure_below_min",
                    "requested_policy": required,
                    "total_envs": total_envs,
                    "threshold": min_envs,
                }
            )
    return {"policies": policy_rows, "failures": failures}


def _require_min(
    failures: list[dict[str, Any]],
    *,
    values: Mapping[str, Any],
    key: str,
    threshold: float,
    reason: str,
) -> None:
    value = _float_or_zero(values.get(key))
    if value < threshold:
        failures.append({"reason": reason, "key": key, "value": value, "threshold": threshold})


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _is_seed_wrapped_suffix_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return left.endswith(f"_{right}") or right.endswith(f"_{left}")


__all__ = [
    "MainLeagueLiveProbeGateConfig",
    "evaluate_main_league_live_probe_gate",
    "write_main_league_live_probe_gate",
]
