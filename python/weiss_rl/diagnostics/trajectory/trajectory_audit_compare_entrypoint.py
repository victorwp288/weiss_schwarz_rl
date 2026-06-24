from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ALIGNMENT_FIELDS = (
    "policy_a_matches_policy_b_top_action_rate",
    "policy_a_matches_policy_b_top_action_family_rate",
    "policy_a_mean_probability_on_policy_b_top_action",
    "policy_a_mean_probability_on_policy_b_top_action_family",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two replay-audit trajectory summaries")
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    baseline = _load_json_object(args.baseline_summary)
    candidate = _load_json_object(args.candidate_summary)
    comparison = compare_audit_summaries(
        baseline=baseline,
        candidate=candidate,
        baseline_summary_path=args.baseline_summary,
        candidate_summary_path=args.candidate_summary,
        baseline_label=str(args.baseline_label),
        candidate_label=str(args.candidate_label),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"trajectory audit comparison written to {args.output_json}")
    return 0


def compare_audit_summaries(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_summary_path: Path,
    candidate_summary_path: Path,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    return {
        "format": "trajectory_audit_comparison_v1",
        "baseline": _summary_identity(
            summary=baseline,
            summary_path=baseline_summary_path,
            label=baseline_label,
        ),
        "candidate": _summary_identity(
            summary=candidate,
            summary_path=candidate_summary_path,
            label=candidate_label,
        ),
        "alignment": _field_delta_payload(baseline, candidate, fields=_ALIGNMENT_FIELDS),
        "episode_outcomes": _episode_outcome_comparison(
            baseline_summary=baseline,
            candidate_summary=candidate,
            baseline_summary_path=baseline_summary_path,
            candidate_summary_path=candidate_summary_path,
        ),
        "roles": {
            "focal": _compare_role(baseline, candidate, role="focal"),
            "opponent": _compare_role(baseline, candidate, role="opponent"),
        },
    }


def _summary_identity(*, summary: Mapping[str, Any], summary_path: Path, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "summary_path": summary_path.resolve().as_posix(),
        "policy_id": summary.get("policy_id"),
        "opponent_policy_id": summary.get("opponent_policy_id"),
        "source_paired_seeds_reused_for_policy_mismatch": bool(
            summary.get("source_paired_seeds_reused_for_policy_mismatch", False)
        ),
    }


def _episode_outcome_comparison(
    *,
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    baseline_summary_path: Path,
    candidate_summary_path: Path,
) -> dict[str, Any]:
    baseline_stats = _episode_stats(_resolve_episodes_path(baseline_summary, summary_path=baseline_summary_path))
    candidate_stats = _episode_stats(_resolve_episodes_path(candidate_summary, summary_path=candidate_summary_path))
    return {
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "delta": _numeric_delta(baseline_stats, candidate_stats),
    }


def _compare_role(baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    baseline_role = _role_summary(baseline, role=role)
    candidate_role = _role_summary(candidate, role=role)
    return {
        "baseline": _role_payload(baseline_role),
        "candidate": _role_payload(candidate_role),
        "delta": {
            "compared_steps": int(candidate_role.get("compared_steps", 0))
            - int(baseline_role.get("compared_steps", 0)),
            "numeric_means": _numeric_mean_deltas(
                baseline_role.get("numeric_summaries", {}),
                candidate_role.get("numeric_summaries", {}),
            ),
            "recorded_family_counts": _counter_deltas(
                baseline_role.get("recorded_family_counts", []),
                candidate_role.get("recorded_family_counts", []),
                key_name="family",
            ),
            "legal_family_presence_rates": _rate_deltas(
                baseline_role.get("legal_family_presence_rates", []),
                candidate_role.get("legal_family_presence_rates", []),
            ),
        },
    }


def _role_payload(role_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compared_steps": int(role_summary.get("compared_steps", 0)),
        "recorded_family_counts": role_summary.get("recorded_family_counts", []),
        "numeric_means": _numeric_means(role_summary.get("numeric_summaries", {})),
        "legal_family_presence_rates": role_summary.get("legal_family_presence_rates", []),
    }


def _field_delta_payload(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    fields: Sequence[str],
) -> dict[str, Any]:
    return {
        field: {
            "baseline": _optional_float(baseline.get(field)),
            "candidate": _optional_float(candidate.get(field)),
            "delta": _optional_delta(baseline.get(field), candidate.get(field)),
        }
        for field in fields
    }


def _numeric_delta(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(set(baseline) | set(candidate)):
        baseline_value = baseline.get(key)
        candidate_value = candidate.get(key)
        if isinstance(baseline_value, int | float) and isinstance(candidate_value, int | float):
            deltas[key] = float(candidate_value) - float(baseline_value)
    return deltas


def _numeric_means(raw_summaries: Any) -> dict[str, float]:
    if not isinstance(raw_summaries, Mapping):
        return {}
    means: dict[str, float] = {}
    for key, raw_summary in raw_summaries.items():
        if not isinstance(raw_summary, Mapping):
            continue
        mean = raw_summary.get("mean")
        if isinstance(mean, int | float):
            means[str(key)] = float(mean)
    return means


def _numeric_mean_deltas(baseline_raw: Any, candidate_raw: Any) -> dict[str, float]:
    return _numeric_delta(_numeric_means(baseline_raw), _numeric_means(candidate_raw))


def _counter_deltas(baseline_raw: Any, candidate_raw: Any, *, key_name: str) -> list[dict[str, Any]]:
    baseline = _counter_map(baseline_raw, key_name=key_name)
    candidate = _counter_map(candidate_raw, key_name=key_name)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(baseline) | set(candidate)):
        baseline_count = int(baseline.get(key, 0))
        candidate_count = int(candidate.get(key, 0))
        rows.append(
            {
                key_name: key,
                "baseline": baseline_count,
                "candidate": candidate_count,
                "delta": candidate_count - baseline_count,
            }
        )
    rows.sort(key=lambda item: (-abs(int(item["delta"])), str(item[key_name])))
    return rows


def _rate_deltas(baseline_raw: Any, candidate_raw: Any) -> list[dict[str, Any]]:
    baseline = _rate_map(baseline_raw)
    candidate = _rate_map(candidate_raw)
    rows = [
        {
            "family": family,
            "baseline": baseline.get(family),
            "candidate": candidate.get(family),
            "delta": (
                None
                if baseline.get(family) is None or candidate.get(family) is None
                else float(candidate[family]) - float(baseline[family])
            ),
        }
        for family in sorted(set(baseline) | set(candidate))
    ]
    rows.sort(key=lambda item: (-(abs(float(item["delta"])) if item["delta"] is not None else -1.0), item["family"]))
    return rows


def _counter_map(raw: Any, *, key_name: str) -> dict[str, int]:
    if not isinstance(raw, list):
        return {}
    result: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        key = item.get(key_name)
        count = item.get("count")
        if key is not None and isinstance(count, int):
            result[str(key)] = int(count)
    return result


def _rate_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, list):
        return {}
    result: dict[str, float] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        family = item.get("family")
        rate = item.get("rate")
        if family is not None and isinstance(rate, int | float):
            result[str(family)] = float(rate)
    return result


def _role_summary(summary: Mapping[str, Any], *, role: str) -> Mapping[str, Any]:
    trajectory_summary = summary.get("trajectory_summary")
    if not isinstance(trajectory_summary, Mapping):
        return {}
    role_summaries = trajectory_summary.get("role_summaries")
    if not isinstance(role_summaries, list):
        return {}
    return next(
        (item for item in role_summaries if isinstance(item, Mapping) and item.get("role") == role),
        {},
    )


def _episode_stats(episodes_path: Path | None) -> dict[str, float | int | None]:
    if episodes_path is None or not episodes_path.is_file():
        return {"games": 0, "wins": 0, "mean": None}
    records = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    games = len(records)
    wins = sum(1 for record in records if record.get("outcome") == "W")
    return {
        "games": games,
        "wins": wins,
        "mean": None if games == 0 else float(wins / games),
        "total_actions_per_game": _per_game(records, "total_actions"),
        "pass_actions_per_game": _per_game(records, "pass_actions"),
        "pass_with_nonpass_available_per_game": _per_game(records, "pass_with_nonpass_available"),
        "main_move_actions_per_game": _per_game(records, "main_move_actions"),
    }


def _per_game(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return float(sum(int(record.get(key, 0)) for record in records) / len(records))


def _resolve_episodes_path(summary: Mapping[str, Any], *, summary_path: Path) -> Path | None:
    raw_path = summary.get("episodes_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.is_file():
        return cwd_candidate
    summary_parent_candidate = summary_path.resolve().parent / path.name
    if summary_parent_candidate.is_file():
        return summary_parent_candidate
    return cwd_candidate


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _optional_delta(baseline: Any, candidate: Any) -> float | None:
    baseline_float = _optional_float(baseline)
    candidate_float = _optional_float(candidate)
    if baseline_float is None or candidate_float is None:
        return None
    return candidate_float - baseline_float


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
