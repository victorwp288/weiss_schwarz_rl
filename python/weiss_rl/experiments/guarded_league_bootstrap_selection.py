from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_selection import load_reference_anchor_scores
from weiss_rl.experiments.bootstrap_commands import read_json_object
from weiss_rl.experiments.main_league_multiobjective_gate import (
    MultiObjectiveGateConfig,
    evaluate_main_league_multiobjective_gate,
)


@dataclass(frozen=True, slots=True)
class SnapshotCandidate:
    policy_id: str
    update: int
    checkpoint_path: Path


def policy_snapshots(run_dir: Path) -> list[SnapshotCandidate]:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    payload = read_json_object(registry_path)
    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError(f"snapshot registry contains no snapshots: {registry_path}")
    candidates: list[SnapshotCandidate] = []
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        policy_id = str(snapshot.get("policy_id", "")).strip()
        update = snapshot.get("update", snapshot.get("update_count"))
        if not policy_id.startswith("policy_") or isinstance(update, bool) or not isinstance(update, int):
            continue
        checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{int(update)}.pt"
        if checkpoint_path.is_file():
            candidates.append(
                SnapshotCandidate(policy_id=policy_id, update=int(update), checkpoint_path=checkpoint_path)
            )
    if not candidates:
        raise ValueError(f"no train policy snapshots with checkpoints found in {registry_path}")
    return sorted(candidates, key=lambda item: (item.update, item.policy_id))


def latest_policy_snapshot(run_dir: Path) -> SnapshotCandidate:
    return policy_snapshots(run_dir)[-1]


def recent_policy_snapshots(run_dir: Path, *, count: int) -> list[SnapshotCandidate]:
    if int(count) < 1:
        raise ValueError("count must be >= 1")
    snapshots = policy_snapshots(run_dir)
    return snapshots[-int(count) :]


def targeted_confirm_summary_path(*, run_dir: Path, output_subdir: str, paired_seeds: int) -> Path:
    return run_dir / "eval" / output_subdir / f"targeted_confirm{int(paired_seeds)}_summary.json"


def load_targeted_confirm_scores(path: Path) -> dict[str, float]:
    payload = read_json_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"targeted confirm summary missing rows: {path}")
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        opponent = row.get("opponent_policy_id")
        mean = row.get("mean")
        if isinstance(opponent, str) and opponent and isinstance(mean, int | float):
            scores[opponent] = float(mean)
    if not scores:
        raise ValueError(f"targeted confirm summary has no opponent scores: {path}")
    return scores


def selected_candidate_or_none(path: Path) -> dict[str, Any] | None:
    payload = read_json_object(path)
    selected = payload.get("selected")
    return dict(selected) if isinstance(selected, Mapping) else None


def selected_candidate(path: Path) -> dict[str, Any]:
    selected = selected_candidate_or_none(path)
    if selected is None:
        raise RuntimeError(f"candidate selector did not produce a selected candidate: {path}")
    return selected


def selection_anchor_scores(candidate: Mapping[str, Any]) -> dict[str, float]:
    for key in ("selection_anchor_scores", "anchor_scores"):
        raw_scores = candidate.get(key)
        if not isinstance(raw_scores, Mapping):
            continue
        scores: dict[str, float] = {}
        for anchor, value in raw_scores.items():
            if isinstance(anchor, str) and anchor and isinstance(value, int | float) and not isinstance(value, bool):
                scores[anchor] = float(value)
        if scores:
            return scores
    return {}


def evaluate_guard(
    *,
    scores: Mapping[str, float],
    required_anchors: Sequence[str],
    min_required_anchor_score: float,
    reference_anchor_scores: Mapping[str, float],
    max_reference_drop: float,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    required_payload: dict[str, dict[str, float | None]] = {}
    for anchor in required_anchors:
        score = scores.get(anchor)
        reference = reference_anchor_scores.get(anchor)
        delta = None if score is None or reference is None else float(score) - float(reference)
        required_payload[anchor] = {
            "score": None if score is None else float(score),
            "reference": None if reference is None else float(reference),
            "delta": delta,
        }
        if score is None:
            failures.append({"anchor": anchor, "reason": "missing_score"})
            continue
        if float(score) < float(min_required_anchor_score):
            failures.append(
                {
                    "anchor": anchor,
                    "reason": "below_min_required_anchor_score",
                    "score": float(score),
                    "threshold": float(min_required_anchor_score),
                }
            )
        if reference is not None and delta is not None and delta < -float(max_reference_drop):
            failures.append(
                {
                    "anchor": anchor,
                    "reason": "below_reference_drop_limit",
                    "score": float(score),
                    "reference": float(reference),
                    "delta": float(delta),
                    "threshold": -float(max_reference_drop),
                }
            )
    return {
        "passed": not failures,
        "failures": failures,
        "required_anchor_scores": required_payload,
        "min_required_anchor_score": float(min_required_anchor_score),
        "max_reference_drop": float(max_reference_drop),
    }


def evaluate_multiobjective_guard(
    *,
    candidate_summary_json: Path,
    reference_summary_jsons: Sequence[Path],
    fixed_opponents: Sequence[str],
    learned_opponents: Sequence[str],
    min_fixed_score: float,
    max_fixed_reference_drop: float,
    min_learned_score: float,
    min_learned_mean: float,
    min_learned_reference_delta: float | None,
    max_learned_reference_drop: float | None,
) -> dict[str, Any] | None:
    if not learned_opponents:
        return None
    return evaluate_main_league_multiobjective_gate(
        MultiObjectiveGateConfig(
            candidate_summary_jsons=(Path(candidate_summary_json),),
            reference_summary_jsons=tuple(Path(path) for path in reference_summary_jsons),
            fixed_opponents=tuple(str(opponent) for opponent in fixed_opponents),
            learned_opponents=tuple(str(opponent) for opponent in learned_opponents),
            min_fixed_score=float(min_fixed_score),
            max_fixed_reference_drop=float(max_fixed_reference_drop),
            min_learned_score=float(min_learned_score),
            min_learned_mean=float(min_learned_mean),
            min_learned_reference_delta=min_learned_reference_delta,
            max_learned_reference_drop=max_learned_reference_drop,
        )
    )


def selected_confirm_summary_path(
    *,
    raw_path: str,
    fallback_record: Mapping[str, Any] | None,
    repo_root: Path,
) -> Path | None:
    if raw_path:
        return resolve_repo_path(Path(raw_path), repo_root=repo_root)
    if fallback_record is None:
        return None
    fallback = fallback_record.get("summary_path")
    if not isinstance(fallback, str) or not fallback.strip():
        return None
    return resolve_repo_path(Path(fallback), repo_root=repo_root)


def resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def load_reference_scores_or_empty(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    return load_reference_anchor_scores(path)


__all__ = [
    "SnapshotCandidate",
    "evaluate_guard",
    "evaluate_multiobjective_guard",
    "latest_policy_snapshot",
    "load_reference_scores_or_empty",
    "load_targeted_confirm_scores",
    "policy_snapshots",
    "recent_policy_snapshots",
    "resolve_repo_path",
    "selected_candidate",
    "selected_candidate_or_none",
    "selected_confirm_summary_path",
    "selection_anchor_scores",
    "targeted_confirm_summary_path",
]
