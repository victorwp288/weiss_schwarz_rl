from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID, config_marks_noleague_baseline
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)

DEFAULT_REQUIRED_ANCHORS = (
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)
DEFAULT_CONFIRM_OPPONENTS = (
    "B0 RandomLegal",
    "B2 HeuristicPublic",
    "B3 HeuristicPublicAggro",
    "B4 HeuristicPublicControl",
)
SELECTED_CANDIDATE_POLICY_ID = "selected_candidate"


def _json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def _anchor_scores(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    scores: dict[str, float] = {}
    for key, raw_score in value.items():
        score = _finite_float(raw_score)
        if score is not None:
            scores[str(key)] = score
    return scores


def load_reference_anchor_scores(path: Path) -> dict[str, float]:
    """Load opponent anchor scores from a targeted-confirm summary or a simple mapping."""

    payload = _json_object(path)
    if payload is None:
        raise ValueError(f"reference summary must be a JSON object: {path}")
    direct_scores = _anchor_scores(payload.get("anchor_scores"))
    if direct_scores:
        return direct_scores
    rows = payload.get("rows")
    if isinstance(rows, list):
        scores: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opponent = row.get("opponent_policy_id")
            mean = _finite_float(row.get("mean"))
            if isinstance(opponent, str) and opponent and mean is not None:
                scores[opponent] = mean
        if scores:
            return scores
    return _anchor_scores(payload)


def _reference_comparison(
    anchor_scores: Mapping[str, float],
    *,
    reference_anchor_scores: Mapping[str, float],
    reference_label: str,
) -> dict[str, Any] | None:
    common_anchors = sorted(set(anchor_scores) & set(reference_anchor_scores))
    if not common_anchors:
        return None
    anchor_deltas = {
        anchor: float(anchor_scores[anchor]) - float(reference_anchor_scores[anchor]) for anchor in common_anchors
    }
    deltas = list(anchor_deltas.values())
    return {
        "reference_label": str(reference_label),
        "common_anchors": common_anchors,
        "reference_anchor_scores": {anchor: float(reference_anchor_scores[anchor]) for anchor in common_anchors},
        "anchor_deltas": anchor_deltas,
        "mean_delta": sum(deltas) / len(deltas),
        "min_delta": min(deltas),
        "all_common_at_or_above_reference": all(delta >= 0.0 for delta in deltas),
    }


def _snapshot_index(run_dir: Path) -> dict[int, dict[str, Any]]:
    registry = _json_object(run_dir / "training" / "snapshots" / REGISTRY_FILENAME)
    snapshots = [] if registry is None else registry.get("snapshots", [])
    index: dict[int, dict[str, Any]] = {}
    if not isinstance(snapshots, list):
        return index
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        update = snapshot.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            continue
        existing = index.get(int(update))
        snapshot_policy_id = snapshot.get("policy_id")
        existing_policy_id = None if existing is None else existing.get("policy_id")
        snapshot_is_primary = isinstance(snapshot_policy_id, str) and snapshot_policy_id.startswith("policy_")
        existing_is_primary = isinstance(existing_policy_id, str) and existing_policy_id.startswith("policy_")
        if existing is None or (snapshot_is_primary and not existing_is_primary):
            index[int(update)] = dict(snapshot)
    return index


def _snapshot_by_policy_id(run_dir: Path) -> dict[str, dict[str, Any]]:
    registry = _json_object(run_dir / "training" / "snapshots" / REGISTRY_FILENAME)
    snapshots = [] if registry is None else registry.get("snapshots", [])
    index: dict[str, dict[str, Any]] = {}
    if not isinstance(snapshots, list):
        return index
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping):
            continue
        policy_id = snapshot.get("policy_id")
        if isinstance(policy_id, str) and policy_id:
            index[policy_id] = dict(snapshot)
    return index


def _policy_version_from_policy_id(policy_id: str) -> int | None:
    normalized = policy_id.strip()
    if not normalized.startswith("policy_"):
        return None
    suffix = normalized.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _paired_seed_count(payload: Mapping[str, Any] | None) -> int:
    if payload is None:
        return 0
    paired_seeds = payload.get("paired_seeds")
    if isinstance(paired_seeds, bool) or not isinstance(paired_seeds, int):
        return 0
    return int(paired_seeds)


def _baseline_alias_info(run_dir: Path) -> dict[str, Any] | None:
    snapshot = _snapshot_by_policy_id(run_dir).get(NOLEAGUE_BASELINE_POLICY_ID)
    if snapshot is None:
        return None
    metadata = _json_object(
        run_dir / "training" / "snapshots" / NOLEAGUE_BASELINE_POLICY_ID / SNAPSHOT_METADATA_FILENAME
    )
    metadata_format = None if metadata is None else metadata.get("format")
    return {
        "policy_id": NOLEAGUE_BASELINE_POLICY_ID,
        "update": snapshot.get("update"),
        "path": snapshot.get("path"),
        "weights_sha256": snapshot.get("weights_sha256"),
        "metadata_format": metadata_format if isinstance(metadata_format, str) else None,
    }


def _record_from_summary(
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    fallback_policy_id: str,
    source_path: Path,
    snapshot_by_update: Mapping[int, Mapping[str, Any]],
    snapshot_by_policy_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    aggregate_score = _finite_float(summary.get("aggregate_score"))
    update_count = summary.get("update_count")
    if aggregate_score is None or isinstance(update_count, bool) or not isinstance(update_count, int):
        return None

    policy_id = str(summary.get("policy_id") or summary.get("focal_policy_id") or fallback_policy_id)
    policy_version = summary.get("policy_version")
    version_policy_id = f"policy_{int(policy_version):06d}" if isinstance(policy_version, int) else None
    snapshot = {}
    for candidate_policy_id in (version_policy_id, policy_id):
        if not isinstance(candidate_policy_id, str) or not candidate_policy_id:
            continue
        candidate = snapshot_by_policy_id.get(candidate_policy_id)
        if candidate is None:
            continue
        snapshot_update = candidate.get("update")
        if not isinstance(snapshot_update, bool) and snapshot_update == int(update_count):
            snapshot = candidate
            break
    if not snapshot:
        snapshot = snapshot_by_update.get(int(update_count), {})
    snapshot_policy_id = snapshot.get("policy_id")
    if not isinstance(snapshot_policy_id, str) or not snapshot_policy_id:
        snapshot_policy_id = version_policy_id if version_policy_id is not None else policy_id
    snapshot_path = snapshot.get("path")
    weights_sha256 = snapshot.get("weights_sha256")
    paired_seeds = summary.get("paired_seeds")
    return {
        "run_dir": run_dir.as_posix(),
        "run_name": run_dir.name,
        "train_policy_id": policy_id,
        "snapshot_policy_id": snapshot_policy_id,
        "snapshot_path": None if not isinstance(snapshot_path, str) else snapshot_path,
        "weights_sha256": None if not isinstance(weights_sha256, str) else weights_sha256,
        "update_count": int(update_count),
        "policy_version": policy_version if isinstance(policy_version, int) else None,
        "aggregate_score": aggregate_score,
        "anchor_scores": _anchor_scores(summary.get("anchor_scores")),
        "paired_seeds": int(paired_seeds)
        if not isinstance(paired_seeds, bool) and isinstance(paired_seeds, int)
        else 0,
        "source_path": source_path.as_posix(),
    }


def load_b1_dev_eval_records(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    snapshot_by_update = _snapshot_index(run_dir)
    snapshot_by_policy_id = _snapshot_by_policy_id(run_dir)
    records: list[dict[str, Any]] = []
    summaries = _json_object(run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json")
    if summaries is not None:
        for policy_id, summary in summaries.items():
            if not isinstance(summary, Mapping):
                continue
            record = _record_from_summary(
                run_dir=run_dir,
                summary=summary,
                fallback_policy_id=str(policy_id),
                source_path=run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
                snapshot_by_update=snapshot_by_update,
                snapshot_by_policy_id=snapshot_by_policy_id,
            )
            if record is not None:
                records.append(record)
    if not records:
        for summary_path in sorted((run_dir / "eval" / "dev_eval").glob("update_*/summary.json")):
            summary = _json_object(summary_path)
            if summary is None:
                continue
            record = _record_from_summary(
                run_dir=run_dir,
                summary=summary,
                fallback_policy_id=summary_path.parent.name,
                source_path=summary_path,
                snapshot_by_update=snapshot_by_update,
                snapshot_by_policy_id=snapshot_by_policy_id,
            )
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda record: (int(record["update_count"]), str(record["snapshot_policy_id"])))


def _confirmation_scores(
    run_dir: Path,
    snapshot_policy_id: str,
    *,
    required_anchors: Sequence[str],
    min_paired_seeds: int = 0,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    snapshot_by_policy_id = _snapshot_by_policy_id(run_dir)
    target_snapshot = snapshot_by_policy_id.get(str(snapshot_policy_id), {})
    target_weights_sha256 = target_snapshot.get("weights_sha256")
    summaries = sorted((run_dir / "eval").glob("**/targeted_confirm*_summary.json"))
    for summary_path in summaries:
        summary = _json_object(summary_path)
        if summary is None:
            continue
        focal_policy_id = str(summary.get("focal_policy_id", "")).strip()
        if focal_policy_id != snapshot_policy_id:
            focal_snapshot = snapshot_by_policy_id.get(focal_policy_id, {})
            focal_weights_sha256 = focal_snapshot.get("weights_sha256")
            if (
                not isinstance(target_weights_sha256, str)
                or not target_weights_sha256
                or not isinstance(focal_weights_sha256, str)
                or focal_weights_sha256 != target_weights_sha256
            ):
                continue
        if not focal_policy_id:
            continue
        rows = summary.get("rows")
        if not isinstance(rows, list):
            continue
        anchor_scores: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opponent = str(row.get("opponent_policy_id", ""))
            mean = _finite_float(row.get("mean"))
            if opponent and mean is not None:
                anchor_scores[opponent] = mean
        missing_required = [anchor for anchor in required_anchors if anchor not in anchor_scores]
        if missing_required:
            continue
        paired_seeds = summary.get("paired_seeds")
        paired_seed_count = (
            int(paired_seeds) if not isinstance(paired_seeds, bool) and isinstance(paired_seeds, int) else 0
        )
        if paired_seed_count < int(min_paired_seeds):
            continue
        matches.append(
            {
                "summary_path": summary_path.as_posix(),
                "paired_seeds": summary.get("paired_seeds"),
                "anchor_scores": anchor_scores,
                "focal_policy_id": focal_policy_id,
                "matched_by_weights_sha256": focal_policy_id != snapshot_policy_id,
                "_paired_seed_count": paired_seed_count,
            }
        )
    if not matches:
        return None
    selected = max(
        matches,
        key=lambda item: (
            int(item["_paired_seed_count"]),
            len(item["anchor_scores"]),
            str(item["summary_path"]),
        ),
    )
    return {
        "summary_path": selected["summary_path"],
        "paired_seeds": selected["paired_seeds"],
        "anchor_scores": selected["anchor_scores"],
        "focal_policy_id": selected["focal_policy_id"],
        "matched_by_weights_sha256": selected["matched_by_weights_sha256"],
    }


def _targeted_confirm_only_records(
    run_dir: Path,
    *,
    required_anchors: Sequence[str],
    existing_snapshot_policy_ids: set[str],
    min_paired_seeds: int = 0,
) -> list[dict[str, Any]]:
    snapshot_by_policy_id = _snapshot_by_policy_id(run_dir)
    focal_policy_ids: set[str] = set()
    for summary_path in sorted((run_dir / "eval").glob("**/targeted_confirm*_summary.json")):
        summary = _json_object(summary_path)
        if summary is None:
            continue
        focal_policy_id = str(summary.get("focal_policy_id", "")).strip()
        if (
            focal_policy_id
            and focal_policy_id not in existing_snapshot_policy_ids
            and focal_policy_id in snapshot_by_policy_id
        ):
            focal_policy_ids.add(focal_policy_id)

    records: list[dict[str, Any]] = []
    for focal_policy_id in sorted(focal_policy_ids):
        confirmation = _confirmation_scores(
            run_dir,
            focal_policy_id,
            required_anchors=required_anchors,
            min_paired_seeds=int(min_paired_seeds),
        )
        if confirmation is None:
            continue
        snapshot = snapshot_by_policy_id[focal_policy_id]
        update = snapshot.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            continue
        anchor_scores = _anchor_scores(confirmation.get("anchor_scores"))
        summary_path = Path(str(confirmation["summary_path"]))
        confirmation_summary = _json_object(summary_path)
        aggregate_score = None if confirmation_summary is None else _finite_float(confirmation_summary.get("mean"))
        if aggregate_score is None:
            aggregate_score = sum(anchor_scores.values()) / len(anchor_scores) if anchor_scores else 0.0
        snapshot_path = snapshot.get("path")
        weights_sha256 = snapshot.get("weights_sha256")
        records.append(
            {
                "run_dir": run_dir.as_posix(),
                "run_name": run_dir.name,
                "train_policy_id": f"targeted_confirm_{focal_policy_id}",
                "snapshot_policy_id": focal_policy_id,
                "snapshot_path": None if not isinstance(snapshot_path, str) else snapshot_path,
                "weights_sha256": None if not isinstance(weights_sha256, str) else weights_sha256,
                "update_count": int(update),
                "policy_version": _policy_version_from_policy_id(focal_policy_id),
                "aggregate_score": float(aggregate_score),
                "anchor_scores": anchor_scores,
                "source_path": str(confirmation["summary_path"]),
                "targeted_confirm_only": True,
            }
        )
    return records


def _confirmatory_dev_eval(run_dir: Path, update_count: int) -> dict[str, Any] | None:
    summary_path = run_dir / "eval" / "dev_eval_confirmatory" / f"update_{int(update_count)}" / "summary.json"
    summary = _json_object(summary_path)
    if summary is None:
        return None
    aggregate_score = _finite_float(summary.get("aggregate_score"))
    return {
        "summary_path": summary_path.as_posix(),
        "aggregate_score": aggregate_score,
        "paired_seeds": _paired_seed_count(summary),
        "anchor_scores": _anchor_scores(summary.get("anchor_scores")),
    }


def _candidate_metrics_from_anchor_scores(
    anchor_scores: Mapping[str, float],
    *,
    required_anchors: Sequence[str],
    min_required_anchor_score: float,
) -> dict[str, Any]:
    required_scores = [anchor_scores[name] for name in required_anchors if name in anchor_scores]
    missing = [name for name in required_anchors if name not in anchor_scores]
    required_mean = sum(required_scores) / len(required_scores) if required_scores else 0.0
    required_min = min(required_scores) if required_scores else 0.0
    eligible = not missing and all(score >= min_required_anchor_score for score in required_scores)
    return {
        "required_anchor_mean": required_mean,
        "required_anchor_min": required_min,
        "selection_score": 0.7 * required_mean + 0.3 * required_min,
        "missing_required_anchors": missing,
        "eligible": eligible,
        "ineligibility_reasons": (
            []
            if eligible
            else [
                *[f"missing {anchor}" for anchor in missing],
                *[
                    f"{anchor} {anchor_scores[anchor]:.4f} < {min_required_anchor_score:.4f}"
                    for anchor in required_anchors
                    if anchor in anchor_scores and anchor_scores[anchor] < min_required_anchor_score
                ],
            ]
        ),
    }


def _candidate_metrics(
    record: Mapping[str, Any],
    *,
    required_anchors: Sequence[str],
    min_required_anchor_score: float,
) -> dict[str, Any]:
    return _candidate_metrics_from_anchor_scores(
        _anchor_scores(record.get("anchor_scores")),
        required_anchors=required_anchors,
        min_required_anchor_score=min_required_anchor_score,
    )


def _selection_confirmation_source(
    *,
    targeted_confirmation: Mapping[str, Any] | None,
    confirmatory_dev_eval: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any] | None]:
    if targeted_confirmation is not None:
        return "targeted_confirm", targeted_confirmation
    if confirmatory_dev_eval is not None:
        return "confirmatory_dev_eval", confirmatory_dev_eval
    return "periodic_dev_eval", None


def _selection_source_rank(source: object) -> int:
    if source == "targeted_confirm":
        return 2
    if source == "confirmatory_dev_eval":
        return 1
    return 0


def _confirmation_command(
    *,
    stack_config: Path,
    run_dir: Path,
    snapshot_policy_id: str,
    opponents: Sequence[str],
    paired_seeds: int,
    output_subdir: str,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--extra",
        "dev",
        "--extra",
        "sim",
        "python",
        "python/scripts/targeted_confirm_eval.py",
        "--stack-config",
        stack_config.as_posix(),
        "--run-dir",
        run_dir.as_posix(),
        "--snapshot-registry-json",
        (run_dir / "training" / "snapshots" / "registry.json").as_posix(),
        "--b1-baseline-run-dir",
        run_dir.as_posix(),
        "--focal-policy-id",
        snapshot_policy_id,
        "--paired-seeds",
        str(int(paired_seeds)),
        "--workers",
        "1",
        "--bootstrap-samples",
        "2000",
        "--output-subdir",
        output_subdir,
    ]
    for opponent in opponents:
        command.extend(["--opponent", opponent])
    return command


def build_b1_candidate_selection(
    run_dirs: Iterable[Path],
    *,
    stack_config: Path | None = None,
    required_anchors: Sequence[str] = DEFAULT_REQUIRED_ANCHORS,
    confirm_opponents: Sequence[str] = DEFAULT_CONFIRM_OPPONENTS,
    min_required_anchor_score: float = 0.5,
    falloff_warning_threshold: float = 0.05,
    confirm_paired_seeds: int = 64,
    reference_anchor_scores: Mapping[str, float] | None = None,
    reference_label: str = "reference",
) -> dict[str, Any]:
    reference_scores = dict(reference_anchor_scores or {})
    candidates: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for run_dir in [path.resolve() for path in run_dirs]:
        records = load_b1_dev_eval_records(run_dir)
        records.extend(
            _targeted_confirm_only_records(
                run_dir,
                required_anchors=required_anchors,
                existing_snapshot_policy_ids={str(record["snapshot_policy_id"]) for record in records},
                min_paired_seeds=int(confirm_paired_seeds),
            )
        )
        records = sorted(records, key=lambda record: (int(record["update_count"]), str(record["snapshot_policy_id"])))
        run_candidates: list[dict[str, Any]] = []
        best_selection_score = None
        latest_selection_score = None
        baseline_alias = _baseline_alias_info(run_dir)
        for record in records:
            metrics = _candidate_metrics(
                record,
                required_anchors=required_anchors,
                min_required_anchor_score=float(min_required_anchor_score),
            )
            candidate = {**record, **metrics}
            candidate["dev_eval_required_anchor_mean"] = candidate["required_anchor_mean"]
            candidate["dev_eval_required_anchor_min"] = candidate["required_anchor_min"]
            candidate["dev_eval_selection_score"] = candidate["selection_score"]
            candidate["dev_eval_eligible"] = candidate["eligible"]
            candidate["selection_score_source"] = "periodic_dev_eval"
            candidate["selection_score_source_rank"] = _selection_source_rank(candidate["selection_score_source"])
            candidate["selection_paired_seeds"] = _paired_seed_count(candidate)
            confirmatory = _confirmatory_dev_eval(run_dir, int(candidate["update_count"]))
            if confirmatory is not None:
                candidate["confirmatory_dev_eval"] = confirmatory
            confirmation = _confirmation_scores(
                run_dir,
                str(candidate["snapshot_policy_id"]),
                required_anchors=required_anchors,
                min_paired_seeds=int(confirm_paired_seeds),
            )
            if confirmation is not None:
                candidate["confirmation"] = confirmation
            selection_source, selection_confirmation = _selection_confirmation_source(
                targeted_confirmation=confirmation,
                confirmatory_dev_eval=confirmatory,
            )
            if selection_confirmation is not None:
                confirmation_metrics = _candidate_metrics_from_anchor_scores(
                    _anchor_scores(selection_confirmation.get("anchor_scores")),
                    required_anchors=required_anchors,
                    min_required_anchor_score=float(min_required_anchor_score),
                )
                candidate["selection_score_source"] = selection_source
                candidate["selection_score_source_rank"] = _selection_source_rank(selection_source)
                candidate["selection_anchor_scores"] = _anchor_scores(selection_confirmation.get("anchor_scores"))
                candidate["selection_confirmation_summary_path"] = selection_confirmation.get("summary_path")
                candidate["selection_paired_seeds"] = _paired_seed_count(selection_confirmation)
                candidate["selection_confirmation_metrics"] = confirmation_metrics
                candidate.update(confirmation_metrics)
            if reference_scores:
                comparison_scores = _anchor_scores(candidate.get("selection_anchor_scores"))
                if not comparison_scores:
                    comparison_scores = _anchor_scores(candidate.get("anchor_scores"))
                reference_comparison = _reference_comparison(
                    comparison_scores,
                    reference_anchor_scores=reference_scores,
                    reference_label=reference_label,
                )
                if reference_comparison is not None:
                    candidate["reference_comparison"] = reference_comparison
            if stack_config is not None:
                output_subdir = f"b1_candidate_confirm{int(confirm_paired_seeds)}_{candidate['snapshot_policy_id']}"
                candidate["confirmation_command"] = _confirmation_command(
                    stack_config=stack_config,
                    run_dir=run_dir,
                    snapshot_policy_id=str(candidate["snapshot_policy_id"]),
                    opponents=confirm_opponents,
                    paired_seeds=int(confirm_paired_seeds),
                    output_subdir=output_subdir,
                )
            candidates.append(candidate)
            run_candidates.append(candidate)
        if run_candidates:
            best_selection_score = max(float(item["selection_score"]) for item in run_candidates)
            latest_selection_score = float(run_candidates[-1]["selection_score"])
        run_summaries.append(
            {
                "run_dir": run_dir.as_posix(),
                "run_name": run_dir.name,
                "candidate_count": len(records),
                "best_selection_score": best_selection_score,
                "latest_selection_score": latest_selection_score,
                "latest_minus_best": (
                    None
                    if best_selection_score is None or latest_selection_score is None
                    else latest_selection_score - best_selection_score
                ),
                "baseline_alias": baseline_alias,
            }
        )

    def _sort_key(candidate: Mapping[str, Any]) -> tuple[bool, int, int, float, float, float, int, str]:
        return (
            bool(candidate["eligible"]),
            int(candidate.get("selection_paired_seeds", 0)),
            int(candidate.get("selection_score_source_rank", 0)),
            float(candidate["selection_score"]),
            float(candidate["required_anchor_min"]),
            float(candidate["aggregate_score"]),
            int(candidate["update_count"]),
            str(candidate["run_name"]),
        )

    ranked = sorted(candidates, key=_sort_key, reverse=True)
    selected = ranked[0] if ranked else None
    warnings: list[str] = []
    for run in run_summaries:
        baseline_alias = run.get("baseline_alias")
        if isinstance(baseline_alias, Mapping):
            metadata_format = baseline_alias.get("metadata_format")
            if metadata_format != "b1_candidate_alias_metadata_v1":
                warnings.append(
                    f"{run['run_name']} has a non-selector b1_noleague_baseline alias "
                    f"(metadata_format={metadata_format or 'missing'}); rerun select_b1_candidate.py "
                    "before importing it as canonical B1"
                )
        latest_minus_best = run["latest_minus_best"]
        if isinstance(latest_minus_best, (int, float)) and float(latest_minus_best) <= -float(
            falloff_warning_threshold
        ):
            warnings.append(
                f"{run['run_name']} fell off by {abs(float(latest_minus_best)):.4f} "
                f"selection-score points from best to latest"
            )
    if selected is not None and not bool(selected["eligible"]):
        warnings.append("no B1 candidate met the required anchor threshold")

    return {
        "kind": "b1_candidate_selection_v1",
        "required_anchors": list(required_anchors),
        "min_required_anchor_score": float(min_required_anchor_score),
        "falloff_warning_threshold": float(falloff_warning_threshold),
        "reference_label": str(reference_label) if reference_scores else None,
        "reference_anchor_scores": reference_scores,
        "run_summaries": run_summaries,
        "candidate_count": len(candidates),
        "selected": selected,
        "ranked_candidates": ranked,
        "warnings": warnings,
    }


def publish_b1_baseline_alias(
    *,
    run_dir: Path,
    source_policy_id: str,
    selection_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    registry_path = run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    manifest = _json_object(run_dir / "manifest.json")
    config_canonical = None if manifest is None else manifest.get("config_canonical")
    if not isinstance(config_canonical, Mapping) or not config_marks_noleague_baseline(config_canonical):
        raise RuntimeError(
            "Refusing to publish b1_noleague_baseline from a run that is not marked experiment.role='baseline_noleague'"
        )
    registry = SnapshotRegistry.load(registry_path)
    source_snapshot = _snapshot_by_policy_id(run_dir).get(str(source_policy_id))
    if source_snapshot is None:
        raise ValueError(f"source policy {source_policy_id!r} is not present in {registry_path}")

    source_path = source_snapshot.get("path")
    weights_sha256 = source_snapshot.get("weights_sha256")
    update = source_snapshot.get("update")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError(f"source policy {source_policy_id!r} is missing a snapshot path")
    if not isinstance(weights_sha256, str) or not weights_sha256:
        raise ValueError(f"source policy {source_policy_id!r} is missing weights_sha256")
    if isinstance(update, bool) or not isinstance(update, int):
        raise ValueError(f"source policy {source_policy_id!r} is missing integer update")

    source_weights_path = run_dir / source_path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"source weights not found: {source_weights_path}")

    target_relpath = snapshot_weights_relpath(NOLEAGUE_BASELINE_POLICY_ID)
    target_weights_path = run_dir / target_relpath
    target_weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_weights_path, target_weights_path)

    metadata_path = target_weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "b1_candidate_alias_metadata_v1",
        "policy_id": NOLEAGUE_BASELINE_POLICY_ID,
        "alias_for_policy_id": str(source_policy_id),
        "source_weights_path": str(source_path),
        "weights_path": target_relpath,
        "weights_sha256": weights_sha256,
        "update": int(update),
        "selection_summary": dict(selection_summary or {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry.add_snapshot(
        policy_id=NOLEAGUE_BASELINE_POLICY_ID,
        update=int(update),
        weights_sha256=weights_sha256,
        path=target_relpath,
    )
    registry.pin_snapshot(NOLEAGUE_BASELINE_POLICY_ID)
    registry.save(registry_path)
    return {
        "policy_id": NOLEAGUE_BASELINE_POLICY_ID,
        "alias_for_policy_id": str(source_policy_id),
        "update": int(update),
        "weights_path": target_relpath,
        "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
        "registry_path": registry_path.as_posix(),
    }


def publish_selected_candidate_alias(
    *,
    run_dir: Path,
    source_policy_id: str,
    alias_policy_id: str = SELECTED_CANDIDATE_POLICY_ID,
    selection_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    normalized_alias = str(alias_policy_id).strip()
    if not normalized_alias:
        raise ValueError("alias_policy_id must be non-empty")
    if normalized_alias == NOLEAGUE_BASELINE_POLICY_ID:
        raise ValueError("use publish_b1_baseline_alias for the canonical B1 baseline alias")

    registry_path = run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    if not registry_path.is_file():
        raise FileNotFoundError(f"snapshot registry not found: {registry_path}")
    registry = SnapshotRegistry.load(registry_path)
    source_snapshot = _snapshot_by_policy_id(run_dir).get(str(source_policy_id))
    if source_snapshot is None:
        raise ValueError(f"source policy {source_policy_id!r} is not present in {registry_path}")

    source_path = source_snapshot.get("path")
    weights_sha256 = source_snapshot.get("weights_sha256")
    update = source_snapshot.get("update")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError(f"source policy {source_policy_id!r} is missing a snapshot path")
    if not isinstance(weights_sha256, str) or not weights_sha256:
        raise ValueError(f"source policy {source_policy_id!r} is missing weights_sha256")
    if isinstance(update, bool) or not isinstance(update, int):
        raise ValueError(f"source policy {source_policy_id!r} is missing integer update")

    source_weights_path = run_dir / source_path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"source weights not found: {source_weights_path}")

    target_relpath = snapshot_weights_relpath(normalized_alias)
    target_weights_path = run_dir / target_relpath
    target_weights_path.parent.mkdir(parents=True, exist_ok=True)
    if source_weights_path.resolve() != target_weights_path.resolve():
        shutil.copy2(source_weights_path, target_weights_path)

    metadata_path = target_weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "selected_candidate_alias_metadata_v1",
        "policy_id": normalized_alias,
        "alias_for_policy_id": str(source_policy_id),
        "source_weights_path": str(source_path),
        "weights_path": target_relpath,
        "weights_sha256": weights_sha256,
        "update": int(update),
        "selection_summary": dict(selection_summary or {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    registry.add_snapshot(
        policy_id=normalized_alias,
        update=int(update),
        weights_sha256=weights_sha256,
        path=target_relpath,
    )
    registry.pin_snapshot(normalized_alias)
    registry.save(registry_path)
    return {
        "policy_id": normalized_alias,
        "alias_for_policy_id": str(source_policy_id),
        "update": int(update),
        "weights_path": target_relpath,
        "metadata_path": metadata_path.relative_to(run_dir).as_posix(),
        "registry_path": registry_path.as_posix(),
    }
