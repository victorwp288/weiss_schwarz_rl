from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_payloads import anchor_scores, finite_float, json_object
from weiss_rl.experiments.baselines import NOLEAGUE_BASELINE_POLICY_ID
from weiss_rl.league.registry import REGISTRY_FILENAME, SNAPSHOT_METADATA_FILENAME


def snapshot_index(run_dir: Path) -> dict[int, dict[str, Any]]:
    registry = json_object(run_dir / "training" / "snapshots" / REGISTRY_FILENAME)
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


def snapshot_by_policy_id(run_dir: Path) -> dict[str, dict[str, Any]]:
    registry = json_object(run_dir / "training" / "snapshots" / REGISTRY_FILENAME)
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


def paired_seed_count(payload: Mapping[str, Any] | None) -> int:
    if payload is None:
        return 0
    paired_seeds = payload.get("paired_seeds")
    if isinstance(paired_seeds, bool) or not isinstance(paired_seeds, int):
        return 0
    return int(paired_seeds)


def baseline_alias_info(run_dir: Path) -> dict[str, Any] | None:
    snapshot = snapshot_by_policy_id(run_dir).get(NOLEAGUE_BASELINE_POLICY_ID)
    if snapshot is None:
        return None
    metadata = json_object(
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


def load_b1_dev_eval_records(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = run_dir.resolve()
    snapshot_by_update = snapshot_index(run_dir)
    snapshots_by_policy = snapshot_by_policy_id(run_dir)
    records: list[dict[str, Any]] = []
    summaries = json_object(run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json")
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
                snapshot_by_policy_id=snapshots_by_policy,
            )
            if record is not None:
                records.append(record)
    if not records:
        for summary_path in sorted((run_dir / "eval" / "dev_eval").glob("update_*/summary.json")):
            summary = json_object(summary_path)
            if summary is None:
                continue
            record = _record_from_summary(
                run_dir=run_dir,
                summary=summary,
                fallback_policy_id=summary_path.parent.name,
                source_path=summary_path,
                snapshot_by_update=snapshot_by_update,
                snapshot_by_policy_id=snapshots_by_policy,
            )
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda record: (int(record["update_count"]), str(record["snapshot_policy_id"])))


def confirmation_scores(
    run_dir: Path,
    snapshot_policy_id: str,
    *,
    required_anchors: Sequence[str],
    min_paired_seeds: int = 0,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    snapshots_by_policy = snapshot_by_policy_id(run_dir)
    target_snapshot = snapshots_by_policy.get(str(snapshot_policy_id), {})
    target_weights_sha256 = target_snapshot.get("weights_sha256")
    summaries = sorted((run_dir / "eval").glob("**/targeted_confirm*_summary.json"))
    for summary_path in summaries:
        summary = json_object(summary_path)
        if summary is None:
            continue
        focal_policy_id = str(summary.get("focal_policy_id", "")).strip()
        if focal_policy_id != snapshot_policy_id:
            focal_snapshot = snapshots_by_policy.get(focal_policy_id, {})
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
        scores_by_anchor: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            opponent = str(row.get("opponent_policy_id", ""))
            mean = finite_float(row.get("mean"))
            if opponent and mean is not None:
                scores_by_anchor[opponent] = mean
        missing_required = [anchor for anchor in required_anchors if anchor not in scores_by_anchor]
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
                "anchor_scores": scores_by_anchor,
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


def targeted_confirm_only_records(
    run_dir: Path,
    *,
    required_anchors: Sequence[str],
    existing_snapshot_policy_ids: set[str],
    min_paired_seeds: int = 0,
) -> list[dict[str, Any]]:
    snapshots_by_policy = snapshot_by_policy_id(run_dir)
    focal_policy_ids: set[str] = set()
    for summary_path in sorted((run_dir / "eval").glob("**/targeted_confirm*_summary.json")):
        summary = json_object(summary_path)
        if summary is None:
            continue
        focal_policy_id = str(summary.get("focal_policy_id", "")).strip()
        if (
            focal_policy_id
            and focal_policy_id not in existing_snapshot_policy_ids
            and focal_policy_id in snapshots_by_policy
        ):
            focal_policy_ids.add(focal_policy_id)

    records: list[dict[str, Any]] = []
    for focal_policy_id in sorted(focal_policy_ids):
        confirmation = confirmation_scores(
            run_dir,
            focal_policy_id,
            required_anchors=required_anchors,
            min_paired_seeds=int(min_paired_seeds),
        )
        if confirmation is None:
            continue
        snapshot = snapshots_by_policy[focal_policy_id]
        update = snapshot.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            continue
        scores_by_anchor = anchor_scores(confirmation.get("anchor_scores"))
        summary_path = Path(str(confirmation["summary_path"]))
        confirmation_summary = json_object(summary_path)
        aggregate_score = None if confirmation_summary is None else finite_float(confirmation_summary.get("mean"))
        if aggregate_score is None:
            aggregate_score = sum(scores_by_anchor.values()) / len(scores_by_anchor) if scores_by_anchor else 0.0
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
                "anchor_scores": scores_by_anchor,
                "source_path": str(confirmation["summary_path"]),
                "targeted_confirm_only": True,
            }
        )
    return records


def confirmatory_dev_eval(run_dir: Path, update_count: int) -> dict[str, Any] | None:
    summary_path = run_dir / "eval" / "dev_eval_confirmatory" / f"update_{int(update_count)}" / "summary.json"
    summary = json_object(summary_path)
    if summary is None:
        return None
    aggregate_score = finite_float(summary.get("aggregate_score"))
    return {
        "summary_path": summary_path.as_posix(),
        "aggregate_score": aggregate_score,
        "paired_seeds": paired_seed_count(summary),
        "anchor_scores": anchor_scores(summary.get("anchor_scores")),
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
    aggregate_score = finite_float(summary.get("aggregate_score"))
    update_count = summary.get("update_count")
    if aggregate_score is None or isinstance(update_count, bool) or not isinstance(update_count, int):
        return None

    policy_id = str(summary.get("policy_id") or summary.get("focal_policy_id") or fallback_policy_id)
    policy_version = summary.get("policy_version")
    version_policy_id = f"policy_{int(policy_version):06d}" if isinstance(policy_version, int) else None
    snapshot: Mapping[str, Any] = {}
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
        "anchor_scores": anchor_scores(summary.get("anchor_scores")),
        "paired_seeds": int(paired_seeds)
        if not isinstance(paired_seeds, bool) and isinstance(paired_seeds, int)
        else 0,
        "source_path": source_path.as_posix(),
    }


def _policy_version_from_policy_id(policy_id: str) -> int | None:
    normalized = policy_id.strip()
    if not normalized.startswith("policy_"):
        return None
    suffix = normalized.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None
