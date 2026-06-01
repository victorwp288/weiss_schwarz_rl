from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from weiss_rl.experiments.b1_candidate_discovery import (
    baseline_alias_info,
    confirmation_scores,
    confirmatory_dev_eval,
    load_b1_dev_eval_records,
    paired_seed_count,
    targeted_confirm_only_records,
)
from weiss_rl.experiments.b1_candidate_payloads import anchor_scores, reference_comparison
from weiss_rl.experiments.bootstrap_commands import build_targeted_confirm_entrypoint_command

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
B1_CANDIDATE_SELECTION_KIND = "b1_candidate_selection_v1"
B1_CANDIDATE_ALIAS_METADATA_FORMAT = "b1_candidate_alias_metadata_v1"


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
            targeted_confirm_only_records(
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
        baseline_alias = baseline_alias_info(run_dir)
        for record in records:
            candidate = build_candidate_record(
                record,
                run_dir=run_dir,
                stack_config=stack_config,
                required_anchors=required_anchors,
                confirm_opponents=confirm_opponents,
                min_required_anchor_score=float(min_required_anchor_score),
                confirm_paired_seeds=int(confirm_paired_seeds),
                reference_scores=reference_scores,
                reference_label=reference_label,
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

    ranked = sorted(candidates, key=candidate_sort_key, reverse=True)
    selected = ranked[0] if ranked else None
    warnings = selection_warnings(
        run_summaries,
        selected=selected,
        falloff_warning_threshold=float(falloff_warning_threshold),
    )

    return {
        "kind": B1_CANDIDATE_SELECTION_KIND,
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


def build_candidate_record(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
    stack_config: Path | None,
    required_anchors: Sequence[str],
    confirm_opponents: Sequence[str],
    min_required_anchor_score: float,
    confirm_paired_seeds: int,
    reference_scores: Mapping[str, float],
    reference_label: str,
) -> dict[str, Any]:
    metrics = candidate_metrics(
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
    candidate["selection_score_source_rank"] = selection_source_rank(candidate["selection_score_source"])
    candidate["selection_paired_seeds"] = paired_seed_count(candidate)

    confirmatory = confirmatory_dev_eval(run_dir, int(candidate["update_count"]))
    if confirmatory is not None:
        candidate["confirmatory_dev_eval"] = confirmatory
    confirmation = confirmation_scores(
        run_dir,
        str(candidate["snapshot_policy_id"]),
        required_anchors=required_anchors,
        min_paired_seeds=int(confirm_paired_seeds),
    )
    if confirmation is not None:
        candidate["confirmation"] = confirmation
    selection_source, selection_confirmation = selection_confirmation_source(
        targeted_confirmation=confirmation,
        confirmatory_dev_eval=confirmatory,
    )
    if selection_confirmation is not None:
        confirmation_metrics = candidate_metrics_from_anchor_scores(
            anchor_scores(selection_confirmation.get("anchor_scores")),
            required_anchors=required_anchors,
            min_required_anchor_score=float(min_required_anchor_score),
        )
        candidate["selection_score_source"] = selection_source
        candidate["selection_score_source_rank"] = selection_source_rank(selection_source)
        candidate["selection_anchor_scores"] = anchor_scores(selection_confirmation.get("anchor_scores"))
        candidate["selection_confirmation_summary_path"] = selection_confirmation.get("summary_path")
        candidate["selection_paired_seeds"] = paired_seed_count(selection_confirmation)
        candidate["selection_confirmation_metrics"] = confirmation_metrics
        candidate.update(confirmation_metrics)
    if reference_scores:
        comparison_scores = anchor_scores(candidate.get("selection_anchor_scores"))
        if not comparison_scores:
            comparison_scores = anchor_scores(candidate.get("anchor_scores"))
        candidate_reference_comparison = reference_comparison(
            comparison_scores,
            reference_anchor_scores=reference_scores,
            reference_label=reference_label,
        )
        if candidate_reference_comparison is not None:
            candidate["reference_comparison"] = candidate_reference_comparison
    if stack_config is not None:
        output_subdir = f"b1_candidate_confirm{int(confirm_paired_seeds)}_{candidate['snapshot_policy_id']}"
        candidate["confirmation_command"] = confirmation_command(
            stack_config=stack_config,
            run_dir=run_dir,
            snapshot_policy_id=str(candidate["snapshot_policy_id"]),
            opponents=confirm_opponents,
            paired_seeds=int(confirm_paired_seeds),
            output_subdir=output_subdir,
        )
    return candidate


def candidate_metrics_from_anchor_scores(
    scores_by_anchor: Mapping[str, float],
    *,
    required_anchors: Sequence[str],
    min_required_anchor_score: float,
) -> dict[str, Any]:
    required_scores = [scores_by_anchor[name] for name in required_anchors if name in scores_by_anchor]
    missing = [name for name in required_anchors if name not in scores_by_anchor]
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
                    f"{anchor} {scores_by_anchor[anchor]:.4f} < {min_required_anchor_score:.4f}"
                    for anchor in required_anchors
                    if anchor in scores_by_anchor and scores_by_anchor[anchor] < min_required_anchor_score
                ],
            ]
        ),
    }


def candidate_metrics(
    record: Mapping[str, Any],
    *,
    required_anchors: Sequence[str],
    min_required_anchor_score: float,
) -> dict[str, Any]:
    return candidate_metrics_from_anchor_scores(
        anchor_scores(record.get("anchor_scores")),
        required_anchors=required_anchors,
        min_required_anchor_score=min_required_anchor_score,
    )


def selection_confirmation_source(
    *,
    targeted_confirmation: Mapping[str, Any] | None,
    confirmatory_dev_eval: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any] | None]:
    if targeted_confirmation is not None:
        return "targeted_confirm", targeted_confirmation
    if confirmatory_dev_eval is not None:
        return "confirmatory_dev_eval", confirmatory_dev_eval
    return "periodic_dev_eval", None


def selection_source_rank(source: object) -> int:
    if source == "targeted_confirm":
        return 2
    if source == "confirmatory_dev_eval":
        return 1
    return 0


def candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[bool, int, int, float, float, float, int, str]:
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


def selection_warnings(
    run_summaries: Sequence[Mapping[str, Any]],
    *,
    selected: Mapping[str, Any] | None,
    falloff_warning_threshold: float,
) -> list[str]:
    warnings: list[str] = []
    for run in run_summaries:
        baseline_alias = run.get("baseline_alias")
        if isinstance(baseline_alias, Mapping):
            metadata_format = baseline_alias.get("metadata_format")
            if metadata_format != B1_CANDIDATE_ALIAS_METADATA_FORMAT:
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
    return warnings


def confirmation_command(
    *,
    stack_config: Path,
    run_dir: Path,
    snapshot_policy_id: str,
    opponents: Sequence[str],
    paired_seeds: int,
    output_subdir: str,
) -> list[str]:
    return build_targeted_confirm_entrypoint_command(
        repo_root=None,
        stack_config=stack_config,
        run_dir=run_dir,
        b1_baseline_run_dir=run_dir,
        focal_policy_id=snapshot_policy_id,
        paired_seeds=int(paired_seeds),
        bootstrap_samples=2000,
        output_subdir=output_subdir,
        opponents=opponents,
        python_command=("uv", "run", "--extra", "dev", "--extra", "sim", "python"),
    )
