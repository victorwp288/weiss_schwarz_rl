"""Final-eval orchestration for the deterministic final policy set."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config.models import FinalPolicySetSelectionConfig, StopRulesConfig
from weiss_rl.eval.diagnostics import build_seat_advantage_diagnostics, write_matchup_diagnostics_json
from weiss_rl.eval.export import build_matchup_export, write_matchup_summary_csv, write_matchup_summary_json
from weiss_rl.eval.harness import (
    EvalGameRecord,
    EvalGameRunner,
    ReplaySampleResult,
    ScheduledGame,
    record_completed_game,
    write_episodes_jsonl,
)
from weiss_rl.eval.payoff_folding import PayoffFoldScheme, paired_seed_scores
from weiss_rl.eval.policy_set import DevEvalPolicySummary, select_final_policy_set_deterministic_v1
from weiss_rl.eval.stage2 import summarize_stage2_records
from weiss_rl.eval.uncertainty import bayesian_bootstrap_posterior_samples
from weiss_rl.repro import canonical_json_bytes, hash_seed_file, stable_hash64

__all__ = [
    "load_dev_eval_summaries",
    "resolve_final_policy_set",
    "run_final_eval",
]


_MATRIX_FIELDS: tuple[str, ...] = (
    "mean",
    "ci_low",
    "ci_high",
    "ci_half_width",
    "prob_gt_half",
    "prob_lt_half",
    "paired_seed_count",
    "observed_paired_seeds",
    "excluded_paired_seeds",
    "has_payoff_samples",
    "games",
    "wins",
    "losses",
    "draws",
    "truncations",
    "engine_errors",
    "stop_reason",
    "should_stop",
)


def load_dev_eval_summaries(path: Path) -> dict[str, float | DevEvalPolicySummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dev-eval summaries JSON must contain an object at the top level")

    summaries: dict[str, float | DevEvalPolicySummary] = {}
    for policy_id, raw_summary in payload.items():
        if isinstance(raw_summary, bool):
            raise TypeError(f"dev-eval summary for {policy_id!r} cannot be a boolean")
        if isinstance(raw_summary, (int, float)):
            summaries[policy_id] = float(raw_summary)
            continue
        if not isinstance(raw_summary, dict):
            raise TypeError(
                "dev-eval summary values must be numbers or objects with aggregate_score/anchor_scores, "
                f"got {type(raw_summary).__name__} for {policy_id!r}"
            )
        aggregate_score = raw_summary.get("aggregate_score")
        if isinstance(aggregate_score, bool) or not isinstance(aggregate_score, (int, float)):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include numeric aggregate_score")
        anchor_scores = raw_summary.get("anchor_scores", {})
        if not isinstance(anchor_scores, dict) or any(not isinstance(key, str) for key in anchor_scores):
            raise TypeError(f"dev-eval summary for {policy_id!r} must include object anchor_scores")
        summaries[policy_id] = DevEvalPolicySummary(
            policy_id=policy_id,
            aggregate_score=float(aggregate_score),
            anchor_scores=anchor_scores,
        )
    return summaries


def resolve_final_policy_set(
    *,
    snapshot_registry_path: Path,
    dev_eval_summaries_path: Path,
    config: FinalPolicySetSelectionConfig,
    final_policy_set_size: int,
) -> list[str]:
    from weiss_rl.league.registry import SnapshotRegistry

    registry = SnapshotRegistry.load(snapshot_registry_path)
    summaries = load_dev_eval_summaries(dev_eval_summaries_path)
    return select_final_policy_set_deterministic_v1(
        snapshot_registry=registry,
        dev_eval_summaries=summaries,
        config=config,
        final_policy_set_size=final_policy_set_size,
    )


def run_final_eval(
    *,
    output_dir: Path,
    runner: EvalGameRunner,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: StopRulesConfig,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme = "S0",
    sample_count: int = 1000,
    policy_ids: Sequence[str] | None = None,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
    selection_config: FinalPolicySetSelectionConfig | None = None,
    final_policy_set_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    seed_file_path: Path | None = None,
) -> dict[str, Any]:
    resolved_policy_ids, selection_payload = _resolve_policy_ids(
        policy_ids=policy_ids,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        selection_config=selection_config,
        final_policy_set_size=final_policy_set_size,
    )
    _validate_seed_budget(
        paired_seeds=paired_seeds,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matchup_results: list[dict[str, Any]] = []
    for focal_index, focal_policy_id in enumerate(resolved_policy_ids):
        for opponent_index, opponent_policy_id in enumerate(resolved_policy_ids[focal_index:], start=focal_index):
            matchup_results.append(
                _run_matchup(
                    output_dir=output_dir,
                    focal_index=focal_index,
                    opponent_index=opponent_index,
                    focal_policy_id=focal_policy_id,
                    opponent_policy_id=opponent_policy_id,
                    paired_seeds=paired_seeds,
                    stage1_paired_seeds=stage1_paired_seeds,
                    max_paired_seeds=max_paired_seeds,
                    stop_rules=stop_rules,
                    runner=runner,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                    scheme=scheme,
                    sample_count=sample_count,
                )
            )

    payload = _build_final_eval_payload(
        output_dir=output_dir,
        policy_ids=resolved_policy_ids,
        matchup_results=matchup_results,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        paired_seeds=paired_seeds,
        stop_rules=stop_rules,
        scheme=scheme,
        sample_count=sample_count,
        selection_payload=selection_payload,
        metadata=metadata,
        seed_file_path=seed_file_path,
    )
    _write_final_eval_artifacts(output_dir=output_dir, payload=payload, matchup_results=matchup_results)
    return payload


def _resolve_policy_ids(
    *,
    policy_ids: Sequence[str] | None,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    selection_config: FinalPolicySetSelectionConfig | None,
    final_policy_set_size: int | None,
) -> tuple[list[str], dict[str, Any]]:
    if policy_ids is not None:
        resolved = [str(policy_id) for policy_id in policy_ids]
        if not resolved:
            raise ValueError("policy_ids must contain at least one policy")
        _validate_policy_ids(resolved, context="policy_ids")
        return resolved, {"mode": "explicit", "policy_count": len(resolved)}

    missing: list[str] = []
    if snapshot_registry_path is None:
        missing.append("snapshot_registry_path")
    if dev_eval_summaries_path is None:
        missing.append("dev_eval_summaries_path")
    if selection_config is None:
        missing.append("selection_config")
    if final_policy_set_size is None:
        missing.append("final_policy_set_size")
    if missing:
        raise ValueError(f"run_final_eval requires policy_ids or selection inputs, missing: {', '.join(missing)}")

    assert snapshot_registry_path is not None
    assert dev_eval_summaries_path is not None
    assert selection_config is not None
    assert final_policy_set_size is not None
    resolved = resolve_final_policy_set(
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        config=selection_config,
        final_policy_set_size=final_policy_set_size,
    )
    _validate_policy_ids(resolved, context="resolved final policy set")
    if len(resolved) < int(final_policy_set_size):
        raise ValueError(
            "resolved final policy set is underfilled: "
            f"expected {int(final_policy_set_size)} policies, found {len(resolved)}"
        )
    return resolved, {
        "mode": "deterministic_v1",
        "policy_count": len(resolved),
        "snapshot_registry_path": snapshot_registry_path.as_posix(),
        "dev_eval_summaries_path": dev_eval_summaries_path.as_posix(),
        "final_policy_set_size": int(final_policy_set_size),
    }


def _validate_policy_ids(policy_ids: Sequence[str], *, context: str) -> None:
    duplicates = sorted(policy_id for policy_id, count in Counter(policy_ids).items() if count > 1)
    if duplicates:
        duplicate_list = ", ".join(repr(policy_id) for policy_id in duplicates)
        raise ValueError(f"{context} must be unique, duplicate entries: {duplicate_list}")


def _validate_seed_budget(*, paired_seeds: Sequence[int], stage1_paired_seeds: int, max_paired_seeds: int) -> None:
    if stage1_paired_seeds < 1:
        raise ValueError("stage1_paired_seeds must be positive")
    if max_paired_seeds < stage1_paired_seeds:
        raise ValueError("max_paired_seeds must be >= stage1_paired_seeds")
    if len(paired_seeds) < max_paired_seeds:
        raise ValueError(f"final eval requires at least {max_paired_seeds} paired seeds, found {len(paired_seeds)}")


def _run_matchup(
    *,
    output_dir: Path,
    focal_index: int,
    opponent_index: int,
    focal_policy_id: str,
    opponent_policy_id: str,
    paired_seeds: Sequence[int],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    stop_rules: StopRulesConfig,
    runner: EvalGameRunner,
    run_id256: str | bytes,
    config_hash256: str,
    spec_hash256: str,
    scheme: PayoffFoldScheme,
    sample_count: int,
) -> dict[str, Any]:
    matchup_dir = (
        output_dir
        / "matchups"
        / _matchup_dir_name(
            focal_index=focal_index,
            opponent_index=opponent_index,
            focal_policy_id=focal_policy_id,
            opponent_policy_id=opponent_policy_id,
        )
    )
    records: list[EvalGameRecord] = []
    replay_samples: list[ReplaySampleResult] = []
    used_paired_seeds: list[int] = []

    for pair_index, episode_seed in enumerate(paired_seeds[:max_paired_seeds]):
        for swap_index in (0, 1):
            scheduled_game = _scheduled_game(
                pair_index=pair_index,
                swap_index=swap_index,
                episode_seed=int(episode_seed),
                focal_policy_id=focal_policy_id,
                opponent_policy_id=opponent_policy_id,
            )
            result = runner.run_game(scheduled_game)
            if result.replay_sample is not None:
                replay_samples.append(result.replay_sample)
            records.append(
                record_completed_game(
                    scheduled_game=scheduled_game,
                    result=result,
                    run_id256=run_id256,
                    config_hash256=config_hash256,
                    spec_hash256=spec_hash256,
                )
            )
        used_paired_seeds.append(int(episode_seed))
        if len(used_paired_seeds) < stage1_paired_seeds:
            continue
        decision = summarize_stage2_records(
            records,
            stop_rules=stop_rules,
            max_paired_seeds=max_paired_seeds,
            scheme=scheme,
            sample_count=sample_count,
            seed=_bootstrap_seed(focal_policy_id=focal_policy_id, opponent_policy_id=opponent_policy_id),
        )
        if decision.should_stop:
            break

    episodes_path = matchup_dir / "episodes.jsonl"
    write_episodes_jsonl(episodes_path, records)

    bootstrap_seed = _bootstrap_seed(focal_policy_id=focal_policy_id, opponent_policy_id=opponent_policy_id)
    summary_payload = _build_matchup_payload(
        records=records,
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
        scheme=scheme,
        sample_count=sample_count,
        seed=bootstrap_seed,
    )
    summary_payload["evaluation_context"] = {
        "artifact_scope": "final_eval",
        "focal_policy_index": focal_index,
        "opponent_policy_index": opponent_index,
        "stage1_paired_seeds": stage1_paired_seeds,
        "max_paired_seeds": max_paired_seeds,
        "used_paired_seeds": list(used_paired_seeds),
    }
    diagnostics_payload = build_seat_advantage_diagnostics(records)
    posterior_samples = _matchup_posterior_samples(
        records=records,
        scheme=scheme,
        sample_count=sample_count,
        seed=bootstrap_seed,
    )

    write_matchup_summary_json(matchup_dir / "matchup_summary.json", summary_payload)
    write_matchup_summary_csv(matchup_dir / "matchup_summary.csv", summary_payload)
    write_matchup_diagnostics_json(matchup_dir / "diagnostics.json", diagnostics_payload)
    (matchup_dir / "posterior_samples.json").write_text(
        json.dumps(
            {
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
                "requested_sample_count": sample_count,
                "sample_count": len(posterior_samples),
                "has_payoff_samples": summary_payload["has_payoff_samples"],
                "samples": list(posterior_samples),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "focal_policy_id": focal_policy_id,
        "opponent_policy_id": opponent_policy_id,
        "focal_index": focal_index,
        "opponent_index": opponent_index,
        "matchup_dir": matchup_dir,
        "episodes_path": episodes_path,
        "summary": summary_payload,
        "diagnostics": diagnostics_payload,
        "posterior_samples": posterior_samples,
        "used_paired_seeds": tuple(used_paired_seeds),
        "records": tuple(records),
        "replay_samples": tuple(replay_samples),
    }


def _scheduled_game(
    *,
    pair_index: int,
    swap_index: int,
    episode_seed: int,
    focal_policy_id: str,
    opponent_policy_id: str,
) -> ScheduledGame:
    if swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1
    return ScheduledGame(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=pair_index * 2 + swap_index,
        episode_seed=episode_seed,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
    )


def _build_matchup_payload(
    *,
    records: Sequence[EvalGameRecord],
    stop_rules: StopRulesConfig,
    max_paired_seeds: int,
    scheme: PayoffFoldScheme,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    return build_matchup_export(
        tuple(records),
        stop_rules=stop_rules,
        max_paired_seeds=max_paired_seeds,
        scheme=scheme,
        sample_count=sample_count,
        seed=seed,
    )


def _matchup_posterior_samples(
    *,
    records: Sequence[EvalGameRecord],
    scheme: PayoffFoldScheme,
    sample_count: int,
    seed: int,
) -> tuple[float, ...]:
    scores = paired_seed_scores(records, scheme=scheme)
    if not scores:
        return ()
    return bayesian_bootstrap_posterior_samples(scores, sample_count=sample_count, seed=seed)


def _build_final_eval_payload(
    *,
    output_dir: Path,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
    stage1_paired_seeds: int,
    max_paired_seeds: int,
    paired_seeds: Sequence[int],
    stop_rules: StopRulesConfig,
    scheme: PayoffFoldScheme,
    sample_count: int,
    selection_payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    seed_file_path: Path | None,
) -> dict[str, Any]:
    canonical_results_by_key = {
        (int(result["focal_index"]), int(result["opponent_index"])): result for result in matchup_results
    }
    matrices = {
        field: _build_matrix(
            policy_ids=policy_ids,
            canonical_results_by_key=canonical_results_by_key,
            field=field,
        )
        for field in _MATRIX_FIELDS
    }
    posterior_matrix = [
        [
            _posterior_samples_cell(
                canonical_results_by_key=canonical_results_by_key,
                focal_index=focal_index,
                opponent_index=opponent_index,
            )
            for opponent_index, _opponent_policy_id in enumerate(policy_ids)
        ]
        for focal_index, _focal_policy_id in enumerate(policy_ids)
    ]
    top_level_metadata = dict(metadata or {})
    top_level_metadata.update(
        {
            "policy_count": len(policy_ids),
            "matchup_count": len(matchup_results),
            "matchup_artifacts": {
                "kind": "canonical_unordered_pairs_v1",
                "canonical_order": "focal_policy_index <= opponent_policy_index",
                "reverse_matrix_cells": "derived_from_canonical_matchup_artifacts",
            },
            "stage1_paired_seeds": stage1_paired_seeds,
            "max_paired_seeds": max_paired_seeds,
            "paired_seed_budget": len(paired_seeds),
            "stop_rules": {
                "stop_delta_ci_half_width": float(stop_rules.stop_delta_ci_half_width),
                "stop_confidence": float(stop_rules.stop_confidence),
            },
            "scheme": scheme,
            "sample_count": sample_count,
            "selection": dict(selection_payload),
        }
    )
    if seed_file_path is not None:
        top_level_metadata["seed_file"] = {
            "path": seed_file_path.as_posix(),
            "sha256": hash_seed_file(seed_file_path),
        }

    return {
        "output_dir": output_dir.as_posix(),
        "policy_ids": list(policy_ids),
        "metadata": top_level_metadata,
        "matrices": matrices,
        "posterior_samples": {
            "policy_ids": list(policy_ids),
            "sample_count": sample_count,
            "values": posterior_matrix,
        },
        "matchups": [
            {
                "focal_policy_id": result["focal_policy_id"],
                "opponent_policy_id": result["opponent_policy_id"],
                "focal_policy_index": result["focal_index"],
                "opponent_policy_index": result["opponent_index"],
                "matchup_dir": _relative_to(result["matchup_dir"], root=output_dir),
                "episodes_path": _relative_to(result["episodes_path"], root=output_dir),
                "summary_path": _relative_to(Path(result["matchup_dir"]) / "matchup_summary.json", root=output_dir),
                "diagnostics_path": _relative_to(Path(result["matchup_dir"]) / "diagnostics.json", root=output_dir),
                "posterior_samples_path": _relative_to(
                    Path(result["matchup_dir"]) / "posterior_samples.json",
                    root=output_dir,
                ),
                "matrix_cells": _covered_matrix_cells(
                    focal_index=int(result["focal_index"]),
                    opponent_index=int(result["opponent_index"]),
                ),
                "paired_seed_count": result["summary"]["paired_seeds"],
                "observed_paired_seed_count": result["summary"]["observed_paired_seeds"],
                "excluded_paired_seed_count": result["summary"]["excluded_paired_seeds"],
                "has_payoff_samples": result["summary"]["has_payoff_samples"],
                "stop_reason": result["summary"]["stop_reason"],
            }
            for result in matchup_results
        ],
    }


def _build_matrix(
    *,
    policy_ids: Sequence[str],
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [
        [
            _matrix_cell_value(
                canonical_results_by_key=canonical_results_by_key,
                focal_index=focal_index,
                opponent_index=opponent_index,
                field=field,
            )
            for opponent_index, _opponent_policy_id in enumerate(policy_ids)
        ]
        for focal_index, _focal_policy_id in enumerate(policy_ids)
    ]
    return {
        "policy_ids": list(policy_ids),
        "values": values,
    }


def _matrix_cell_value(
    *,
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    focal_index: int,
    opponent_index: int,
    field: str,
) -> Any:
    result, reverse = _canonical_result_for_cell(
        canonical_results_by_key=canonical_results_by_key,
        focal_index=focal_index,
        opponent_index=opponent_index,
    )
    payload = cast(Mapping[str, Any], result["summary"])
    return _matrix_value(payload, field=field, reverse=reverse)


def _matrix_value(payload: Mapping[str, Any], *, field: str, reverse: bool = False) -> Any:
    summary = cast(Mapping[str, Any], payload["summary"])
    uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
    if reverse:
        return _reverse_matrix_value(payload=payload, summary=summary, uncertainty=uncertainty, field=field)
    if field in uncertainty:
        return uncertainty[field]
    if field in summary:
        return summary[field]
    if field == "paired_seed_count":
        return uncertainty["paired_seed_count"]
    return payload[field]


def _reverse_matrix_value(
    *,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    field: str,
) -> Any:
    if field == "mean":
        return _invert_optional_float(uncertainty["mean"])
    if field == "ci_low":
        return _invert_optional_float(uncertainty["ci_high"])
    if field == "ci_high":
        return _invert_optional_float(uncertainty["ci_low"])
    if field == "prob_gt_half":
        return uncertainty["prob_lt_half"]
    if field == "prob_lt_half":
        return uncertainty["prob_gt_half"]
    if field == "wins":
        return summary["losses"]
    if field == "losses":
        return summary["wins"]
    if field in uncertainty:
        return uncertainty[field]
    if field in summary:
        return summary[field]
    if field == "paired_seed_count":
        return uncertainty["paired_seed_count"]
    return payload[field]


def _invert_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return 1.0 - float(value)


def _canonical_result_for_cell(
    *,
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    focal_index: int,
    opponent_index: int,
) -> tuple[dict[str, Any], bool]:
    canonical_key = (min(focal_index, opponent_index), max(focal_index, opponent_index))
    return canonical_results_by_key[canonical_key], focal_index > opponent_index


def _posterior_samples_cell(
    *,
    canonical_results_by_key: Mapping[tuple[int, int], dict[str, Any]],
    focal_index: int,
    opponent_index: int,
) -> list[float]:
    result, reverse = _canonical_result_for_cell(
        canonical_results_by_key=canonical_results_by_key,
        focal_index=focal_index,
        opponent_index=opponent_index,
    )
    samples = cast(Sequence[float], result["posterior_samples"])
    if not reverse:
        return [float(sample) for sample in samples]
    return [1.0 - float(sample) for sample in samples]


def _covered_matrix_cells(*, focal_index: int, opponent_index: int) -> list[dict[str, int]]:
    cells = [{"focal_policy_index": focal_index, "opponent_policy_index": opponent_index}]
    if focal_index != opponent_index:
        cells.append({"focal_policy_index": opponent_index, "opponent_policy_index": focal_index})
    return cells


def _write_final_eval_artifacts(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    layout = _maybe_layout(output_dir)
    metadata_path = layout.final_eval_metadata_json() if layout is not None else output_dir / "metadata.json"
    policy_set_path = layout.final_eval_policy_set_json() if layout is not None else output_dir / "policy_set.json"
    summary_path = layout.final_eval_summary_json() if layout is not None else output_dir / "summary.json"
    posterior_samples_json_path = (
        layout.final_eval_posterior_samples_json() if layout is not None else output_dir / "posterior_samples.json"
    )

    metadata_path.write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_set_path.write_text(
        json.dumps({"policy_ids": payload["policy_ids"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    posterior_samples_json_path.write_text(
        json.dumps(payload["posterior_samples"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if layout is not None:
        posterior_payload = cast(Mapping[str, Any], payload["posterior_samples"])
        np.savez_compressed(
            layout.final_eval_posterior_samples_npz(),
            values=np.asarray(posterior_payload.get("values", ()), dtype=np.float64),
            policy_ids=np.asarray(posterior_payload.get("policy_ids", ()), dtype=object),
        )

    matrices_dir = layout.final_eval_matrices_dir if layout is not None else output_dir / "matrices"
    matrices = cast(Mapping[str, Mapping[str, Any]], payload["matrices"])
    for field, matrix_payload in matrices.items():
        json_path = matrices_dir / f"{field}.json"
        csv_path = matrices_dir / f"{field}.csv"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_matrix_csv(csv_path, matrix_payload)
        if layout is not None:
            legacy_name = _legacy_payoff_matrix_name(field)
            if legacy_name is not None:
                _write_matrix_csv(layout.final_eval_payoff_matrix_csv(legacy_name), matrix_payload)

    manifest_rows = [
        {
            "focal_policy_id": result["focal_policy_id"],
            "opponent_policy_id": result["opponent_policy_id"],
            "matchup_dir": _relative_to(Path(result["matchup_dir"]), root=output_dir),
            "paired_seed_count": result["summary"]["paired_seeds"],
            "observed_paired_seed_count": result["summary"]["observed_paired_seeds"],
            "excluded_paired_seed_count": result["summary"]["excluded_paired_seeds"],
            "has_payoff_samples": result["summary"]["has_payoff_samples"],
            "stop_reason": result["summary"]["stop_reason"],
        }
        for result in matchup_results
    ]
    manifest_path = layout.final_eval_matchups_csv() if layout is not None else output_dir / "matchups.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    aggregate_records = [
        record for result in matchup_results for record in cast(Sequence[EvalGameRecord], result.get("records", ()))
    ]
    if aggregate_records:
        episodes_path = layout.final_eval_episodes_jsonl() if layout is not None else output_dir / "episodes.jsonl"
        write_episodes_jsonl(episodes_path, aggregate_records)
    if layout is not None:
        _write_run_level_diagnostics(layout=layout, policy_ids=payload["policy_ids"], matchup_results=matchup_results)
        _write_artifact_hashes(layout=layout)


def _write_matrix_csv(path: Path, matrix_payload: Mapping[str, Any]) -> None:
    policy_ids = cast(list[str], matrix_payload["policy_ids"])
    values = cast(list[list[Any]], matrix_payload["values"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id, row in zip(policy_ids, values, strict=True):
            writer.writerow([focal_policy_id, *row])


def _legacy_payoff_matrix_name(field: str) -> str | None:
    if field == "mean":
        return "p_mean"
    return None


def _maybe_layout(output_dir: Path) -> ArtifactLayout | None:
    try:
        return ArtifactLayout.from_final_eval_dir(output_dir)
    except ValueError:
        return None


def _write_run_level_diagnostics(
    *,
    layout: ArtifactLayout,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    aggregate_records = [
        record for result in matchup_results for record in cast(Sequence[EvalGameRecord], result.get("records", ()))
    ]
    if aggregate_records:
        layout.seat_bias_json().write_text(
            json.dumps(
                _build_run_level_seat_bias_payload(matchup_results=matchup_results),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_truncation_heatmap_csv(layout=layout, policy_ids=policy_ids, matchup_results=matchup_results)
    replay_verification_payload = _write_replay_diagnostics(layout=layout, matchup_results=matchup_results)
    layout.replay_verification_json().write_text(
        json.dumps(replay_verification_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_truncation_heatmap_csv(
    *,
    layout: ArtifactLayout,
    policy_ids: Sequence[str],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    result_by_pair = {
        (str(result["focal_policy_id"]), str(result["opponent_policy_id"])): result for result in matchup_results
    }
    layout.truncation_heatmap_csv().parent.mkdir(parents=True, exist_ok=True)
    with layout.truncation_heatmap_csv().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id in policy_ids:
            row: list[Any] = [focal_policy_id]
            for opponent_policy_id in policy_ids:
                key = (focal_policy_id, opponent_policy_id)
                mirror_key = (opponent_policy_id, focal_policy_id)
                result = result_by_pair.get(key) or result_by_pair.get(mirror_key)
                truncations = 0
                games = 0
                if result is not None:
                    summary = cast(Mapping[str, Any], result["summary"]).get("summary", {})
                    truncations = int(cast(Mapping[str, Any], summary).get("truncations", 0))
                    games = int(cast(Mapping[str, Any], summary).get("games", 0))
                row.append((truncations / games) if games else 0.0)
            writer.writerow(row)


def _build_run_level_seat_bias_payload(*, matchup_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    matchup_rows: list[dict[str, Any]] = []
    decisive_games_total = 0
    seat0_wins_total = 0
    for result in matchup_results:
        diagnostics = cast(Mapping[str, Any], result["diagnostics"])
        seat_results = cast(Mapping[str, Any], diagnostics.get("seat_results", {}))
        seat0_wins = int(seat_results.get("seat0_wins", 0))
        seat1_wins = int(seat_results.get("seat1_wins", 0))
        decisive_games = int(seat_results.get("decisive_games", seat0_wins + seat1_wins))
        if decisive_games <= 0:
            continue
        seat0_rate = seat0_wins / decisive_games
        seat1_rate = seat1_wins / decisive_games
        matchup_rows.append(
            {
                "policy_a": str(result["focal_policy_id"]),
                "policy_b": str(result["opponent_policy_id"]),
                "seat0_win_rate": seat0_rate,
                "seat1_win_rate": seat1_rate,
                "decisive_games": decisive_games,
            }
        )
        decisive_games_total += decisive_games
        seat0_wins_total += seat0_wins

    global_seat0_rate = (seat0_wins_total / decisive_games_total) if decisive_games_total else 0.5
    return {
        "kind": "seat_bias_summary_v1",
        "global": {
            "seat0_win_rate": global_seat0_rate,
            "ci_low": global_seat0_rate,
            "ci_high": global_seat0_rate,
            "decisive_games": decisive_games_total,
        },
        "matchups": matchup_rows,
    }


def _write_replay_diagnostics(
    *,
    layout: ArtifactLayout,
    matchup_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    replay_samples = [
        sample
        for result in matchup_results
        for sample in cast(Sequence[ReplaySampleResult], result.get("replay_samples", ()))
    ]
    replay_index_payload = {
        "kind": "replay_index_v1",
        "samples": [
            {
                "pair_index": int(sample.pair_index),
                "swap_index": int(sample.swap_index),
                "episode_index": int(sample.episode_index),
                "focal_policy_id": str(sample.focal_policy_id),
                "opponent_policy_id": str(sample.opponent_policy_id),
                "raw_replay_path": sample.raw_replay_path,
                "bundle_path": str(sample.bundle_path),
                "verification_report_path": str(sample.verification_report_path),
                "verification_status": str(sample.verification_status),
                "replay_key64": str(sample.replay_key64),
                "matched": bool(sample.matched),
                "error": sample.error,
            }
            for sample in replay_samples
        ],
    }
    layout.replay_index_json().write_text(
        json.dumps(replay_index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not replay_samples:
        return {
            "kind": "replay_verification_summary_v1",
            "status": "not_sampled",
            "sampled_episode_count": 0,
            "verified_episode_count": 0,
            "failed_episode_count": 0,
            "verified_report_count": 0,
            "message": "final_eval completed without sampled replay captures",
            "index_path": layout.relative(layout.replay_index_json()),
        }

    verified_episode_count = sum(
        1 for sample in replay_samples if sample.matched and str(sample.verification_status) == "success"
    )
    failed_samples = [
        sample for sample in replay_samples if not sample.matched or str(sample.verification_status) != "success"
    ]
    return {
        "kind": "replay_verification_summary_v1",
        "status": "ok" if not failed_samples else "failed",
        "sampled_episode_count": len(replay_samples),
        "verified_episode_count": verified_episode_count,
        "failed_episode_count": len(failed_samples),
        "verified_report_count": len(replay_samples),
        "index_path": layout.relative(layout.replay_index_json()),
        "failed_replays": [
            {
                "replay_key64": str(sample.replay_key64),
                "verification_status": str(sample.verification_status),
                "verification_report_path": str(sample.verification_report_path),
                "error": sample.error,
            }
            for sample in failed_samples
        ],
    }


def _write_artifact_hashes(*, layout: ArtifactLayout) -> None:
    tracked_paths = [
        layout.final_eval_summary_json(),
        layout.final_eval_policy_set_json(),
        layout.final_eval_metadata_json(),
        layout.final_eval_matchups_csv(),
        layout.final_eval_posterior_samples_json(),
        layout.final_eval_posterior_samples_npz(),
        layout.final_eval_matrix_csv("mean"),
        layout.seat_bias_json(),
        layout.truncation_heatmap_csv(),
        layout.replay_verification_json(),
        layout.replay_index_json(),
    ]
    payload = {
        "kind": "final_eval_artifact_hashes_v1",
        "artifacts": {layout.relative(path): _sha256_file(path) for path in tracked_paths if path.is_file()},
    }
    layout.final_eval_aggregate_hashes_json().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matchup_dir_name(*, focal_index: int, opponent_index: int, focal_policy_id: str, opponent_policy_id: str) -> str:
    return f"{focal_index:02d}_{_slug(focal_policy_id)}__vs__{opponent_index:02d}_{_slug(opponent_policy_id)}"


def _slug(value: str) -> str:
    parts = [
        "".join(char.lower() for char in chunk if char.isalnum())
        for chunk in str(value).replace("-", " ").replace("_", " ").split()
    ]
    slug = "_".join(part for part in parts if part)
    return slug or "policy"


def _bootstrap_seed(*, focal_policy_id: str, opponent_policy_id: str) -> int:
    return stable_hash64(
        canonical_json_bytes(
            {
                "kind": "final_eval_bootstrap_v1",
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
            }
        )
    )


def _relative_to(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
