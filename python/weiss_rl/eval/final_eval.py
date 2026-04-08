"""Final-eval orchestration for the deterministic final policy set."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from weiss_rl.config.models import FinalPolicySetSelectionConfig, StopRulesConfig
from weiss_rl.eval.diagnostics import build_seat_advantage_diagnostics, write_matchup_diagnostics_json
from weiss_rl.eval.export import build_matchup_export, write_matchup_summary_csv, write_matchup_summary_json
from weiss_rl.eval.harness import (
    EvalGameRecord,
    EvalGameRunner,
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
        for opponent_index, opponent_policy_id in enumerate(resolved_policy_ids):
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
    return resolved, {
        "mode": "deterministic_v1",
        "policy_count": len(resolved),
        "snapshot_registry_path": snapshot_registry_path.as_posix(),
        "dev_eval_summaries_path": dev_eval_summaries_path.as_posix(),
        "final_policy_set_size": int(final_policy_set_size),
    }


def _validate_seed_budget(*, paired_seeds: Sequence[int], stage1_paired_seeds: int, max_paired_seeds: int) -> None:
    if stage1_paired_seeds < 1:
        raise ValueError("stage1_paired_seeds must be positive")
    if max_paired_seeds < stage1_paired_seeds:
        raise ValueError("max_paired_seeds must be >= stage1_paired_seeds")
    if len(paired_seeds) < max_paired_seeds:
        raise ValueError(
            f"final eval requires at least {max_paired_seeds} paired seeds, found {len(paired_seeds)}"
        )


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
    matchup_dir = output_dir / "matchups" / _matchup_dir_name(
        focal_index=focal_index,
        opponent_index=opponent_index,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
    )
    records: list[EvalGameRecord] = []
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
                "sample_count": sample_count,
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
    results_by_key = {
        (str(result["focal_policy_id"]), str(result["opponent_policy_id"])): result for result in matchup_results
    }
    matrices = {
        field: _build_matrix(
            policy_ids=policy_ids,
            results_by_key=results_by_key,
            field=field,
        )
        for field in _MATRIX_FIELDS
    }
    posterior_matrix = [
        [
            list(results_by_key[(focal_policy_id, opponent_policy_id)]["posterior_samples"])
            for opponent_policy_id in policy_ids
        ]
        for focal_policy_id in policy_ids
    ]
    top_level_metadata = dict(metadata or {})
    top_level_metadata.update(
        {
            "policy_count": len(policy_ids),
            "matchup_count": len(matchup_results),
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
                "paired_seed_count": result["summary"]["paired_seeds"],
                "stop_reason": result["summary"]["stop_reason"],
            }
            for result in matchup_results
        ],
    }


def _build_matrix(
    *,
    policy_ids: Sequence[str],
    results_by_key: Mapping[tuple[str, str], dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = [
        [
            _matrix_value(results_by_key[(focal_policy_id, opponent_policy_id)]["summary"], field=field)
            for opponent_policy_id in policy_ids
        ]
        for focal_policy_id in policy_ids
    ]
    return {
        "policy_ids": list(policy_ids),
        "values": values,
    }


def _matrix_value(payload: Mapping[str, Any], *, field: str) -> Any:
    summary = cast(Mapping[str, Any], payload["summary"])
    uncertainty = cast(Mapping[str, Any], payload["uncertainty"])
    if field in uncertainty:
        return uncertainty[field]
    if field in summary:
        return summary[field]
    if field == "paired_seed_count":
        return uncertainty["paired_seed_count"]
    return payload[field]


def _write_final_eval_artifacts(
    *,
    output_dir: Path,
    payload: Mapping[str, Any],
    matchup_results: Sequence[dict[str, Any]],
) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "policy_set.json").write_text(
        json.dumps({"policy_ids": payload["policy_ids"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "posterior_samples.json").write_text(
        json.dumps(payload["posterior_samples"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matrices_dir = output_dir / "matrices"
    matrices = cast(Mapping[str, Mapping[str, Any]], payload["matrices"])
    for field, matrix_payload in matrices.items():
        json_path = matrices_dir / f"{field}.json"
        csv_path = matrices_dir / f"{field}.csv"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_matrix_csv(csv_path, matrix_payload)

    manifest_rows = [
        {
            "focal_policy_id": result["focal_policy_id"],
            "opponent_policy_id": result["opponent_policy_id"],
            "matchup_dir": _relative_to(Path(result["matchup_dir"]), root=output_dir),
            "paired_seed_count": result["summary"]["paired_seeds"],
            "stop_reason": result["summary"]["stop_reason"],
        }
        for result in matchup_results
    ]
    manifest_path = output_dir / "matchups.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def _write_matrix_csv(path: Path, matrix_payload: Mapping[str, Any]) -> None:
    policy_ids = cast(list[str], matrix_payload["policy_ids"])
    values = cast(list[list[Any]], matrix_payload["values"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["focal_policy_id", *policy_ids])
        for focal_policy_id, row in zip(policy_ids, values, strict=True):
            writer.writerow([focal_policy_id, *row])


def _matchup_dir_name(*, focal_index: int, opponent_index: int, focal_policy_id: str, opponent_policy_id: str) -> str:
    return (
        f"{focal_index:02d}_{_slug(focal_policy_id)}__vs__"
        f"{opponent_index:02d}_{_slug(opponent_policy_id)}"
    )


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
