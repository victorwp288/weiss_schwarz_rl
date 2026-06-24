from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import require_fixed_python_hash_seed
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.diagnostics.b2_audit.b2_audit_aggregation import (
    aggregate_audit_summary as _aggregate_audit_summary,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    aggregate_trajectory_summary as _aggregate_trajectory_summary,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    annotate_step_diff as _annotate_step_diff,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    b2_audit_plan_payload,
    output_run_dir_path,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    build_bundle_summary as _build_bundle_summary,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    format_decoded_action as _format_decoded_action,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    inspection_policy_id as _inspection_policy_id,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    materialize_audit_bundle_copy as _materialize_audit_bundle_copy,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_reports import (
    top_counter_items as _top_counter_items,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    MatchupSource,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    audit_run_id256 as _audit_run_id256,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    candidate_policy_aliases as _candidate_policy_aliases,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    load_matchup_source as _load_matchup_source,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    manifest_config_hash as _manifest_config_hash,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    resolve_requested_policy_id as _resolve_requested_policy_id,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    resolve_source_config_hash as _resolve_source_config_hash,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_source import (
    run_config_hashes as _run_config_hashes,
)
from weiss_rl.eval.analysis.export import write_matchup_summary_json
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.eval.simulator.harness import (
    EvalGameRecord,
    build_seat_swapped_schedule,
    record_completed_game,
    write_episodes_jsonl,
)
from weiss_rl.eval.simulator.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.replay.inspector import inspect_replay_bundle
from weiss_rl.replay.inspector_report import write_replay_inspection_report

_DEFAULT_TOP_K = 25
_DEFAULT_TOP_ACTIONS = 5
_AUDIT_DIRNAME = "audit"
_INSPECTIONS_DIRNAME = "replay_inspections"
_BUNDLE_COPIES_DIRNAME = "replay_bundles"

__all__ = [
    "MatchupSource",
    "_aggregate_trajectory_summary",
    "_annotate_step_diff",
    "_audit_run_id256",
    "_build_bundle_summary",
    "_build_parser",
    "_candidate_policy_aliases",
    "_format_decoded_action",
    "_inspection_policy_id",
    "_load_matchup_source",
    "_manifest_config_hash",
    "_materialize_audit_bundle_copy",
    "_resolve_requested_policy_id",
    "_resolve_source_config_hash",
    "_run_config_hashes",
    "_top_counter_items",
    "b2_audit_plan_payload",
    "main",
    "output_run_dir_path",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a seed-targeted learner-vs-heuristic replay audit over an existing episodes.jsonl file"
    )
    parser.add_argument("--stack-config", type=Path, required=True, help="Stack config for the evaluation contract")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Source run dir used to resolve the focal policy snapshot and simulator observation spec",
    )
    parser.add_argument(
        "--output-run-dir",
        type=Path,
        required=True,
        help="Run dir where the rerun matchup, replay bundles, and audit summary will be written",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        required=True,
        help="Existing seat-swapped matchup episodes.jsonl to extract paired seeds from",
    )
    parser.add_argument(
        "--policy-id",
        required=True,
        help="Focal learner policy id to rerun and inspect against the heuristic opponent",
    )
    parser.add_argument(
        "--allow-policy-id-mismatch",
        action="store_true",
        help=(
            "Allow --policy-id to differ from the source episodes.jsonl focal_policy_id. "
            "Use this only to reuse the source paired seeds for a counterfactual rerun; "
            "the output summary records both policy ids."
        ),
    )
    parser.add_argument(
        "--opponent-policy-id",
        default=HEURISTIC_PUBLIC_POLICY_ID,
        help="Heuristic opponent policy id to rerun against; must match the source episodes.jsonl opponent",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON for resolving the focal policy id",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional locked B1 NoLeague run dir. Required when --opponent-policy-id is "
            "'B1 NoLeague baseline' and the source registry does not contain the B1 alias."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for the aggregated audit summary JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=_DEFAULT_TOP_K,
        help="Number of top replay steps to persist from each inspection report",
    )
    parser.add_argument(
        "--top-actions",
        type=int,
        default=_DEFAULT_TOP_ACTIONS,
        help="Number of highest-delta legal actions to include per replay step",
    )
    parser.add_argument(
        "--accept-snapshot-config-hash",
        action="append",
        default=[],
        help=(
            "Extra snapshot config_hash256 accepted by replay inspection. "
            "Repeat for imported/warm-started snapshots whose weights intentionally come from another stack."
        ),
    )
    parser.add_argument(
        "--require-opponent-context-index",
        action="store_true",
        help=(
            "Pass --opponent-policy-id into replay inspection as opponent context and fail if the focal model "
            "does not map it to a nonzero context index."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if args.top_actions <= 0:
        parser.error("--top-actions must be >= 1")
    opponent_policy_id = str(args.opponent_policy_id).strip()
    if not opponent_policy_id:
        parser.error("--opponent-policy-id must not be empty")
    try:
        require_fixed_python_hash_seed("heuristic disagreement audit")
    except RuntimeError as err:
        parser.error(str(err))

    stack = load_stack_config(args.stack_config)
    source = _load_matchup_source(args.episodes_jsonl)
    stack_config_hash256 = compute_config_hash256(stack)
    try:
        config_hash256, run_manifest_config_hash256 = _resolve_source_config_hash(
            source_config_hash256=source.config_hash256,
            stack_config_hash256=stack_config_hash256,
            run_dir=args.run_dir,
        )
    except ValueError as err:
        parser.error(str(err))
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=source.spec_hash256)
    resolved_policy_id = _resolve_requested_policy_id(
        requested_policy_id=str(args.policy_id),
        source_focal_policy_id=source.focal_policy_id,
        allow_mismatch=bool(args.allow_policy_id_mismatch),
    )
    if resolved_policy_id is None:
        parser.error(
            "--policy-id must match or alias the focal_policy_id in the source episodes.jsonl "
            f"(source={source.focal_policy_id!r}, requested={args.policy_id!r}); "
            "pass --allow-policy-id-mismatch only when reusing the source paired seeds for a counterfactual rerun"
        )
    if source.opponent_policy_id != opponent_policy_id:
        parser.error(
            "--episodes-jsonl must describe a matchup against --opponent-policy-id "
            f"(expected {opponent_policy_id!r}, found opponent_policy_id={source.opponent_policy_id!r})"
        )

    output_layout = ArtifactLayout.from_run_dir(args.output_run_dir)
    output_layout.ensure_directories()
    audit_dir = args.output_run_dir / _AUDIT_DIRNAME
    inspections_dir = audit_dir / _INSPECTIONS_DIRNAME
    bundle_copies_dir = audit_dir / _BUNDLE_COPIES_DIRNAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    inspections_dir.mkdir(parents=True, exist_ok=True)
    bundle_copies_dir.mkdir(parents=True, exist_ok=True)

    source_registry_path = args.snapshot_registry_json or (args.run_dir / "training" / "snapshots" / "registry.json")
    if not source_registry_path.is_file():
        parser.error(f"snapshot registry not found: {source_registry_path}")
    accepted_snapshot_config_hashes = list(args.accept_snapshot_config_hash)
    accepted_snapshot_config_hashes.extend(_run_config_hashes(args.b1_baseline_run_dir))

    resolved_policies = resolve_eval_policies(
        stack=stack,
        policy_ids=[resolved_policy_id, opponent_policy_id],
        run_dir=args.run_dir,
        observation_dim=int(contract.spec_bundle["observation"]["obs_len"]),
        action_dim=int(contract.spec_bundle["action"]["action_space_size"]),
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=source_registry_path,
        b1_baseline_run_dir=args.b1_baseline_run_dir,
    )
    evaluation = stack.config.evaluation
    if evaluation is None:
        parser.error("stack config is missing evaluation settings")
    runner = SimulatorEvalRunner(
        stack=stack,
        policies=resolved_policies,
        artifact_layout=output_layout,
        run_id256=_audit_run_id256(
            policy_id=resolved_policy_id,
            opponent_policy_id=opponent_policy_id,
            episodes_jsonl=args.episodes_jsonl,
            output_run_dir=args.output_run_dir,
            paired_seeds=source.paired_seeds,
        ),
        spec_hash256=contract.spec_hash256,
        action_dim=int(contract.spec_bundle["action"]["action_space_size"]),
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=1.0,
        regression_capture_count=max(1, len(source.paired_seeds) * 2),
    )

    schedule = build_seat_swapped_schedule(
        focal_policy_id=resolved_policy_id,
        opponent_policy_id=opponent_policy_id,
        paired_seeds=source.paired_seeds,
    )
    records: list[EvalGameRecord] = []
    bundle_summaries: list[dict[str, Any]] = []
    inspection_errors: list[dict[str, Any]] = []
    decoder = ActionCatalog.from_spec_bundle(contract.spec_bundle)

    for scheduled_game in schedule:
        result = runner.run_game(scheduled_game)
        if result.replay_sample is None:
            raise RuntimeError(
                "replay capture did not produce a bundle; ensure the simulator environment supports replay sampling"
            )
        record = record_completed_game(
            scheduled_game=scheduled_game,
            result=result,
            run_id256=runner.run_id256_bytes,
            config_hash256=config_hash256,
            spec_hash256=contract.spec_hash256,
        )
        records.append(record)

        bundle_path = _materialize_audit_bundle_copy(
            source_bundle_path=output_run_dir_path(args.output_run_dir, result.replay_sample.bundle_path),
            bundle_copies_dir=bundle_copies_dir,
            pair_index=int(scheduled_game.pair_index),
            swap_index=int(scheduled_game.swap_index),
        )
        report_path = inspections_dir / f"{bundle_path.stem}.json"
        try:
            report = inspect_replay_bundle(
                bundle_path=bundle_path,
                stack=stack,
                policy_a=resolved_policy_id,
                policy_b=_inspection_policy_id(opponent_policy_id),
                run_dir=args.run_dir,
                snapshot_registry_path=source_registry_path,
                top_k=args.top_k,
                top_actions=args.top_actions,
                accepted_snapshot_config_hashes=accepted_snapshot_config_hashes,
                opponent_context_policy_id=opponent_policy_id,
                require_opponent_context_index=bool(args.require_opponent_context_index),
            )
        except Exception as exc:
            inspection_errors.append(
                {
                    "bundle_path": bundle_path.as_posix(),
                    "pair_index": int(scheduled_game.pair_index),
                    "swap_index": int(scheduled_game.swap_index),
                    "episode_seed": int(scheduled_game.episode_seed),
                    "error": str(exc),
                }
            )
            continue

        write_replay_inspection_report(report_path, report)
        bundle_summaries.append(
            _build_bundle_summary(
                report=report,
                report_path=report_path,
                bundle_path=bundle_path,
                scheduled_game=scheduled_game,
                decoder=decoder,
            )
        )

    episodes_path = audit_dir / "episodes.jsonl"
    write_episodes_jsonl(episodes_path, records)

    summary = _aggregate_audit_summary(
        source=source,
        policy_id=resolved_policy_id,
        opponent_policy_id=opponent_policy_id,
        episodes_jsonl=args.episodes_jsonl,
        run_dir=args.run_dir,
        output_run_dir=args.output_run_dir,
        episodes_path=episodes_path,
        game_count=len(schedule),
        bundle_summaries=bundle_summaries,
        inspection_errors=inspection_errors,
        stack_config_hash256=stack_config_hash256,
        run_manifest_config_hash256=run_manifest_config_hash256,
        policy_id_mismatch_allowed=bool(args.allow_policy_id_mismatch),
        requested_policy_id=str(args.policy_id),
    )
    summary_path = args.summary_json or (audit_dir / "summary.json")
    write_matchup_summary_json(summary_path, summary)

    print(
        f"Reran {len(schedule)} games across {len(source.paired_seeds)} paired seeds and captured {len(bundle_summaries)} bundles "
        f"into {args.output_run_dir}; audit summary written to {summary_path}"
    )
    if inspection_errors:
        print(
            f"Warning: {len(inspection_errors)} replay inspections failed; see {summary_path}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
