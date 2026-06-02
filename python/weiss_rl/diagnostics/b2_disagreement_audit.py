from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import canonical_json_bytes, require_fixed_python_hash_seed, sha256_hex
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.diagnostics.b2_audit_aggregation import (
    aggregate_audit_summary as _aggregate_audit_summary,
)
from weiss_rl.diagnostics.b2_audit_aggregation import (
    aggregate_trajectory_summary as _shared_aggregate_trajectory_summary,
)
from weiss_rl.diagnostics.b2_audit_aggregation import (
    counter_payload as _counter_payload,
)
from weiss_rl.diagnostics.b2_audit_aggregation import (
    top_counter_items as _shared_top_counter_items,
)
from weiss_rl.eval.export import load_eval_game_records, write_matchup_summary_json
from weiss_rl.eval.harness import (
    EvalGameRecord,
    build_seat_swapped_schedule,
    record_completed_game,
    write_episodes_jsonl,
)
from weiss_rl.eval.heuristic_public import ActionCatalog, DecodedAction
from weiss_rl.eval.policies.set import HEURISTIC_PUBLIC_POLICY_ID, LEGACY_NO_LEAGUE_POLICY_ID, NO_LEAGUE_POLICY_ID
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.replay.inspector import inspect_replay_bundle, write_replay_inspection_report

_DEFAULT_TOP_K = 25
_DEFAULT_TOP_ACTIONS = 5
_AUDIT_DIRNAME = "audit"
_INSPECTIONS_DIRNAME = "replay_inspections"
_BUNDLE_COPIES_DIRNAME = "replay_bundles"


@dataclass(frozen=True, slots=True)
class MatchupSource:
    focal_policy_id: str
    opponent_policy_id: str
    config_hash256: str
    spec_hash256: str
    paired_seeds: tuple[int, ...]


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


def _load_matchup_source(path: Path) -> MatchupSource:
    records = load_eval_game_records(path)
    focal_policy_ids = {record.focal_policy_id for record in records}
    opponent_policy_ids = {record.opponent_policy_id for record in records}
    config_hashes = {record.config_hash256 for record in records}
    spec_hashes = {record.spec_hash256 for record in records}
    if len(focal_policy_ids) != 1 or len(opponent_policy_ids) != 1:
        raise ValueError("source episodes.jsonl must contain exactly one focal/opponent matchup")
    if len(config_hashes) != 1 or len(spec_hashes) != 1:
        raise ValueError("source episodes.jsonl must contain exactly one config/spec contract")

    pair_to_seed: dict[int, int] = {}
    pair_order: list[int] = []
    pair_swaps: dict[int, set[int]] = defaultdict(set)
    for record in records:
        pair_index = int(record.pair_index)
        swap_index = int(record.swap_index)
        if swap_index not in (0, 1):
            raise ValueError(f"source episodes.jsonl must be seat-swapped, got swap_index={swap_index}")
        pair_swaps[pair_index].add(swap_index)
        if pair_index not in pair_to_seed:
            pair_order.append(pair_index)
            pair_to_seed[pair_index] = int(record.episode_seed)
        elif pair_to_seed[pair_index] != int(record.episode_seed):
            raise ValueError(f"pair_index {pair_index} uses inconsistent episode_seed values")

    for pair_index, swaps in pair_swaps.items():
        if swaps != {0, 1}:
            raise ValueError(f"pair_index {pair_index} must contain swap_index 0 and 1")

    paired_seeds: list[int] = []
    seen_seeds: set[int] = set()
    for pair_index in pair_order:
        seed = pair_to_seed[pair_index]
        if seed in seen_seeds:
            continue
        seen_seeds.add(seed)
        paired_seeds.append(seed)

    if not paired_seeds:
        raise ValueError(f"no paired seeds found in {path}")

    return MatchupSource(
        focal_policy_id=next(iter(focal_policy_ids)),
        opponent_policy_id=next(iter(opponent_policy_ids)),
        config_hash256=next(iter(config_hashes)),
        spec_hash256=next(iter(spec_hashes)),
        paired_seeds=tuple(paired_seeds),
    )


def _resolve_source_config_hash(
    *,
    source_config_hash256: str,
    stack_config_hash256: str,
    run_dir: Path,
) -> tuple[str, str | None]:
    run_manifest_config_hash256 = _manifest_config_hash(run_dir)
    accepted_hashes = {stack_config_hash256}
    if run_manifest_config_hash256:
        accepted_hashes.add(run_manifest_config_hash256)
    if source_config_hash256 not in accepted_hashes:
        message = (
            "stack config hash does not match the source episodes.jsonl "
            f"(expected {source_config_hash256!r}, got {stack_config_hash256!r}"
        )
        if run_manifest_config_hash256:
            message += f", run manifest {run_manifest_config_hash256!r}"
        message += ")"
        raise ValueError(message)
    return source_config_hash256, run_manifest_config_hash256


def _manifest_config_hash(run_dir: Path) -> str | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    config_hash256 = payload.get("config_hash256")
    if isinstance(config_hash256, str) and config_hash256.strip():
        return config_hash256.strip()
    return None


def _resolve_requested_policy_id(
    *,
    requested_policy_id: str,
    source_focal_policy_id: str,
    allow_mismatch: bool = False,
) -> str | None:
    normalized_requested = requested_policy_id.strip()
    if not normalized_requested:
        return None
    if allow_mismatch:
        return normalized_requested
    if normalized_requested == source_focal_policy_id:
        return normalized_requested
    for candidate in _candidate_policy_aliases(source_focal_policy_id):
        if normalized_requested == candidate:
            return normalized_requested
    return None


def _candidate_policy_aliases(source_focal_policy_id: str) -> tuple[str, ...]:
    normalized = source_focal_policy_id.strip()
    aliases: list[str] = [normalized]
    if "_p" in normalized:
        suffix = normalized.rsplit("_p", 1)[-1]
        if suffix.isdigit():
            aliases.append(f"policy_{int(suffix):06d}")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _build_bundle_summary(
    *,
    report: dict[str, Any],
    report_path: Path,
    bundle_path: Path,
    scheduled_game: Any,
    decoder: ActionCatalog,
) -> dict[str, Any]:
    annotated_diffs = [_annotate_step_diff(diff, decoder=decoder) for diff in report.get("top_differences", [])]
    family_pair_counts = Counter(
        (diff["policy_a_top_action_family"], diff["policy_b_top_action_family"]) for diff in annotated_diffs
    )
    policy_a_family_counts = Counter(diff["policy_a_top_action_family"] for diff in annotated_diffs)
    policy_b_family_counts = Counter(diff["policy_b_top_action_family"] for diff in annotated_diffs)
    recorded_family_counts = Counter(diff["recorded_action_family"] for diff in annotated_diffs)
    action_label_pair_counts = Counter(
        (diff["policy_a_top_action_label"], diff["policy_b_top_action_label"]) for diff in annotated_diffs
    )
    policy_a_action_label_counts = Counter(diff["policy_a_top_action_label"] for diff in annotated_diffs)
    policy_b_action_label_counts = Counter(diff["policy_b_top_action_label"] for diff in annotated_diffs)

    return {
        "bundle_path": bundle_path.as_posix(),
        "report_path": report_path.as_posix(),
        "pair_index": int(scheduled_game.pair_index),
        "swap_index": int(scheduled_game.swap_index),
        "focal_seat": int(scheduled_game.focal_seat),
        "seat0_policy_id": str(scheduled_game.seat0_policy_id),
        "seat1_policy_id": str(scheduled_game.seat1_policy_id),
        "episode_seed": int(scheduled_game.episode_seed),
        "replay_key64": str(report["replay"]["replay_key64"]),
        "summary": dict(report["summary"]),
        "trajectory_summary": dict(report.get("trajectory_summary", {})),
        "compared_steps": int(report["compared_steps"]),
        "inspected_step_count": len(annotated_diffs),
        "family_pair_counts": _counter_payload(
            family_pair_counts,
            key_names=("policy_a_family", "policy_b_family"),
        ),
        "policy_a_family_counts": _counter_payload(policy_a_family_counts, key_names=("family",)),
        "policy_b_family_counts": _counter_payload(policy_b_family_counts, key_names=("family",)),
        "recorded_family_counts": _counter_payload(recorded_family_counts, key_names=("family",)),
        "action_label_pair_counts": _counter_payload(
            action_label_pair_counts,
            key_names=("policy_a_action_label", "policy_b_action_label"),
        ),
        "policy_a_action_label_counts": _counter_payload(policy_a_action_label_counts, key_names=("action_label",)),
        "policy_b_action_label_counts": _counter_payload(policy_b_action_label_counts, key_names=("action_label",)),
        "top_examples": annotated_diffs[:3],
    }


def _annotate_step_diff(diff: dict[str, Any], *, decoder: ActionCatalog) -> dict[str, Any]:
    recorded_action = decoder.decode(int(diff["recorded_action"]))
    policy_a_top_action = decoder.decode(int(diff["policy_a_top_action"]["action"]))
    policy_b_top_action = decoder.decode(int(diff["policy_b_top_action"]["action"]))
    annotated = dict(diff)
    annotated["recorded_action_family"] = recorded_action.family
    annotated["recorded_action_label"] = _format_decoded_action(recorded_action)
    annotated["policy_a_top_action_family"] = policy_a_top_action.family
    annotated["policy_a_top_action_label"] = _format_decoded_action(policy_a_top_action)
    annotated["policy_b_top_action_family"] = policy_b_top_action.family
    annotated["policy_b_top_action_label"] = _format_decoded_action(policy_b_top_action)
    return annotated


def _aggregate_trajectory_summary(bundle_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _shared_aggregate_trajectory_summary(bundle_summaries)


def _top_counter_items(counter: Counter[Any], *, key_names: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    return _shared_top_counter_items(counter, key_names=key_names, limit=limit)


def _inspection_policy_id(policy_id: str) -> str:
    normalized = str(policy_id).strip()
    if normalized == NO_LEAGUE_POLICY_ID:
        return LEGACY_NO_LEAGUE_POLICY_ID
    return normalized


def _run_config_hashes(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    hashes: list[str] = []
    hash_path = Path(run_dir) / "config_hash256.txt"
    if hash_path.is_file():
        value = hash_path.read_text(encoding="utf-8").strip()
        if value:
            hashes.append(value)
    manifest_path = Path(run_dir) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = str(manifest.get("config_hash256", "")).strip() if isinstance(manifest, dict) else ""
        if value:
            hashes.append(value)
    return list(dict.fromkeys(hashes))


def _format_decoded_action(action: DecodedAction) -> str:
    if action.family in {"pass", "mulligan_confirm", "choice_prev_page", "choice_next_page", "concede"}:
        return action.family
    if action.family in {"clock_from_hand", "main_play_event", "climax_play"}:
        return _format_with_fields(action.family, ("hand_index", action.hand_index))
    if action.family in {"level_up", "trigger_order", "choice_select", "mulligan_select"}:
        return _format_with_fields(action.family, ("index", action.index), ("hand_index", action.hand_index))
    if action.family == "main_play_character":
        return _format_with_fields(
            action.family,
            ("hand_index", action.hand_index),
            ("stage_slot", action.stage_slot),
        )
    if action.family == "main_move":
        return _format_with_fields(action.family, ("from_slot", action.from_slot), ("to_slot", action.to_slot))
    if action.family == "attack":
        return _format_with_fields(action.family, ("slot", action.slot), ("attack_type", action.attack_type))
    if action.family in {"encore_pay", "encore_decline"}:
        return _format_with_fields(action.family, ("slot", action.slot))
    return action.family


def _format_with_fields(family: str, *fields: tuple[str, Any]) -> str:
    payload = ", ".join(f"{name}={value}" for name, value in fields if value is not None)
    return family if not payload else f"{family}({payload})"


def _audit_run_id256(
    *,
    policy_id: str,
    opponent_policy_id: str,
    episodes_jsonl: Path,
    output_run_dir: Path,
    paired_seeds: Sequence[int],
) -> str:
    payload = canonical_json_bytes(
        {
            "kind": "b2_disagreement_audit_v1",
            "policy_id": policy_id,
            "opponent_policy_id": opponent_policy_id,
            "episodes_jsonl": episodes_jsonl.resolve().as_posix(),
            "output_run_dir": output_run_dir.resolve().as_posix(),
            "paired_seeds": list(int(seed) for seed in paired_seeds),
        }
    )
    return sha256_hex(payload)


def output_run_dir_path(output_run_dir: Path, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else output_run_dir / path


def _materialize_audit_bundle_copy(
    *,
    source_bundle_path: Path,
    bundle_copies_dir: Path,
    pair_index: int,
    swap_index: int,
) -> Path:
    if not source_bundle_path.is_file():
        raise FileNotFoundError(f"replay bundle not found: {source_bundle_path}")
    copied_path = bundle_copies_dir / (
        f"{source_bundle_path.stem}_pair{pair_index:03d}_swap{swap_index}{source_bundle_path.suffix}"
    )
    shutil.copy2(source_bundle_path, copied_path)
    return copied_path


if __name__ == "__main__":
    raise SystemExit(main())
