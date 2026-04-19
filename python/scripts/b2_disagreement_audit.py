from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.eval.export import load_eval_game_records, write_matchup_summary_json
from weiss_rl.eval.heuristic_public import ActionCatalog, DecodedAction
from weiss_rl.eval.harness import (
    EvalGameRecord,
    build_seat_swapped_schedule,
    record_completed_game,
    write_episodes_jsonl,
)
from weiss_rl.eval.policy_set import HEURISTIC_PUBLIC_POLICY_ID
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.replay.inspector import inspect_replay_bundle, write_replay_inspection_report
from weiss_rl.repro import canonical_json_bytes, sha256_hex
from weiss_rl.simulator_contract import load_verified_simulator_contract

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
        description="Run a seed-targeted learner-vs-B2 replay audit over an existing episodes.jsonl file"
    )
    parser.add_argument("--stack-config", type=Path, required=True, help="Stack config for the evaluation contract")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Source run dir used to resolve the focal policy snapshot and B2 observation spec",
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
        help="Existing seat-swapped B2 matchup episodes.jsonl to extract paired seeds from",
    )
    parser.add_argument(
        "--policy-id",
        required=True,
        help="Focal learner policy id to rerun and inspect against B2 HeuristicPublic",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON for resolving the focal policy id",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")
    if args.top_actions <= 0:
        parser.error("--top-actions must be >= 1")

    stack = load_stack_config(args.stack_config)
    source = _load_matchup_source(args.episodes_jsonl)
    config_hash256 = compute_config_hash256(stack)
    if config_hash256 != source.config_hash256:
        parser.error(
            "stack config hash does not match the source episodes.jsonl "
            f"(expected {source.config_hash256!r}, got {config_hash256!r})"
        )
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=source.spec_hash256)
    resolved_policy_id = _resolve_requested_policy_id(
        requested_policy_id=str(args.policy_id),
        source_focal_policy_id=source.focal_policy_id,
    )
    if resolved_policy_id is None:
        parser.error(
            "--policy-id must match or alias the focal_policy_id in the source episodes.jsonl "
            f"(source={source.focal_policy_id!r}, requested={args.policy_id!r})"
        )
    if source.opponent_policy_id != HEURISTIC_PUBLIC_POLICY_ID:
        parser.error(
            "--episodes-jsonl must describe a matchup against B2 HeuristicPublic "
            f"(found opponent_policy_id={source.opponent_policy_id!r})"
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

    resolved_policies = resolve_eval_policies(
        stack=stack,
        policy_ids=[resolved_policy_id, HEURISTIC_PUBLIC_POLICY_ID],
        run_dir=args.run_dir,
        observation_dim=int(contract.spec_bundle["observation"]["obs_len"]),
        action_dim=int(contract.spec_bundle["action"]["action_space_size"]),
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=source_registry_path,
        b1_baseline_run_dir=None,
    )
    runner = SimulatorEvalRunner(
        stack=stack,
        policies=resolved_policies,
        artifact_layout=output_layout,
        run_id256=_audit_run_id256(
            policy_id=resolved_policy_id,
            episodes_jsonl=args.episodes_jsonl,
            output_run_dir=args.output_run_dir,
            paired_seeds=source.paired_seeds,
        ),
        spec_hash256=contract.spec_hash256,
        action_dim=int(contract.spec_bundle["action"]["action_space_size"]),
        pass_action_id=int(contract.spec_bundle["action"]["pass_action_id"]),
        require_sorted_legal_ids=bool(stack.config.evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=1.0,
        regression_capture_count=max(1, len(source.paired_seeds) * 2),
    )

    schedule = build_seat_swapped_schedule(
        focal_policy_id=resolved_policy_id,
        opponent_policy_id=HEURISTIC_PUBLIC_POLICY_ID,
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
                policy_b=HEURISTIC_PUBLIC_POLICY_ID,
                run_dir=args.run_dir,
                snapshot_registry_path=source_registry_path,
                top_k=args.top_k,
                top_actions=args.top_actions,
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
        episodes_jsonl=args.episodes_jsonl,
        run_dir=args.run_dir,
        output_run_dir=args.output_run_dir,
        episodes_path=episodes_path,
        game_count=len(schedule),
        bundle_summaries=bundle_summaries,
        inspection_errors=inspection_errors,
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


def _resolve_requested_policy_id(*, requested_policy_id: str, source_focal_policy_id: str) -> str | None:
    normalized_requested = requested_policy_id.strip()
    if not normalized_requested:
        return None
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
        "episode_seed": int(scheduled_game.episode_seed),
        "replay_key64": str(report["replay"]["replay_key64"]),
        "summary": dict(report["summary"]),
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


def _aggregate_audit_summary(
    *,
    source: MatchupSource,
    policy_id: str,
    episodes_jsonl: Path,
    run_dir: Path,
    output_run_dir: Path,
    episodes_path: Path,
    game_count: int,
    bundle_summaries: Sequence[dict[str, Any]],
    inspection_errors: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    family_pair_counts: Counter[tuple[str, str]] = Counter()
    policy_a_family_counts: Counter[str] = Counter()
    policy_b_family_counts: Counter[str] = Counter()
    recorded_family_counts: Counter[str] = Counter()
    action_label_pair_counts: Counter[tuple[str, str]] = Counter()
    policy_a_action_label_counts: Counter[str] = Counter()
    policy_b_action_label_counts: Counter[str] = Counter()
    compared_steps = 0
    inspected_steps = 0
    max_total_variation = 0.0
    weighted_total_variation = 0.0

    for bundle_summary in bundle_summaries:
        compared_steps += int(bundle_summary["compared_steps"])
        inspected_steps += int(bundle_summary["inspected_step_count"])
        summary = bundle_summary["summary"]
        max_total_variation = max(max_total_variation, float(summary.get("max_total_variation", 0.0)))
        weighted_total_variation += float(summary.get("mean_total_variation", 0.0)) * int(
            bundle_summary["compared_steps"]
        )
        for item in bundle_summary["family_pair_counts"]:
            family_pair_counts[(str(item["policy_a_family"]), str(item["policy_b_family"]))] += int(item["count"])
        for item in bundle_summary["policy_a_family_counts"]:
            policy_a_family_counts[str(item["family"])] += int(item["count"])
        for item in bundle_summary["policy_b_family_counts"]:
            policy_b_family_counts[str(item["family"])] += int(item["count"])
        for item in bundle_summary["recorded_family_counts"]:
            recorded_family_counts[str(item["family"])] += int(item["count"])
        for item in bundle_summary["action_label_pair_counts"]:
            action_label_pair_counts[(str(item["policy_a_action_label"]), str(item["policy_b_action_label"]))] += int(
                item["count"]
            )
        for item in bundle_summary["policy_a_action_label_counts"]:
            policy_a_action_label_counts[str(item["action_label"])] += int(item["count"])
        for item in bundle_summary["policy_b_action_label_counts"]:
            policy_b_action_label_counts[str(item["action_label"])] += int(item["count"])

    return {
        "status": "ok" if not inspection_errors else "partial_failure",
        "policy_id": policy_id,
        "opponent_policy_id": HEURISTIC_PUBLIC_POLICY_ID,
        "source": {
            "run_dir": run_dir.resolve().as_posix(),
            "episodes_jsonl": episodes_jsonl.resolve().as_posix(),
            "config_hash256": source.config_hash256,
            "spec_hash256": source.spec_hash256,
            "paired_seed_count": len(source.paired_seeds),
            "paired_seeds": list(source.paired_seeds),
            "focal_policy_id": source.focal_policy_id,
            "opponent_policy_id": source.opponent_policy_id,
        },
        "output_run_dir": output_run_dir.resolve().as_posix(),
        "episodes_path": episodes_path.as_posix(),
        "replayed_game_count": int(game_count),
        "bundle_count": len(bundle_summaries),
        "games": int(game_count),
        "compared_steps": compared_steps,
        "inspected_step_count": inspected_steps,
        "max_total_variation": max_total_variation,
        "mean_total_variation": (weighted_total_variation / compared_steps if compared_steps else 0.0),
        "top_family_pairs": _top_counter_items(
            family_pair_counts,
            key_names=("policy_a_family", "policy_b_family"),
        ),
        "top_policy_a_families": _top_counter_items(policy_a_family_counts, key_names=("family",)),
        "top_policy_b_families": _top_counter_items(policy_b_family_counts, key_names=("family",)),
        "top_recorded_families": _top_counter_items(recorded_family_counts, key_names=("family",)),
        "top_action_label_pairs": _top_counter_items(
            action_label_pair_counts,
            key_names=("policy_a_action_label", "policy_b_action_label"),
        ),
        "top_policy_a_action_labels": _top_counter_items(
            policy_a_action_label_counts,
            key_names=("action_label",),
        ),
        "top_policy_b_action_labels": _top_counter_items(
            policy_b_action_label_counts,
            key_names=("action_label",),
        ),
        "bundle_summaries": list(bundle_summaries),
        "inspection_errors": list(inspection_errors),
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


def _counter_payload(counter: Counter[Any], *, key_names: tuple[str, ...]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, count in sorted(
        counter.items(), key=lambda item: (-int(item[1]), tuple(str(part) for part in _as_tuple(item[0])))
    ):
        payload: dict[str, Any] = {"count": int(count)}
        for key_name, part in zip(key_names, _as_tuple(key), strict=False):
            payload[key_name] = part
        items.append(payload)
    return items


def _top_counter_items(counter: Counter[Any], *, key_names: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    return _counter_payload(counter, key_names=key_names)[:limit]


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _audit_run_id256(
    *,
    policy_id: str,
    episodes_jsonl: Path,
    output_run_dir: Path,
    paired_seeds: Sequence[int],
) -> str:
    payload = canonical_json_bytes(
        {
            "kind": "b2_disagreement_audit_v1",
            "policy_id": policy_id,
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
