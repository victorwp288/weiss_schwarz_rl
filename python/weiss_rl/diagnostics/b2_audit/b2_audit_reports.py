"""Replay-bundle labeling and file helpers for B2 disagreement audits."""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from weiss_rl.core.action_catalog import ActionCatalog, DecodedAction
from weiss_rl.diagnostics.b2_audit.b2_audit_summary_math import (
    aggregate_trajectory_summary as _shared_aggregate_trajectory_summary,
)
from weiss_rl.diagnostics.b2_audit.b2_audit_summary_math import counter_payload
from weiss_rl.diagnostics.b2_audit.b2_audit_summary_math import top_counter_items as _shared_top_counter_items
from weiss_rl.eval.policies.set import LEGACY_NO_LEAGUE_POLICY_ID, NO_LEAGUE_POLICY_ID


@dataclass(frozen=True, slots=True)
class B2AuditPlanStep:
    step_id: str
    purpose: str
    evidence: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "purpose": self.purpose,
            "evidence": list(self.evidence),
        }


B2_AUDIT_PLAN: tuple[B2AuditPlanStep, ...] = (
    B2AuditPlanStep(
        step_id="reuse_source_seeds",
        purpose="Load the paired seeds and matchup identity from an existing final-eval episodes file.",
        evidence=("episodes_jsonl", "paired seeds", "focal/opponent policy ids"),
    ),
    B2AuditPlanStep(
        step_id="resolve_policies",
        purpose="Resolve the inspected learner policy and the public heuristic opponent under the same simulator spec.",
        evidence=("snapshot registry", "policy id", "opponent policy id", "spec hash"),
    ),
    B2AuditPlanStep(
        step_id="rerun_matchup",
        purpose="Replay the seed-targeted matchup and write fresh game records plus replay bundles.",
        evidence=("audit episodes", "replay bundles", "matchup summary"),
    ),
    B2AuditPlanStep(
        step_id="inspect_replays",
        purpose="Compare learner and heuristic action surfaces on the same public states.",
        evidence=("top differences", "legal action surfaces", "action-family rankings"),
    ),
    B2AuditPlanStep(
        step_id="aggregate_findings",
        purpose="Summarize repeated disagreement families, action labels, legality filters, and margin statistics.",
        evidence=("audit summary", "trajectory summary", "top examples"),
    ),
)


def b2_audit_plan_payload() -> list[dict[str, object]]:
    return [step.as_payload() for step in B2_AUDIT_PLAN]


def build_bundle_summary(
    *,
    report: dict[str, Any],
    report_path: Path,
    bundle_path: Path,
    scheduled_game: Any,
    decoder: ActionCatalog,
) -> dict[str, Any]:
    annotated_diffs = [annotate_step_diff(diff, decoder=decoder) for diff in report.get("top_differences", [])]
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
        "family_pair_counts": counter_payload(
            family_pair_counts,
            key_names=("policy_a_family", "policy_b_family"),
        ),
        "policy_a_family_counts": counter_payload(policy_a_family_counts, key_names=("family",)),
        "policy_b_family_counts": counter_payload(policy_b_family_counts, key_names=("family",)),
        "recorded_family_counts": counter_payload(recorded_family_counts, key_names=("family",)),
        "action_label_pair_counts": counter_payload(
            action_label_pair_counts,
            key_names=("policy_a_action_label", "policy_b_action_label"),
        ),
        "policy_a_action_label_counts": counter_payload(policy_a_action_label_counts, key_names=("action_label",)),
        "policy_b_action_label_counts": counter_payload(policy_b_action_label_counts, key_names=("action_label",)),
        "top_examples": annotated_diffs[:3],
    }


def annotate_step_diff(diff: dict[str, Any], *, decoder: ActionCatalog) -> dict[str, Any]:
    recorded_action = decoder.decode(int(diff["recorded_action"]))
    policy_a_top_action = decoder.decode(int(diff["policy_a_top_action"]["action"]))
    policy_b_top_action = decoder.decode(int(diff["policy_b_top_action"]["action"]))
    annotated = dict(diff)
    annotated["recorded_action_family"] = recorded_action.family
    annotated["recorded_action_label"] = format_decoded_action(recorded_action)
    annotated["policy_a_top_action_family"] = policy_a_top_action.family
    annotated["policy_a_top_action_label"] = format_decoded_action(policy_a_top_action)
    annotated["policy_b_top_action_family"] = policy_b_top_action.family
    annotated["policy_b_top_action_label"] = format_decoded_action(policy_b_top_action)
    return annotated


def aggregate_trajectory_summary(bundle_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _shared_aggregate_trajectory_summary(bundle_summaries)


def top_counter_items(counter: Counter[Any], *, key_names: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    return _shared_top_counter_items(counter, key_names=key_names, limit=limit)


def inspection_policy_id(policy_id: str) -> str:
    normalized = str(policy_id).strip()
    if normalized == NO_LEAGUE_POLICY_ID:
        return LEGACY_NO_LEAGUE_POLICY_ID
    return normalized


def format_decoded_action(action: DecodedAction) -> str:
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


def output_run_dir_path(output_run_dir: Path, relative_path: str | Path) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else output_run_dir / path


def materialize_audit_bundle_copy(
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
