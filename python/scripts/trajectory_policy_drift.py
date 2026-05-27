from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from weiss_rl.artifacts.reproducibility import require_fixed_python_hash_seed
from weiss_rl.config import load_stack_config
from weiss_rl.core.legal_actions import LegalActionBatch
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.diagnostics.trajectory_policy_drift import (
    summarize_policy_drift,
    summarize_policy_drift_by_group,
    summarize_policy_scores,
)
from weiss_rl.models.loading import load_snapshot_eval_model
from weiss_rl.replay.trajectory_bc import ReplayTrajectoryDataset, load_replay_trajectory_bc_dataset


@dataclass(frozen=True, slots=True)
class PolicySpec:
    label: str
    run_dir: Path
    checkpoint_relpath: str


@dataclass(frozen=True, slots=True)
class PolicyScores:
    label: str
    top_actions: np.ndarray
    top_log_probs: np.ndarray
    target_log_probs: np.ndarray
    target_probabilities: np.ndarray
    top_families: np.ndarray
    values: np.ndarray
    opponent_context_episode_count: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score identical replay trajectory rows with multiple policies and report drift"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        help="Policy spec as LABEL|RUN_DIR|CHECKPOINT_RELPATH. Repeat for direct/update checkpoints.",
    )
    parser.add_argument("--reference-label", default=None, help="Policy label used as the drift reference")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    python_hash_seed = require_fixed_python_hash_seed("trajectory_policy_drift")
    dataset = load_replay_trajectory_bc_dataset(args.dataset)
    if int(dataset.metadata.get("train_rows", 0)) <= 0:
        raise SystemExit("dataset has no trainable rows")
    stack = load_stack_config(args.stack_config)
    spec_hash = str(dataset.metadata.get("spec_hash256") or "").strip()
    if not spec_hash:
        raise SystemExit("dataset metadata is missing spec_hash256")
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash)
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    action_catalog = _action_catalog_from_stack_spec(contract.spec_bundle)
    family_names, family_by_action = _family_metadata(action_catalog=action_catalog, action_dim=action_dim)
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    _configure_torch_determinism(torch_threads=int(args.torch_threads))
    policy_specs = [_parse_policy_spec(raw) for raw in args.policy]
    if len({spec.label for spec in policy_specs}) != len(policy_specs):
        raise SystemExit("--policy labels must be unique")
    reference_label = str(args.reference_label or policy_specs[0].label)

    scores_by_label = {
        spec.label: _score_policy(
            spec=spec,
            stack=stack,
            dataset=dataset,
            contract_spec_bundle=contract.spec_bundle,
            action_dim=action_dim,
            family_by_action=family_by_action,
            device=device,
        )
        for spec in policy_specs
    }
    if reference_label not in scores_by_label:
        raise SystemExit(f"reference label {reference_label!r} was not among --policy labels")

    target_actions = dataset.actions.reshape(-1)
    target_families = family_by_action[np.clip(target_actions, 0, family_by_action.shape[0] - 1)]
    row_mask = dataset.policy_train_mask.reshape(-1)
    row_coordinates = _row_coordinates(dataset)
    row_group_labels = _row_group_labels(dataset)
    policy_summaries = []
    for label, scores in scores_by_label.items():
        summary = summarize_policy_scores(
            label=label,
            top_actions=scores.top_actions,
            target_actions=target_actions,
            target_probabilities=scores.target_probabilities,
            target_log_probs=scores.target_log_probs,
            top_families=scores.top_families,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            values=scores.values,
        )
        summary["opponent_context_episode_count"] = int(scores.opponent_context_episode_count)
        policy_summaries.append(summary)
    reference = scores_by_label[reference_label]
    drift_summaries = []
    for label, scores in scores_by_label.items():
        if label == reference_label:
            continue
        summary = summarize_policy_drift(
            reference_label=reference_label,
            candidate_label=label,
            reference_top_actions=reference.top_actions,
            candidate_top_actions=scores.top_actions,
            reference_target_probabilities=reference.target_probabilities,
            candidate_target_probabilities=scores.target_probabilities,
            reference_top_families=reference.top_families,
            candidate_top_families=scores.top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            reference_target_log_probs=reference.target_log_probs,
            candidate_target_log_probs=scores.target_log_probs,
            reference_top_log_probs=reference.top_log_probs,
            candidate_top_log_probs=scores.top_log_probs,
            reference_values=reference.values,
            candidate_values=scores.values,
            row_coordinates=row_coordinates,
            max_examples=int(args.max_examples),
        )
        summary["preference_role_drift_summaries"] = summarize_policy_drift_by_group(
            group_name="preference_role_label",
            group_labels=row_group_labels["preference_role_label"],
            reference_label=reference_label,
            candidate_label=label,
            reference_top_actions=reference.top_actions,
            candidate_top_actions=scores.top_actions,
            reference_target_probabilities=reference.target_probabilities,
            candidate_target_probabilities=scores.target_probabilities,
            reference_top_families=reference.top_families,
            candidate_top_families=scores.top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            reference_target_log_probs=reference.target_log_probs,
            candidate_target_log_probs=scores.target_log_probs,
            reference_top_log_probs=reference.top_log_probs,
            candidate_top_log_probs=scores.top_log_probs,
            reference_values=reference.values,
            candidate_values=scores.values,
            row_coordinates=row_coordinates,
            max_examples=int(args.max_examples),
        )
        summary["source_opponent_drift_summaries"] = summarize_policy_drift_by_group(
            group_name="source_opponent_policy_id",
            group_labels=row_group_labels["source_opponent_policy_id"],
            reference_label=reference_label,
            candidate_label=label,
            reference_top_actions=reference.top_actions,
            candidate_top_actions=scores.top_actions,
            reference_target_probabilities=reference.target_probabilities,
            candidate_target_probabilities=scores.target_probabilities,
            reference_top_families=reference.top_families,
            candidate_top_families=scores.top_families,
            target_actions=target_actions,
            target_families=target_families,
            row_mask=row_mask,
            family_names=family_names,
            reference_target_log_probs=reference.target_log_probs,
            candidate_target_log_probs=scores.target_log_probs,
            reference_top_log_probs=reference.top_log_probs,
            candidate_top_log_probs=scores.top_log_probs,
            reference_values=reference.values,
            candidate_values=scores.values,
            row_coordinates=row_coordinates,
            max_examples=int(args.max_examples),
        )
        drift_summaries.append(summary)
    report = {
        "format": "trajectory_policy_drift_v1",
        "stack_config": args.stack_config.as_posix(),
        "dataset": args.dataset.as_posix(),
        "dataset_metadata": {
            "bundle_count": int(dataset.metadata.get("bundle_count", 0)),
            "train_rows": int(dataset.metadata.get("train_rows", 0)),
            "row_count": int(dataset.metadata.get("row_count", 0)),
            "unsupported_target_rows": int(dataset.metadata.get("unsupported_target_rows", 0)),
            "spec_hash256": dataset.metadata.get("spec_hash256"),
        },
        "device": str(device),
        "python_hash_seed": int(python_hash_seed),
        "torch_threads": int(args.torch_threads),
        "output_json": args.output_json.as_posix(),
        "reference_label": reference_label,
        "policies": [
            {
                "label": spec.label,
                "run_dir": spec.run_dir.as_posix(),
                "checkpoint_relpath": spec.checkpoint_relpath,
            }
            for spec in policy_specs
        ],
        "policy_summaries": policy_summaries,
        "drift_summaries": drift_summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report)
    return 0


def _parse_policy_spec(raw: str) -> PolicySpec:
    parts = [part.strip() for part in str(raw).split("|")]
    if len(parts) != 3 or not all(parts):
        raise SystemExit("--policy must use LABEL|RUN_DIR|CHECKPOINT_RELPATH")
    return PolicySpec(label=parts[0], run_dir=Path(parts[1]), checkpoint_relpath=parts[2].replace("\\", "/"))


def _configure_torch_determinism(*, torch_threads: int) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _action_catalog_from_stack_spec(spec_bundle: dict[str, Any]) -> Any:
    from weiss_rl.core.action_catalog import ActionCatalog

    return ActionCatalog.from_spec_bundle(spec_bundle)


def _family_metadata(*, action_catalog: Any, action_dim: int) -> tuple[tuple[str, ...], np.ndarray]:
    family_names = tuple(str(family.name) for family in action_catalog.families)
    family_index = {name: index for index, name in enumerate(family_names)}
    family_by_action = np.full((int(action_dim),), -1, dtype=np.int64)
    for action_id in range(int(action_dim)):
        decoded = action_catalog.decode(action_id)
        family_by_action[action_id] = int(family_index.get(decoded.family, -1))
    return family_names, family_by_action


def _score_policy(
    *,
    spec: PolicySpec,
    stack: Any,
    dataset: ReplayTrajectoryDataset,
    contract_spec_bundle: dict[str, Any],
    action_dim: int,
    family_by_action: np.ndarray,
    device: torch.device,
) -> PolicyScores:
    model = load_snapshot_eval_model(
        run_dir=spec.run_dir,
        snapshot_path=spec.checkpoint_relpath,
        stack=stack,
        observation_dim=int(dataset.obs.shape[-1]),
        action_dim=int(action_dim),
        observation_spec=contract_spec_bundle.get("observation"),
        spec_bundle=contract_spec_bundle,
    ).to(device)
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    opponent_context_indices = np.asarray(
        model.opponent_context_indices_for_policy_ids(opponent_ids),
        dtype=np.int64,
    )
    legal_actions = LegalActionBatch.from_packed(
        dataset.legal_ids,
        dataset.legal_offsets,
        meta=dataset.legal_action_meta,
        action_space=int(action_dim),
    )
    obs = torch.as_tensor(dataset.obs, device=device, dtype=torch.float32)
    acting_seat = torch.as_tensor(dataset.actor, device=device, dtype=torch.long)
    reset_before_step = torch.as_tensor(dataset.reset_before_step, device=device, dtype=torch.bool)
    initial_hidden = model.initial_seat_hidden(
        dataset.episode_count,
        device=device,
        opponent_context_indices=opponent_context_indices,
    )
    opponent_context_index = torch.as_tensor(
        np.broadcast_to(opponent_context_indices.reshape(1, -1), dataset.actions.shape).copy(),
        device=device,
        dtype=torch.long,
    )
    actions = torch.as_tensor(_safe_actions_for_scoring(dataset), device=device, dtype=torch.long)
    with torch.inference_mode():
        if bool(getattr(model, "supports_factorized_legal_policy", False)):
            result = model.evaluate_factorized_sequence_packed_seat_aware(
                obs,
                acting_seat,
                initial_hidden,
                legal_actions=legal_actions,
                actions=actions,
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
            if result.top_action_ids is None or result.action_logp is None:
                raise RuntimeError(f"factorized policy did not return top actions/logp: {spec.label}")
            top_actions = result.top_action_ids.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
            target_logp = result.action_logp.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            values = result.values.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            top_action_tensor = torch.as_tensor(
                top_actions.reshape(dataset.actions.shape), device=device, dtype=torch.long
            )
            top_result = model.evaluate_factorized_sequence_packed_seat_aware(
                obs,
                acting_seat,
                initial_hidden,
                legal_actions=legal_actions,
                actions=top_action_tensor,
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
            if top_result.action_logp is None:
                raise RuntimeError(f"factorized policy did not return top-action logp: {spec.label}")
            top_logp = top_result.action_logp.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        else:
            packed_logits, value_tensor, _hidden = model.forward_sequence_packed_seat_aware(
                obs,
                acting_seat,
                initial_hidden,
                legal_actions=legal_actions,
                scoring_mode="learner",
                reset_before_step=reset_before_step,
                opponent_context_index=opponent_context_index,
            )
            top_actions, target_logp, top_logp = _dense_policy_scores_from_packed_logits(
                packed_logits.detach().cpu().numpy().astype(np.float64, copy=False),
                legal_ids=dataset.legal_ids,
                legal_offsets=dataset.legal_offsets,
                target_actions=dataset.actions.reshape(-1),
            )
            values = value_tensor.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    target_prob = np.exp(np.clip(target_logp, -80.0, 0.0))
    top_families = family_by_action[np.clip(top_actions, 0, family_by_action.shape[0] - 1)]
    return PolicyScores(
        label=spec.label,
        top_actions=top_actions,
        top_log_probs=top_logp,
        target_log_probs=target_logp,
        target_probabilities=target_prob,
        top_families=top_families,
        values=values,
        opponent_context_episode_count=int(np.count_nonzero(opponent_context_indices)),
    )


def _dense_policy_scores_from_packed_logits(
    packed_logits: np.ndarray,
    *,
    legal_ids: np.ndarray,
    legal_offsets: np.ndarray,
    target_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(legal_offsets, dtype=np.int64)
    ids = np.asarray(legal_ids, dtype=np.int64)
    row_count = int(offsets.shape[0] - 1)
    top_actions = np.full((row_count,), -1, dtype=np.int64)
    target_logp = np.full((row_count,), -np.inf, dtype=np.float64)
    top_logp = np.full((row_count,), -np.inf, dtype=np.float64)
    for row_index in range(row_count):
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        row_ids = ids[start:stop]
        row_logits = packed_logits[start:stop]
        if row_ids.size == 0:
            continue
        top_actions[row_index] = int(row_ids[int(np.argmax(row_logits))])
        finite = np.isfinite(row_logits)
        if not bool(np.any(finite)):
            log_probs = np.full_like(row_logits, -np.log(float(row_logits.size)))
        else:
            finite_logits = row_logits[finite]
            max_logit = float(np.max(finite_logits))
            log_z = max_logit + float(np.log(np.sum(np.exp(finite_logits - max_logit))))
            log_probs = np.full_like(row_logits, -np.inf)
            log_probs[finite] = row_logits[finite] - log_z
        target_positions = np.nonzero(row_ids == int(target_actions[row_index]))[0]
        if target_positions.size:
            target_logp[row_index] = float(log_probs[int(target_positions[0])])
        top_positions = np.nonzero(row_ids == int(top_actions[row_index]))[0]
        if top_positions.size:
            top_logp[row_index] = float(log_probs[int(top_positions[0])])
    return top_actions, target_logp, top_logp


def _safe_actions_for_scoring(dataset: ReplayTrajectoryDataset) -> np.ndarray:
    """Use replay actions on train rows and legal placeholders elsewhere."""

    actions = np.asarray(dataset.actions, dtype=np.int64).reshape(-1).copy()
    train_mask = np.asarray(dataset.policy_train_mask, dtype=np.bool_).reshape(-1)
    offsets = np.asarray(dataset.legal_offsets, dtype=np.int64)
    legal_ids = np.asarray(dataset.legal_ids, dtype=np.int64)
    for row_index in np.nonzero(~train_mask)[0].tolist():
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        if stop > start:
            actions[row_index] = int(legal_ids[start])
    return actions.reshape(dataset.actions.shape)


def _source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    selected = dataset.metadata.get("selected_bundles")
    selected_bundles = selected if isinstance(selected, list) else []
    opponent_ids: list[str] = []
    for episode_index in range(int(dataset.episode_count)):
        bundle = selected_bundles[episode_index] if episode_index < len(selected_bundles) else {}
        if isinstance(bundle, dict):
            opponent_ids.append(str(bundle.get("source_opponent_policy_id") or "").strip())
        else:
            opponent_ids.append("")
    return opponent_ids


def _row_coordinates(dataset: ReplayTrajectoryDataset) -> list[dict[str, Any]]:
    selected = dataset.metadata.get("selected_bundles")
    selected_bundles = selected if isinstance(selected, list) else []
    coordinates: list[dict[str, Any]] = []
    batch_size = int(dataset.episode_count)
    for step_index in range(int(dataset.time_steps)):
        for episode_index in range(batch_size):
            bundle_meta = selected_bundles[episode_index] if episode_index < len(selected_bundles) else {}
            if not isinstance(bundle_meta, dict):
                bundle_meta = {}
            coordinates.append(
                {
                    "row_index": int(step_index * batch_size + episode_index),
                    "step_index": int(step_index),
                    "episode_index": int(episode_index),
                    "pair_index": bundle_meta.get("pair_index"),
                    "swap_index": bundle_meta.get("swap_index"),
                    "focal_seat": bundle_meta.get("focal_seat"),
                    "episode_seed": bundle_meta.get("episode_seed"),
                    "preference_pair_id": bundle_meta.get("preference_pair_id"),
                    "preference_role": bundle_meta.get("preference_role"),
                    "preference_role_label": bundle_meta.get("preference_role_label"),
                    "source_opponent_policy_id": bundle_meta.get("source_opponent_policy_id"),
                }
            )
    return coordinates


def _row_group_labels(dataset: ReplayTrajectoryDataset) -> dict[str, np.ndarray]:
    selected = dataset.metadata.get("selected_bundles")
    selected_bundles = selected if isinstance(selected, list) else []
    role_labels: list[str] = []
    opponent_ids: list[str] = []
    batch_size = int(dataset.episode_count)
    for _step_index in range(int(dataset.time_steps)):
        for episode_index in range(batch_size):
            bundle_meta = selected_bundles[episode_index] if episode_index < len(selected_bundles) else {}
            if not isinstance(bundle_meta, dict):
                bundle_meta = {}
            role_labels.append(str(bundle_meta.get("preference_role_label") or "").strip())
            opponent_ids.append(str(bundle_meta.get("source_opponent_policy_id") or "").strip())
    return {
        "preference_role_label": np.asarray(role_labels, dtype=object),
        "source_opponent_policy_id": np.asarray(opponent_ids, dtype=object),
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"trajectory policy drift dataset: {report['dataset']}")
    for summary in report["policy_summaries"]:
        print(
            f"{summary['label']}: rows={summary['row_count']} "
            f"top_action_match={summary['top_action_matches_target_rate']:.4f} "
            f"top_family_match={summary['top_family_matches_target_rate']:.4f} "
            f"p_target={summary['mean_probability_on_target_action']:.4f}"
        )
    for summary in report["drift_summaries"]:
        print(
            f"{summary['reference_label']} -> {summary['candidate_label']}: "
            f"top_action_changed={summary['top_action_changed_rate']:.4f} "
            f"top_family_changed={summary['top_family_changed_rate']:.4f} "
            f"lost_target_top={summary['lost_target_top_action_rate']:.4f} "
            f"mean_p_delta={summary['mean_target_action_probability_delta']:.4f}"
        )
    print(f"output: {report['output_json']}")


if __name__ == "__main__":
    raise SystemExit(main())
