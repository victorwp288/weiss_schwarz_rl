from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import canonical_config_dict, compute_config_hash256, load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.league.registry import (
    REGISTRY_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
    SnapshotRegistry,
    snapshot_weights_relpath,
)
from weiss_rl.model import build_policy_value_model
from weiss_rl.replay.trajectory_bc import load_replay_trajectory_bc_dataset, replay_trajectory_bc_batch
from weiss_rl.training.algorithm_contracts import validate_algorithm_model_contract
from weiss_rl.training.checkpoints import (
    checkpoint_path_for_update,
    initialize_model_from_checkpoint,
    publish_checkpoint_aliases,
    write_minimal_train_checkpoint,
    write_scalars_record,
)
from weiss_rl.training.guidance import model_guidance_payload, restore_model_guidance_from_payload
from weiss_rl.training.learner_factory import build_training_learner
from weiss_rl.training.paths import training_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply paired-outcome preference replay as an auxiliary-only warmstart checkpoint"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-from-checkpoint", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--coef", type=float, default=0.08)
    parser.add_argument(
        "--optimizer-lr-scale",
        type=float,
        default=1.0,
        help="Multiply optimizer learning rates for this auxiliary-only warmstart run.",
    )
    parser.add_argument("--aggregation", choices=("mean", "sum", "edge_mean"), default="mean")
    parser.add_argument("--group-balance", action="store_true")
    parser.add_argument(
        "--pair-weight",
        action="append",
        default=[],
        metavar="PAIR_ID=WEIGHT",
        help=(
            "Upweight a specific preference pair id during auxiliary replay. "
            "May be repeated, for example --pair-weight 9=8.0."
        ),
    )
    parser.add_argument("--target-logp-retention-coef", type=float, default=0.0)
    parser.add_argument("--target-logp-retention-margin", type=float, default=0.0)
    parser.add_argument(
        "--target-logp-retention-role",
        choices=("all", "preferred", "rejected"),
        default="preferred",
    )
    parser.add_argument(
        "--target-logp-retention-reference-top-only",
        action="store_true",
        help="Apply target-logp retention only on rows where the reference policy ranked the replay action top.",
    )
    parser.add_argument(
        "--target-logp-retention-pair-role",
        action="append",
        default=[],
        metavar="PAIR_ID:ROLE",
        help=(
            "Scope target-logp retention to a preference pair and role. "
            "ROLE is preferred, rejected, or all. May be repeated."
        ),
    )
    parser.add_argument("--top-action-retention-coef", type=float, default=0.0)
    parser.add_argument("--top-action-retention-margin", type=float, default=0.0)
    parser.add_argument(
        "--top-action-retention-role",
        choices=("all", "preferred", "rejected"),
        default="all",
    )
    parser.add_argument(
        "--top-action-retention-reference-top-only",
        action="store_true",
        help="Apply top-action retention only on rows where the reference policy ranked the replay action top.",
    )
    parser.add_argument(
        "--top-action-retention-pair-role",
        action="append",
        default=[],
        metavar="PAIR_ID:ROLE",
        help=(
            "Scope top-action retention to a preference pair and role. "
            "ROLE is preferred, rejected, or all. May be repeated."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if int(args.epochs) <= 0:
        parser.error("--epochs must be positive")
    if int(args.batch_episodes) <= 0:
        parser.error("--batch-episodes must be positive")
    if float(args.beta) <= 0.0:
        parser.error("--beta must be positive")
    if float(args.coef) < 0.0:
        parser.error("--coef must be nonnegative")
    if float(args.optimizer_lr_scale) <= 0.0:
        parser.error("--optimizer-lr-scale must be positive")
    if float(args.target_logp_retention_coef) < 0.0:
        parser.error("--target-logp-retention-coef must be nonnegative")
    if float(args.target_logp_retention_margin) < 0.0:
        parser.error("--target-logp-retention-margin must be nonnegative")
    if float(args.top_action_retention_coef) < 0.0:
        parser.error("--top-action-retention-coef must be nonnegative")
    if float(args.top_action_retention_margin) < 0.0:
        parser.error("--top-action-retention-margin must be nonnegative")
    try:
        pair_weights = _parse_pair_weights(args.pair_weight)
        target_retention_selectors = _parse_pair_role_selectors(args.target_logp_retention_pair_role)
        top_action_retention_selectors = _parse_pair_role_selectors(args.top_action_retention_pair_role)
    except ValueError as exc:
        parser.error(str(exc))

    dataset = load_replay_trajectory_bc_dataset(args.dataset)
    if int(dataset.metadata.get("train_rows", 0)) <= 0:
        parser.error("dataset has no trainable rows")
    stack = load_stack_config(args.stack_config)
    training_config = stack.config.training
    model_config = stack.config.model
    if training_config is None or model_config is None:
        parser.error("stack config must include training and model sections")
    algorithm = str(training_config.algorithm).strip()
    validate_algorithm_model_contract(
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        encoder_kind=model_config.encoder_kind,
    )
    spec_hash = str(dataset.metadata.get("spec_hash256") or "").strip()
    if not spec_hash:
        parser.error("dataset metadata is missing spec_hash256")
    contract = load_verified_simulator_contract(stack.root, expected_spec_hash=spec_hash)
    observation_dim = int(dataset.obs.shape[-1])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")

    output_layout = ArtifactLayout.from_run_dir(args.output_run_dir)
    output_layout.ensure_directories()
    _write_run_contract_artifacts(
        output_layout=output_layout,
        stack=stack,
        source_run_dir=Path(str(dataset.metadata.get("run_dir", ""))) if dataset.metadata.get("run_dir") else None,
        spec_hash=contract.spec_hash256,
    )
    paths = training_paths(output_layout.run_dir)

    model = build_policy_value_model(
        observation_dim=observation_dim,
        config=model_config,
        action_dim=action_dim,
        observation_spec=contract.spec_bundle.get("observation"),
        spec_bundle=contract.spec_bundle,
    ).to(device)
    learner = build_training_learner(
        algorithm=algorithm,
        model=model,
        compiled_model=None,
        training_config=training_config,
        training_paths=paths,
        pass_action_id=pass_action_id,
        checkpoint_interval_updates=1,
    )
    source_state = initialize_model_from_checkpoint(
        checkpoint_path=args.init_from_checkpoint,
        learner=learner,
        device=device,
        expected_spec_hash256=contract.spec_hash256,
        algorithm=algorithm,
        restore_model_guidance=restore_model_guidance_from_payload,
    )
    learner.update_count = int(source_state.update_count)
    learner.policy_version = int(source_state.policy_version)
    learner.total_samples_processed = int(source_state.total_samples_processed)
    optimizer = getattr(learner, "optimizer", None)
    if optimizer is None and hasattr(learner, "_optimizer_for_step"):
        optimizer = learner._optimizer_for_step()
    optimizer_lr_summary = _scale_optimizer_learning_rates(optimizer, scale=float(args.optimizer_lr_scale))

    rng = np.random.default_rng(int(args.seed))
    start_time = time.time()
    latest_metrics: dict[str, float] = {}
    aux_steps = 0
    total_context_episodes = 0
    for epoch in range(int(args.epochs)):
        order = rng.permutation(dataset.episode_count)
        for batch_start in range(0, dataset.episode_count, int(args.batch_episodes)):
            episode_indices = order[batch_start : batch_start + int(args.batch_episodes)].astype(np.int64).tolist()
            opponent_context_indices = _opponent_context_indices_for_episodes(
                learner.model,
                dataset,
                episode_indices=episode_indices,
            )
            total_context_episodes += int(np.count_nonzero(opponent_context_indices))
            hidden = _initial_hidden_state(
                learner.model,
                batch_size=len(episode_indices),
                device=device,
                opponent_context_indices=opponent_context_indices,
            )
            batch = replay_trajectory_bc_batch(
                dataset,
                episode_indices=episode_indices,
                initial_hidden_state=hidden,
                opponent_context_indices=opponent_context_indices,
            )
            preference_group_indices = _preference_group_indices_for_episodes(dataset, episode_indices=episode_indices)
            if preference_group_indices is not None:
                batch["preference_group_id"] = np.broadcast_to(
                    preference_group_indices.reshape(1, -1),
                    np.asarray(batch["actions"]).shape,
                ).copy()
            if pair_weights:
                batch["preference_pair_weight"] = _preference_pair_weight_matrix(
                    batch.get("preference_pair_id"),
                    pair_weights,
                )
            if target_retention_selectors:
                batch["preference_retention_mask"] = _preference_pair_role_mask(
                    batch.get("preference_pair_id"),
                    batch.get("preference_role"),
                    target_retention_selectors,
                )
            if top_action_retention_selectors:
                batch["preference_top_action_retention_mask"] = _preference_pair_role_mask(
                    batch.get("preference_pair_id"),
                    batch.get("preference_role"),
                    top_action_retention_selectors,
                )
            latest_metrics = learner.paired_outcome_preference_update(
                batch,
                beta=float(args.beta),
                coef=float(args.coef),
                aggregation=str(args.aggregation),
                group_balance=bool(args.group_balance),
                retention_coef=float(args.target_logp_retention_coef),
                retention_margin=float(args.target_logp_retention_margin),
                retention_role=str(args.target_logp_retention_role),
                retention_reference_top_only=bool(args.target_logp_retention_reference_top_only),
                top_action_retention_coef=float(args.top_action_retention_coef),
                top_action_retention_margin=float(args.top_action_retention_margin),
                top_action_retention_role=str(args.top_action_retention_role),
                top_action_retention_reference_top_only=bool(args.top_action_retention_reference_top_only),
            )
            aux_steps += 1
            latest_metrics.update(
                {
                    "paired_outcome_preference_warmstart_phase": 1.0,
                    "paired_outcome_preference_warmstart_epoch": float(epoch + 1),
                    "paired_outcome_preference_warmstart_aux_step": float(aux_steps),
                    "paired_outcome_preference_warmstart_batch_episodes": float(len(episode_indices)),
                    "paired_outcome_preference_warmstart_dataset_train_rows": float(dataset.metadata["train_rows"]),
                    "paired_outcome_preference_warmstart_context_episodes": float(total_context_episodes),
                    "paired_outcome_preference_warmstart_optimizer_lr_scale": float(args.optimizer_lr_scale),
                    "paired_outcome_preference_warmstart_pair_weight_count": float(len(pair_weights)),
                    "paired_outcome_preference_warmstart_target_retention_selector_count": float(
                        len(target_retention_selectors)
                    ),
                    "paired_outcome_preference_warmstart_top_action_retention_selector_count": float(
                        len(top_action_retention_selectors)
                    ),
                }
            )
            write_scalars_record(
                scalars_path=paths.scalars_path,
                learner=learner,
                metrics=latest_metrics,
                start_time=start_time,
            )

    checkpoint_path = checkpoint_path_for_update(paths.checkpoints_dir, update_count=int(learner.update_count))
    if checkpoint_path.is_file():
        checkpoint_path = paths.checkpoints_dir / f"checkpoint_{int(learner.update_count):06d}_preference.pt"
    write_minimal_train_checkpoint(
        checkpoint_path=checkpoint_path,
        learner=learner,
        device=device,
        config_hash256=compute_config_hash256(stack),
        spec_hash256=contract.spec_hash256,
        algorithm=algorithm,
        recurrent_core=model_config.recurrent_core,
        guidance_payload=model_guidance_payload(learner.model),
    )
    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=paths,
        run_dir=output_layout.run_dir,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics=latest_metrics,
        dev_eval_summary=None,
    )
    snapshot_payload = _publish_preference_snapshot(
        output_run_dir=output_layout.run_dir,
        checkpoint_path=checkpoint_path,
        update_count=int(learner.update_count),
    )
    summary = {
        "format": "paired_outcome_preference_warmstart_summary_v1",
        "dataset": args.dataset.resolve().as_posix(),
        "init_from_checkpoint": args.init_from_checkpoint.resolve().as_posix(),
        "output_run_dir": output_layout.run_dir.resolve().as_posix(),
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "latest_checkpoint_path": paths.latest_checkpoint_path.resolve().as_posix(),
        "aux_steps": aux_steps,
        "epochs": int(args.epochs),
        "batch_episodes": int(args.batch_episodes),
        "beta": float(args.beta),
        "coef": float(args.coef),
        "optimizer_lr_scale": float(args.optimizer_lr_scale),
        "optimizer_lr_summary": optimizer_lr_summary,
        "aggregation": str(args.aggregation),
        "group_balance": bool(args.group_balance),
        "pair_weights": {str(pair_id): float(weight) for pair_id, weight in sorted(pair_weights.items())},
        "target_logp_retention_coef": float(args.target_logp_retention_coef),
        "target_logp_retention_margin": float(args.target_logp_retention_margin),
        "target_logp_retention_role": str(args.target_logp_retention_role),
        "target_logp_retention_reference_top_only": bool(args.target_logp_retention_reference_top_only),
        "target_logp_retention_pair_roles": _serialize_pair_role_selectors(target_retention_selectors),
        "top_action_retention_coef": float(args.top_action_retention_coef),
        "top_action_retention_margin": float(args.top_action_retention_margin),
        "top_action_retention_role": str(args.top_action_retention_role),
        "top_action_retention_reference_top_only": bool(args.top_action_retention_reference_top_only),
        "top_action_retention_pair_roles": _serialize_pair_role_selectors(top_action_retention_selectors),
        "context_episodes": total_context_episodes,
        "dataset_metadata": dataset.metadata,
        "latest_metrics": latest_metrics,
        "checkpoint_tracker": tracker,
        "snapshot": snapshot_payload,
    }
    summary_path = output_layout.diagnostics_dir / "paired_outcome_preference_warmstart_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Paired outcome preference warmstart wrote {checkpoint_path} after {aux_steps} auxiliary steps; "
        f"latest alias is {paths.latest_checkpoint_path}; summary written to {summary_path}"
    )
    return 0


def _parse_pair_weights(raw_values: Sequence[str] | None) -> dict[int, float]:
    weights: dict[int, float] = {}
    for raw_value in raw_values or []:
        text = str(raw_value).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError("--pair-weight values must use PAIR_ID=WEIGHT")
        raw_pair_id, raw_weight = text.split("=", 1)
        try:
            pair_id = int(raw_pair_id.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --pair-weight pair id: {raw_pair_id!r}") from exc
        if pair_id < 0:
            raise ValueError("--pair-weight pair id must be nonnegative")
        try:
            weight = float(raw_weight.strip())
        except ValueError as exc:
            raise ValueError(f"invalid --pair-weight value for pair {pair_id}: {raw_weight!r}") from exc
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("--pair-weight weights must be finite and positive")
        weights[pair_id] = weight
    return weights


def _parse_pair_role_selectors(raw_values: Sequence[str] | None) -> tuple[tuple[int, int | None], ...]:
    selectors: list[tuple[int, int | None]] = []
    for raw_value in raw_values or []:
        text = str(raw_value).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError("retention pair-role values must use PAIR_ID:ROLE")
        raw_pair_id, raw_role = text.split(":", 1)
        try:
            pair_id = int(raw_pair_id.strip())
        except ValueError as exc:
            raise ValueError(f"invalid retention pair id: {raw_pair_id!r}") from exc
        if pair_id < 0:
            raise ValueError("retention pair id must be nonnegative")
        role_text = raw_role.strip().lower()
        if role_text in {"all", "*"}:
            role: int | None = None
        elif role_text in {"preferred", "pref", "1"}:
            role = 1
        elif role_text in {"rejected", "rej", "0"}:
            role = 0
        else:
            raise ValueError("retention role must be preferred, rejected, or all")
        selector = (pair_id, role)
        if selector not in selectors:
            selectors.append(selector)
    return tuple(selectors)


def _serialize_pair_role_selectors(selectors: Sequence[tuple[int, int | None]]) -> list[str]:
    role_names = {None: "all", 1: "preferred", 0: "rejected"}
    return [f"{int(pair_id)}:{role_names[role]}" for pair_id, role in selectors]


def _preference_pair_weight_matrix(preference_pair_ids: Any, pair_weights: Mapping[int, float]) -> np.ndarray:
    if preference_pair_ids is None:
        raise ValueError("batch is missing preference_pair_id; cannot apply --pair-weight")
    pair_ids = np.asarray(preference_pair_ids)
    weights = np.ones(pair_ids.shape, dtype=np.float32)
    for pair_id, weight in pair_weights.items():
        weights[pair_ids == int(pair_id)] = float(weight)
    return weights


def _preference_pair_role_mask(
    preference_pair_ids: Any,
    preference_roles: Any,
    selectors: Sequence[tuple[int, int | None]],
) -> np.ndarray:
    if preference_pair_ids is None:
        raise ValueError("batch is missing preference_pair_id; cannot apply retention pair-role selectors")
    if preference_roles is None:
        raise ValueError("batch is missing preference_role; cannot apply retention pair-role selectors")
    pair_ids = np.asarray(preference_pair_ids)
    roles = np.asarray(preference_roles)
    if pair_ids.shape != roles.shape:
        raise ValueError("preference_pair_id and preference_role must have the same shape")
    mask = np.zeros(pair_ids.shape, dtype=np.float32)
    for pair_id, role in selectors:
        selector_mask = pair_ids == int(pair_id)
        if role is not None:
            selector_mask = selector_mask & (roles == int(role))
        mask[selector_mask] = 1.0
    return mask


def _opponent_context_indices_for_episodes(model: Any, dataset: Any, *, episode_indices: list[int]) -> np.ndarray:
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    selected_policy_ids = [
        opponent_ids[int(index)] if int(index) < len(opponent_ids) else "" for index in episode_indices
    ]
    if model is None or not hasattr(model, "opponent_context_indices_for_policy_ids"):
        return np.zeros((len(episode_indices),), dtype=np.int64)
    return np.asarray(model.opponent_context_indices_for_policy_ids(selected_policy_ids), dtype=np.int64).reshape(-1)


def _source_opponent_policy_ids_by_episode(dataset: Any) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return []
    ids: list[str] = []
    for bundle in bundles:
        raw_id = bundle.get("source_opponent_policy_id") if isinstance(bundle, Mapping) else None
        ids.append(str(raw_id or "").strip())
    return ids


def _preference_group_indices_for_episodes(dataset: Any, *, episode_indices: list[int]) -> np.ndarray | None:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return None
    labels: list[str] = []
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            labels.append("")
            continue
        labels.append(str(bundle.get("merge_source_dataset_label") or bundle.get("source_dataset_label") or ""))
    nonempty_labels = sorted({label for label in labels if label})
    if not nonempty_labels:
        return None
    label_to_index = {label: index for index, label in enumerate(nonempty_labels)}
    indices = [
        label_to_index.get(labels[int(index)] if int(index) < len(labels) else "", -1) for index in episode_indices
    ]
    return np.asarray(indices, dtype=np.int64)


def _scale_optimizer_learning_rates(optimizer: Any, *, scale: float) -> dict[str, Any]:
    """Apply a run-local LR multiplier and return a reproducibility summary."""

    groups = []
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list):
        return {"scale": float(scale), "groups": groups}
    for index, group in enumerate(param_groups):
        if not isinstance(group, dict) or "lr" not in group:
            continue
        original_lr = float(group["lr"])
        scaled_lr = original_lr * float(scale)
        group["lr"] = scaled_lr
        groups.append({"index": int(index), "original_lr": original_lr, "scaled_lr": scaled_lr})
    return {"scale": float(scale), "groups": groups}


def _initial_hidden_state(
    model: Any,
    *,
    batch_size: int,
    device: torch.device,
    opponent_context_indices: np.ndarray,
) -> np.ndarray | None:
    if model is None or not hasattr(model, "initial_seat_hidden"):
        return None
    try:
        hidden = model.initial_seat_hidden(
            int(batch_size),
            device=device,
            opponent_context_indices=opponent_context_indices,
        )
    except TypeError:
        hidden = model.initial_seat_hidden(int(batch_size), device=device)
    return hidden.detach().cpu().numpy()


def _write_run_contract_artifacts(
    *,
    output_layout: ArtifactLayout,
    stack: Any,
    source_run_dir: Path | None,
    spec_hash: str,
) -> None:
    config_hash = compute_config_hash256(stack)
    output_layout.config_hash_path.write_text(config_hash + "\n", encoding="utf-8")
    output_layout.config_json_path.write_text(
        json.dumps(canonical_config_dict(stack), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_layout.spec_hash_path.write_text(str(spec_hash) + "\n", encoding="utf-8")
    manifest = {
        "format": "paired_outcome_preference_warmstart_manifest_v1",
        "run_id256": hashlib.sha256(
            json.dumps(
                {
                    "kind": "paired_outcome_preference_warmstart",
                    "run_dir": output_layout.run_dir.resolve().as_posix(),
                    "config_hash256": config_hash,
                    "spec_hash256": str(spec_hash),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "config_hash256": config_hash,
        "spec_hash256": str(spec_hash),
    }
    output_layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if source_run_dir is not None:
        source_spec = source_run_dir / "spec_bundle.json"
        if source_spec.is_file():
            shutil.copy2(source_spec, output_layout.spec_bundle_path)


def _publish_preference_snapshot(*, output_run_dir: Path, checkpoint_path: Path, update_count: int) -> dict[str, Any]:
    policy_id = "paired_outcome_preference_latest"
    weights_relpath = snapshot_weights_relpath(policy_id)
    weights_path = output_run_dir / weights_relpath
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, weights_path)
    weights_sha256 = _sha256_file(weights_path)
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "paired_outcome_preference_snapshot_meta_v1",
        "policy_id": policy_id,
        "update": int(update_count),
        "weights_sha256": weights_sha256,
        "source_checkpoint_path": checkpoint_path.resolve().as_posix(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id=policy_id,
        update=int(update_count),
        weights_sha256=weights_sha256,
        path=weights_relpath,
    )
    registry.pin_snapshot(policy_id)
    registry_path = output_run_dir / "training" / "snapshots" / REGISTRY_FILENAME
    registry.save(registry_path)
    return {
        "policy_id": policy_id,
        "weights_path": weights_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "registry_path": registry_path.as_posix(),
        "weights_sha256": weights_sha256,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
