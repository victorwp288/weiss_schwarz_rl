from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
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
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_cli import (
    build_paired_outcome_preference_warmstart_parser,
    validate_paired_outcome_preference_warmstart_args,
)
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_cli import (
    parse_paired_outcome_preference_warmstart_args as parse_paired_outcome_preference_warmstart_args,
)
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_support import (
    _initial_hidden_state,
    _opponent_context_indices_for_episodes,
    _parse_pair_role_selectors,
    _parse_pair_weights,
    _preference_group_indices_for_episodes,
    _preference_pair_role_mask,
    _preference_pair_weight_matrix,
    _scale_optimizer_learning_rates,
    _serialize_pair_role_selectors,
)
from weiss_rl.training.warmstarts.paired_outcome_preference_warmstart_support import (
    _source_opponent_policy_ids_by_episode as _source_opponent_policy_ids_by_episode,
)
from weiss_rl.training.warmstarts.warmstart_artifacts import (
    sha256_file,
    warmstart_run_contract_writer,
    warmstart_snapshot_publisher,
)

_build_parser = build_paired_outcome_preference_warmstart_parser
_write_run_contract_artifacts = warmstart_run_contract_writer(
    manifest_format="paired_outcome_preference_warmstart_manifest_v1",
    run_kind="paired_outcome_preference_warmstart",
)
_publish_preference_snapshot = warmstart_snapshot_publisher(
    policy_id="paired_outcome_preference_latest",
    metadata_format="paired_outcome_preference_snapshot_meta_v1",
)
_sha256_file = sha256_file


def run_paired_outcome_preference_warmstart(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    validate_paired_outcome_preference_warmstart_args(parser, args)
    pair_weights = _parse_pair_weights(args.pair_weight)
    target_retention_selectors = _parse_pair_role_selectors(args.target_logp_retention_pair_role)
    top_action_retention_selectors = _parse_pair_role_selectors(args.top_action_retention_pair_role)

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


__all__ = [
    "_build_parser",
    "_initial_hidden_state",
    "_opponent_context_indices_for_episodes",
    "_parse_pair_role_selectors",
    "_parse_pair_weights",
    "_preference_group_indices_for_episodes",
    "_preference_pair_role_mask",
    "_preference_pair_weight_matrix",
    "_publish_preference_snapshot",
    "_scale_optimizer_learning_rates",
    "_serialize_pair_role_selectors",
    "_sha256_file",
    "_source_opponent_policy_ids_by_episode",
    "_write_run_contract_artifacts",
    "parse_paired_outcome_preference_warmstart_args",
    "run_paired_outcome_preference_warmstart",
]
