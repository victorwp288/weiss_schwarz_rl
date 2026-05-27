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
from weiss_rl.replay.trajectory_bc import (
    ReplayTrajectoryDataset,
    load_replay_trajectory_bc_dataset,
    replay_trajectory_bc_batch,
)
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
from weiss_rl.training.paired_swing_replay import (
    filter_paired_swing_conflict_rows,
    paired_swing_distinct_train_row_count,
)
from weiss_rl.training.paths import training_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply paired-swing replay as an auxiliary-only warmstart checkpoint")
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--init-from-checkpoint", type=Path, required=True)
    parser.add_argument("--output-run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--coef", type=float, default=0.08)
    parser.add_argument("--positive-action-source", choices=("actions", "teacher_action"), default="actions")
    parser.add_argument("--negative-action-source", choices=("actions", "teacher_action"), default="teacher_action")
    parser.add_argument("--loss-scope", choices=("row", "episode_mean", "label_mean"), default="row")
    parser.add_argument("--compare-to", choices=("negative", "top_other"), default="negative")
    parser.add_argument("--margin-retention-coef", type=float, default=0.0)
    parser.add_argument("--margin-retention-margin", type=float, default=0.0)
    parser.add_argument("--top-action-retention-coef", type=float, default=0.0)
    parser.add_argument("--top-action-retention-margin", type=float, default=0.0)
    parser.add_argument("--full-surface-retention-dataset", type=Path, default=None)
    parser.add_argument("--full-surface-retention-coef", type=float, default=0.0)
    parser.add_argument("--full-surface-retention-margin", type=float, default=0.0)
    parser.add_argument("--full-surface-retention-batch-episodes", type=int, default=0)
    parser.add_argument(
        "--full-surface-retention-mode",
        choices=("reference_top", "target_action"),
        default="reference_top",
    )
    parser.add_argument("--conflict-filter", choices=("none", "current_state", "history"), default="none")
    parser.add_argument("--allow-missing-context", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if int(args.epochs) <= 0:
        parser.error("--epochs must be positive")
    if int(args.batch_episodes) <= 0:
        parser.error("--batch-episodes must be positive")
    if float(args.margin) < 0.0:
        parser.error("--margin must be nonnegative")
    if float(args.coef) < 0.0:
        parser.error("--coef must be nonnegative")
    if float(args.margin_retention_coef) < 0.0:
        parser.error("--margin-retention-coef must be nonnegative")
    if float(args.margin_retention_margin) < 0.0:
        parser.error("--margin-retention-margin must be nonnegative")
    if float(args.top_action_retention_coef) < 0.0:
        parser.error("--top-action-retention-coef must be nonnegative")
    if float(args.top_action_retention_margin) < 0.0:
        parser.error("--top-action-retention-margin must be nonnegative")
    if float(args.full_surface_retention_coef) < 0.0:
        parser.error("--full-surface-retention-coef must be nonnegative")
    if float(args.full_surface_retention_margin) < 0.0:
        parser.error("--full-surface-retention-margin must be nonnegative")
    if int(args.full_surface_retention_batch_episodes) < 0:
        parser.error("--full-surface-retention-batch-episodes must be nonnegative")
    if float(args.full_surface_retention_coef) != 0.0 and args.full_surface_retention_dataset is None:
        parser.error("--full-surface-retention-dataset is required when --full-surface-retention-coef is nonzero")
    if str(args.positive_action_source) == str(args.negative_action_source):
        parser.error("--positive-action-source and --negative-action-source must differ")

    dataset = load_replay_trajectory_bc_dataset(args.dataset)
    conflict_filter_summary: dict[str, Any] | None = None
    if str(args.conflict_filter) != "none":
        dataset, conflict_filter_summary = filter_paired_swing_conflict_rows(
            dataset,
            mode=str(args.conflict_filter),
            positive_action_source=str(args.positive_action_source),
            negative_action_source=str(args.negative_action_source),
        )
    distinct_train_rows = paired_swing_distinct_train_row_count(
        dataset,
        positive_action_source=str(args.positive_action_source),
        negative_action_source=str(args.negative_action_source),
    )
    if int(dataset.metadata.get("train_rows", 0)) <= 0:
        parser.error("dataset has no trainable rows")
    if distinct_train_rows <= 0:
        parser.error("dataset has no paired-swing rows where positive and negative actions differ")
    retention_dataset: ReplayTrajectoryDataset | None = None
    if args.full_surface_retention_dataset is not None:
        retention_dataset = load_replay_trajectory_bc_dataset(args.full_surface_retention_dataset)
        if int(retention_dataset.metadata.get("train_rows", 0)) <= 0:
            parser.error("full-surface retention dataset has no trainable rows")
        retention_spec_hash = str(retention_dataset.metadata.get("spec_hash256") or "").strip()
        dataset_spec_hash = str(dataset.metadata.get("spec_hash256") or "").strip()
        if retention_spec_hash and dataset_spec_hash and retention_spec_hash != dataset_spec_hash:
            parser.error("full-surface retention dataset spec_hash256 does not match paired-swing dataset")

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

    all_context_indices = _opponent_context_indices_for_episodes(
        learner.model,
        dataset,
        episode_indices=list(range(dataset.episode_count)),
    )
    initial_context_episode_count = int(np.count_nonzero(all_context_indices))
    if initial_context_episode_count <= 0 and not bool(args.allow_missing_context):
        parser.error(
            "no nonzero opponent-context indices resolved; pass --allow-missing-context for unconditioned probes"
        )
    retention_initial_context_episode_count = 0
    if retention_dataset is not None:
        retention_context_indices = _opponent_context_indices_for_episodes(
            learner.model,
            retention_dataset,
            episode_indices=list(range(retention_dataset.episode_count)),
        )
        retention_initial_context_episode_count = int(np.count_nonzero(retention_context_indices))
        if retention_initial_context_episode_count <= 0 and not bool(args.allow_missing_context):
            parser.error(
                "no nonzero opponent-context indices resolved for full-surface retention dataset; "
                "pass --allow-missing-context for unconditioned probes"
            )

    rng = np.random.default_rng(int(args.seed))
    start_time = time.time()
    latest_metrics: dict[str, float] = {}
    aux_steps = 0
    gradient_steps = 0
    no_grad_steps = 0
    total_context_episodes = 0
    total_retention_context_episodes = 0
    retention_batch_episodes = (
        int(args.full_surface_retention_batch_episodes)
        if int(args.full_surface_retention_batch_episodes) > 0
        else int(args.batch_episodes)
    )
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
            retention_batch = None
            retention_step_episode_count = 0
            if retention_dataset is not None and float(args.full_surface_retention_coef) != 0.0:
                retention_episode_indices = _sample_episode_indices(
                    rng,
                    episode_count=retention_dataset.episode_count,
                    batch_episodes=retention_batch_episodes,
                )
                retention_step_episode_count = len(retention_episode_indices)
                retention_opponent_context_indices = _opponent_context_indices_for_episodes(
                    learner.model,
                    retention_dataset,
                    episode_indices=retention_episode_indices,
                )
                total_retention_context_episodes += int(np.count_nonzero(retention_opponent_context_indices))
                retention_hidden = _initial_hidden_state(
                    learner.model,
                    batch_size=len(retention_episode_indices),
                    device=device,
                    opponent_context_indices=retention_opponent_context_indices,
                )
                retention_batch = replay_trajectory_bc_batch(
                    retention_dataset,
                    episode_indices=retention_episode_indices,
                    initial_hidden_state=retention_hidden,
                    opponent_context_indices=retention_opponent_context_indices,
                )
            latest_metrics = learner.paired_swing_update(
                batch,
                margin=float(args.margin),
                coef=float(args.coef),
                positive_action_source=str(args.positive_action_source),
                negative_action_source=str(args.negative_action_source),
                loss_scope=str(args.loss_scope),
                compare_to=str(args.compare_to),
                margin_retention_coef=float(args.margin_retention_coef),
                margin_retention_margin=float(args.margin_retention_margin),
                top_action_retention_coef=float(args.top_action_retention_coef),
                top_action_retention_margin=float(args.top_action_retention_margin),
                full_surface_retention_batch=retention_batch,
                full_surface_top_action_retention_coef=float(args.full_surface_retention_coef),
                full_surface_top_action_retention_margin=float(args.full_surface_retention_margin),
                full_surface_top_action_retention_mode=str(args.full_surface_retention_mode),
            )
            aux_steps += 1
            if float(latest_metrics.get("optimizer_no_grad", 0.0)) > 0.0:
                no_grad_steps += 1
            else:
                gradient_steps += 1
            latest_metrics.update(
                {
                    "paired_swing_warmstart_phase": 1.0,
                    "paired_swing_warmstart_epoch": float(epoch + 1),
                    "paired_swing_warmstart_aux_step": float(aux_steps),
                    "paired_swing_warmstart_batch_episodes": float(len(episode_indices)),
                    "paired_swing_warmstart_dataset_train_rows": float(dataset.metadata["train_rows"]),
                    "paired_swing_warmstart_distinct_train_rows": float(distinct_train_rows),
                    "paired_swing_warmstart_context_episodes": float(total_context_episodes),
                    "paired_swing_warmstart_full_surface_retention_context_episodes": float(
                        total_retention_context_episodes
                    ),
                    "paired_swing_warmstart_full_surface_retention_batch_episodes": float(retention_step_episode_count),
                    "paired_swing_warmstart_gradient_steps": float(gradient_steps),
                    "paired_swing_warmstart_no_grad_steps": float(no_grad_steps),
                }
            )
            write_scalars_record(
                scalars_path=paths.scalars_path,
                learner=learner,
                metrics=latest_metrics,
                start_time=start_time,
            )

    if gradient_steps <= 0:
        parser.error("paired-swing warmstart produced no gradient steps")

    checkpoint_path = checkpoint_path_for_update(paths.checkpoints_dir, update_count=int(learner.update_count))
    if checkpoint_path.is_file():
        checkpoint_path = paths.checkpoints_dir / f"checkpoint_{int(learner.update_count):06d}_paired_swing.pt"
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
    snapshot_payload = _publish_paired_swing_snapshot(
        output_run_dir=output_layout.run_dir,
        checkpoint_path=checkpoint_path,
        update_count=int(learner.update_count),
    )
    summary = {
        "format": "paired_swing_warmstart_summary_v1",
        "dataset": args.dataset.resolve().as_posix(),
        "init_from_checkpoint": args.init_from_checkpoint.resolve().as_posix(),
        "output_run_dir": output_layout.run_dir.resolve().as_posix(),
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "latest_checkpoint_path": paths.latest_checkpoint_path.resolve().as_posix(),
        "aux_steps": aux_steps,
        "gradient_steps": gradient_steps,
        "no_grad_steps": no_grad_steps,
        "epochs": int(args.epochs),
        "batch_episodes": int(args.batch_episodes),
        "margin": float(args.margin),
        "coef": float(args.coef),
        "positive_action_source": str(args.positive_action_source),
        "negative_action_source": str(args.negative_action_source),
        "loss_scope": str(args.loss_scope),
        "compare_to": str(args.compare_to),
        "margin_retention_coef": float(args.margin_retention_coef),
        "margin_retention_margin": float(args.margin_retention_margin),
        "top_action_retention_coef": float(args.top_action_retention_coef),
        "top_action_retention_margin": float(args.top_action_retention_margin),
        "full_surface_retention_dataset": None
        if args.full_surface_retention_dataset is None
        else args.full_surface_retention_dataset.resolve().as_posix(),
        "full_surface_retention_coef": float(args.full_surface_retention_coef),
        "full_surface_retention_margin": float(args.full_surface_retention_margin),
        "full_surface_retention_batch_episodes": int(retention_batch_episodes),
        "full_surface_retention_mode": str(args.full_surface_retention_mode),
        "conflict_filter": str(args.conflict_filter),
        "conflict_filter_summary": conflict_filter_summary,
        "initial_context_episodes": initial_context_episode_count,
        "full_surface_retention_initial_context_episodes": retention_initial_context_episode_count,
        "context_episodes": total_context_episodes,
        "full_surface_retention_context_episodes": total_retention_context_episodes,
        "distinct_train_rows": distinct_train_rows,
        "full_surface_retention_dataset_metadata": None if retention_dataset is None else retention_dataset.metadata,
        "dataset_metadata": dataset.metadata,
        "latest_metrics": latest_metrics,
        "checkpoint_tracker": tracker,
        "snapshot": snapshot_payload,
    }
    summary_path = output_layout.diagnostics_dir / "paired_swing_warmstart_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Paired swing warmstart wrote {checkpoint_path} after {aux_steps} auxiliary steps; "
        f"latest alias is {paths.latest_checkpoint_path}; summary written to {summary_path}"
    )
    return 0


def _opponent_context_indices_for_episodes(
    model: Any,
    dataset: ReplayTrajectoryDataset,
    *,
    episode_indices: list[int],
) -> np.ndarray:
    opponent_ids = _source_opponent_policy_ids_by_episode(dataset)
    selected_policy_ids = [
        opponent_ids[int(index)] if int(index) < len(opponent_ids) else "" for index in episode_indices
    ]
    if model is None or not hasattr(model, "opponent_context_indices_for_policy_ids"):
        return np.zeros((len(episode_indices),), dtype=np.int64)
    return np.asarray(model.opponent_context_indices_for_policy_ids(selected_policy_ids), dtype=np.int64).reshape(-1)


def _source_opponent_policy_ids_by_episode(dataset: ReplayTrajectoryDataset) -> list[str]:
    bundles = dataset.metadata.get("selected_bundles")
    if not isinstance(bundles, list) or len(bundles) != int(dataset.episode_count):
        return []
    ids: list[str] = []
    for bundle in bundles:
        raw_id = bundle.get("source_opponent_policy_id") if isinstance(bundle, Mapping) else None
        ids.append(str(raw_id or "").strip())
    return ids


def _sample_episode_indices(
    rng: np.random.Generator,
    *,
    episode_count: int,
    batch_episodes: int,
) -> list[int]:
    count = int(episode_count)
    batch_size = int(batch_episodes)
    if count <= 0:
        raise ValueError("episode_count must be positive")
    if batch_size <= 0:
        raise ValueError("batch_episodes must be positive")
    replace = count < batch_size
    indices = rng.choice(count, size=batch_size, replace=replace)
    return [int(index) for index in np.asarray(indices, dtype=np.int64).reshape(-1).tolist()]


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
        "format": "paired_swing_warmstart_manifest_v1",
        "run_id256": hashlib.sha256(
            json.dumps(
                {
                    "kind": "paired_swing_warmstart",
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


def _publish_paired_swing_snapshot(*, output_run_dir: Path, checkpoint_path: Path, update_count: int) -> dict[str, Any]:
    policy_id = "paired_swing_latest"
    weights_relpath = snapshot_weights_relpath(policy_id)
    weights_path = output_run_dir / weights_relpath
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, weights_path)
    weights_sha256 = _sha256_file(weights_path)
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "paired_swing_snapshot_meta_v1",
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
