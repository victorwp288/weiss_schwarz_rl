from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Sequence
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
        description="Apply targeted replay trajectory behavior cloning as an auxiliary warmstart checkpoint"
    )
    parser.add_argument("--stack-config", type=Path, required=True, help="Training stack config")
    parser.add_argument("--dataset", type=Path, required=True, help="Replay trajectory BC .npz dataset")
    parser.add_argument("--init-from-checkpoint", type=Path, required=True, help="Checkpoint to warmstart from")
    parser.add_argument(
        "--output-run-dir", type=Path, required=True, help="Run directory for the warmstarted checkpoint"
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--epochs", type=int, default=2, help="Dataset passes")
    parser.add_argument("--batch-episodes", type=int, default=8, help="Episode columns per auxiliary update")
    parser.add_argument("--seed", type=int, default=20260516, help="Dataset shuffle seed")
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Force mixed precision for the auxiliary updates"
    )
    parser.add_argument("--teacher-family-coef", type=float, default=0.05)
    parser.add_argument("--teacher-slot-coef", type=float, default=0.05)
    parser.add_argument("--teacher-move-source-coef", type=float, default=0.02)
    parser.add_argument("--teacher-attack-type-coef", type=float, default=0.02)
    parser.add_argument("--teacher-action-coef", type=float, default=0.20)
    parser.add_argument("--teacher-same-family-action-coef", type=float, default=0.60)
    parser.add_argument("--teacher-action-margin-coef", type=float, default=0.0)
    parser.add_argument("--teacher-action-margin", type=float, default=0.5)
    parser.add_argument("--teacher-same-family-action-margin-coef", type=float, default=0.10)
    parser.add_argument("--teacher-same-family-action-margin", type=float, default=0.5)
    parser.add_argument(
        "--exact-action-family",
        action="append",
        default=None,
        help="Restrict exact-action auxiliary losses to this family. Repeat for multiple.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_episodes <= 0:
        parser.error("--batch-episodes must be positive")

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
    if bool(args.mixed_precision):
        training_config.mixed_precision = True
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
    learner.set_teacher_aux_coefs(
        family=float(args.teacher_family_coef),
        slot=float(args.teacher_slot_coef),
        move_source=float(args.teacher_move_source_coef),
        attack_type=float(args.teacher_attack_type_coef),
        action=float(args.teacher_action_coef),
        same_family_action=float(args.teacher_same_family_action_coef),
        action_margin=float(args.teacher_action_margin_coef),
        action_margin_value=float(args.teacher_action_margin),
        same_family_action_margin=float(args.teacher_same_family_action_margin_coef),
        same_family_action_margin_value=float(args.teacher_same_family_action_margin),
        exact_action_families=tuple(args.exact_action_family or ()),
        public_heuristic=0.0,
        public_nonpass_over_pass=0.0,
    )
    learner.teacher_aux_mode = "warmstart_only"

    rng = np.random.default_rng(int(args.seed))
    start_time = time.time()
    latest_metrics: dict[str, float] = {}
    aux_steps = 0
    for epoch in range(int(args.epochs)):
        order = rng.permutation(dataset.episode_count)
        for batch_start in range(0, dataset.episode_count, int(args.batch_episodes)):
            episode_indices = order[batch_start : batch_start + int(args.batch_episodes)].astype(np.int64).tolist()
            hidden = _initial_hidden_state(learner.model, batch_size=len(episode_indices), device=device)
            batch = replay_trajectory_bc_batch(
                dataset,
                episode_indices=episode_indices,
                initial_hidden_state=hidden,
            )
            latest_metrics = learner.auxiliary_update(batch)
            aux_steps += 1
            latest_metrics.update(
                {
                    "trajectory_bc_phase": 1.0,
                    "trajectory_bc_epoch": float(epoch + 1),
                    "trajectory_bc_aux_step": float(aux_steps),
                    "trajectory_bc_batch_episodes": float(len(episode_indices)),
                    "trajectory_bc_dataset_train_rows": float(dataset.metadata["train_rows"]),
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
        checkpoint_path = paths.checkpoints_dir / f"checkpoint_{int(learner.update_count):06d}_trajectory_bc.pt"
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
    snapshot_payload = _publish_trajectory_bc_snapshot(
        output_run_dir=output_layout.run_dir,
        checkpoint_path=checkpoint_path,
        update_count=int(learner.update_count),
    )
    summary = {
        "format": "trajectory_bc_warmstart_summary_v1",
        "dataset": args.dataset.resolve().as_posix(),
        "init_from_checkpoint": args.init_from_checkpoint.resolve().as_posix(),
        "output_run_dir": output_layout.run_dir.resolve().as_posix(),
        "checkpoint_path": checkpoint_path.resolve().as_posix(),
        "latest_checkpoint_path": paths.latest_checkpoint_path.resolve().as_posix(),
        "aux_steps": aux_steps,
        "epochs": int(args.epochs),
        "batch_episodes": int(args.batch_episodes),
        "dataset_metadata": dataset.metadata,
        "latest_metrics": latest_metrics,
        "checkpoint_tracker": tracker,
        "snapshot": snapshot_payload,
    }
    summary_path = output_layout.diagnostics_dir / "trajectory_bc_warmstart_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Trajectory BC warmstart wrote {checkpoint_path} after {aux_steps} auxiliary steps; "
        f"latest alias is {paths.latest_checkpoint_path}; summary written to {summary_path}"
    )
    return 0


def _initial_hidden_state(model: Any, *, batch_size: int, device: torch.device) -> np.ndarray | None:
    if model is None or not hasattr(model, "initial_seat_hidden"):
        return None
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
        "format": "trajectory_bc_warmstart_manifest_v1",
        "run_id256": hashlib.sha256(
            json.dumps(
                {
                    "kind": "trajectory_bc_warmstart",
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


def _publish_trajectory_bc_snapshot(
    *,
    output_run_dir: Path,
    checkpoint_path: Path,
    update_count: int,
) -> dict[str, Any]:
    policy_id = "trajectory_bc_latest"
    weights_relpath = snapshot_weights_relpath(policy_id)
    weights_path = output_run_dir / weights_relpath
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, weights_path)
    weights_sha256 = _sha256_file(weights_path)
    metadata_path = weights_path.parent / SNAPSHOT_METADATA_FILENAME
    metadata = {
        "format": "trajectory_bc_snapshot_meta_v1",
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
