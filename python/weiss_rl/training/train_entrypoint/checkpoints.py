"""Checkpoint and learner hook glue for the training entrypoint facade."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class WriteCheckpointRequest:
    checkpoint_path: Path
    learner: Any
    stack: Any
    device: torch.device
    spec_hash256: str | None = None
    algorithm: str | None = None


@dataclass(frozen=True)
class BuildCheckpointRecordRequest:
    alias_name: str
    alias_path: Path
    source_checkpoint_path: Path
    artifacts: Any
    learner: Any
    metric_kind: str | None = None
    metric_value: float | None = None


@dataclass(frozen=True)
class PublishCheckpointAliasesRequest:
    stack: Any
    training_paths: Any
    artifacts: Any
    checkpoint_path: Path
    learner: Any
    latest_metrics: Mapping[str, float] | None
    dev_eval_summary: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class RestoreLearnerCheckpointRequest:
    checkpoint_path: Path
    learner: Any
    stack: Any
    device: torch.device
    expected_spec_hash256: str
    algorithm: str
    restore_counters: bool = True


@dataclass(frozen=True)
class InitializeLearnerCheckpointRequest:
    checkpoint_path: Path
    learner: Any
    device: torch.device
    expected_spec_hash256: str
    algorithm: str


@dataclass(frozen=True)
class BuildTrainingLearnerRequest:
    algorithm: str
    model: Any
    compiled_model: Any
    training_config: Any
    training_paths: Any
    pass_action_id: int
    checkpoint_interval_updates: int


@dataclass(frozen=True)
class StructuredWarmstartRequest:
    learner: Any
    runtime: Any
    algorithm: str
    training_config: Any
    rewards_config: Any
    training_paths: Any
    tensorboard_logger: Any
    start_time: float
    profile_timers: bool = False
    actor_torch_threads: int | None = None
    learner_torch_threads: int | None = None


@dataclass(frozen=True)
class BuildHeuristicPublicPolicyRequest:
    spec_bundle: Mapping[str, object]
    scoring_profile: str


@dataclass(frozen=True)
class ImportNoLeagueBaselineAnchorRequest:
    training_paths: Any
    run_dir: Path
    baseline_run_dir: Path
    expected_model_state_dict: dict[str, Any]
    expected_config_canonical: dict[str, Any] | None
    expected_spec_hash256: str | None


@dataclass(frozen=True)
class ValidateSeedSnapshotImportContractRequest:
    source_run_dir: Path
    payload: dict[str, Any]
    expected_model_state_dict: dict[str, Any]
    expected_config_canonical: dict[str, Any] | None
    expected_spec_hash256: str | None


@dataclass(frozen=True)
class SeedSnapshotPolicyIdRequest:
    source_run_dir: Path
    source_policy_id: str


@dataclass(frozen=True)
class ImportSeedSnapshotPoolRequest:
    stack: Any
    training_paths: Any
    run_dir: Path
    seed_snapshot_run_dir: Path
    expected_model_state_dict: dict[str, Any]
    expected_config_canonical: dict[str, Any] | None
    expected_spec_hash256: str | None


@dataclass(frozen=True)
class EnsureNoLeagueBaselineAnchorRequest:
    stack: Any
    training_paths: Any
    run_dir: Path
    learner: Any
    device: torch.device
    config_hash256: str
    spec_hash256: str | None = None
    baseline_run_dir: Path | None = None
    permit_current_run_alias: bool = False
    source_checkpoint_path: Path | None = None
    update: int | None = None


@dataclass(frozen=True)
class LoadSnapshotEvalModelRequest:
    run_dir: Path
    snapshot_path: str
    observation_dim: int
    action_dim: int
    stack: Any
    observation_spec: dict[str, Any] | None = None
    spec_bundle: dict[str, Any] | None = None


def write_checkpoint_with_entrypoint_hooks(api: Any, request: WriteCheckpointRequest) -> dict[str, Any]:
    return api.write_minimal_train_checkpoint(
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        device=request.device,
        config_hash256=api.compute_config_hash256(request.stack),
        spec_hash256=request.spec_hash256,
        algorithm=request.algorithm,
        recurrent_core=getattr(request.stack.config.model, "recurrent_core", None),
        guidance_payload=api._model_guidance_payload(request.learner.model),
    )


def build_checkpoint_record_with_entrypoint_hooks(api: Any, request: BuildCheckpointRecordRequest) -> dict[str, Any]:
    return api.build_checkpoint_record(
        alias_name=request.alias_name,
        alias_path=request.alias_path,
        source_checkpoint_path=request.source_checkpoint_path,
        run_dir=request.artifacts.run_dir,
        learner=request.learner,
        metric_kind=request.metric_kind,
        metric_value=request.metric_value,
    )


def publish_checkpoint_aliases_with_entrypoint_hooks(
    api: Any,
    request: PublishCheckpointAliasesRequest,
) -> dict[str, Any]:
    return api.publish_checkpoint_aliases(
        stack=request.stack,
        training_paths=request.training_paths,
        run_dir=request.artifacts.run_dir,
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        latest_metrics=request.latest_metrics,
        dev_eval_summary=request.dev_eval_summary,
    )


def restore_learner_from_checkpoint_with_entrypoint_hooks(
    api: Any,
    request: RestoreLearnerCheckpointRequest,
) -> Any:
    allow_config_mismatch = os.environ.get("WEISS_RL_ALLOW_RESUME_CONFIG_MISMATCH", "").strip() == "1"
    return api.restore_minimal_train_checkpoint(
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        device=request.device,
        expected_config_hash=api.compute_config_hash256(request.stack),
        expected_spec_hash256=request.expected_spec_hash256,
        algorithm=request.algorithm,
        restore_model_guidance=api.restore_model_guidance_from_payload,
        allow_config_mismatch=allow_config_mismatch,
        restore_counters=request.restore_counters,
    )


def initialize_learner_from_checkpoint_with_entrypoint_hooks(
    api: Any,
    request: InitializeLearnerCheckpointRequest,
) -> Any:
    return api.initialize_model_from_checkpoint(
        checkpoint_path=request.checkpoint_path,
        learner=request.learner,
        device=request.device,
        expected_spec_hash256=request.expected_spec_hash256,
        algorithm=request.algorithm,
        restore_model_guidance=api.restore_model_guidance_from_payload,
    )


def build_training_learner_with_entrypoint_hooks(api: Any, request: BuildTrainingLearnerRequest) -> Any:
    return api.build_training_learner(
        algorithm=request.algorithm,
        model=request.model,
        compiled_model=request.compiled_model,
        training_config=request.training_config,
        training_paths=request.training_paths,
        pass_action_id=request.pass_action_id,
        checkpoint_interval_updates=request.checkpoint_interval_updates,
    )


def run_structured_warmstart_with_entrypoint_hooks(api: Any, request: StructuredWarmstartRequest) -> dict[str, float]:
    return api.run_structured_warmstart(
        learner=request.learner,
        runtime=request.runtime,
        algorithm=request.algorithm,
        training_config=request.training_config,
        rewards_config=request.rewards_config,
        training_paths=request.training_paths,
        tensorboard_logger=request.tensorboard_logger,
        start_time=request.start_time,
        profile_timers=request.profile_timers,
        actor_torch_threads=request.actor_torch_threads,
        learner_torch_threads=request.learner_torch_threads,
    )


def build_heuristic_public_policy_with_entrypoint_hooks(
    api: Any,
    request: BuildHeuristicPublicPolicyRequest,
) -> Any:
    return api.build_heuristic_public_policy(
        request.spec_bundle,
        scoring_profile=request.scoring_profile,
        policy_cls=api.HeuristicPublicPolicy,
    )


def import_noleague_baseline_anchor_with_entrypoint_hooks(
    api: Any,
    request: ImportNoLeagueBaselineAnchorRequest,
) -> tuple[Path, str, int]:
    source_run_dir = Path(request.baseline_run_dir).resolve()
    source_snapshot = api._find_noleague_baseline_snapshot(source_run_dir)
    if source_snapshot is None:
        raise FileNotFoundError(
            "Could not resolve the canonical B1 no-league baseline snapshot in "
            f"{source_run_dir}. Run a dedicated baseline_noleague training job first."
        )

    source_weights_path = source_run_dir / source_snapshot.path
    if not source_weights_path.is_file():
        raise FileNotFoundError(f"Resolved B1 baseline snapshot is missing its weights artifact: {source_weights_path}")

    payload = torch.load(source_weights_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Imported B1 baseline weights payload must be a dict: {source_weights_path}")
    api._validate_imported_snapshot_contract(
        source_run_dir=source_run_dir,
        source_policy_id=source_snapshot.policy_id,
        payload=payload,
        expected_model_state_dict=request.expected_model_state_dict,
        expected_config_canonical=request.expected_config_canonical,
        expected_spec_hash256=request.expected_spec_hash256,
    )
    weights_path, weights_sha256 = api.write_imported_snapshot_artifact(
        snapshots_dir=request.training_paths.snapshots_dir,
        run_dir=request.run_dir,
        source_payload=payload,
        source_run_dir=source_run_dir,
        source_policy_id=source_snapshot.policy_id,
        source_snapshot_path=source_snapshot.path,
        target_policy_id=api._PROMOTION_GATE_NOLEAGUE_BASELINE_POLICY_ID,
        update=int(source_snapshot.update),
        metadata_format="imported_train_snapshot_metadata_v1",
    )
    return weights_path, weights_sha256, int(source_snapshot.update)


def validate_seed_snapshot_import_contract_with_entrypoint_hooks(
    api: Any,
    request: ValidateSeedSnapshotImportContractRequest,
) -> None:
    api.validate_seed_snapshot_import_contract(
        source_run_dir=request.source_run_dir,
        payload=request.payload,
        expected_model_state_dict=request.expected_model_state_dict,
        expected_config_canonical=request.expected_config_canonical,
        expected_spec_hash256=request.expected_spec_hash256,
    )


def seed_snapshot_policy_id_with_entrypoint_hooks(api: Any, request: SeedSnapshotPolicyIdRequest) -> str:
    return api.seed_snapshot_policy_id(
        source_run_dir=request.source_run_dir,
        source_policy_id=request.source_policy_id,
    )


def import_seed_snapshot_pool_with_entrypoint_hooks(api: Any, request: ImportSeedSnapshotPoolRequest) -> list[str]:
    return api.import_seed_snapshot_pool(
        stack=request.stack,
        training_paths=request.training_paths,
        run_dir=request.run_dir,
        seed_snapshot_run_dir=request.seed_snapshot_run_dir,
        expected_model_state_dict=request.expected_model_state_dict,
        expected_config_canonical=request.expected_config_canonical,
        expected_spec_hash256=request.expected_spec_hash256,
    )


def ensure_noleague_baseline_anchor_with_entrypoint_hooks(
    api: Any,
    request: EnsureNoLeagueBaselineAnchorRequest,
) -> str | None:
    return api.ensure_noleague_baseline_anchor(
        stack=request.stack,
        training_paths=request.training_paths,
        run_dir=request.run_dir,
        learner=request.learner,
        device=request.device,
        config_hash256=request.config_hash256,
        spec_hash256=request.spec_hash256,
        baseline_run_dir=request.baseline_run_dir,
        permit_current_run_alias=request.permit_current_run_alias,
        source_checkpoint_path=request.source_checkpoint_path,
        update=request.update,
        write_checkpoint_fn=api._write_checkpoint,
        import_noleague_baseline_anchor_fn=api._import_noleague_baseline_anchor,
        model_guidance_payload_fn=api._model_guidance_payload,
        write_snapshot_artifact_fn=api._write_snapshot_artifact,
        experiment_role_fn=api._experiment_role,
    )


def load_snapshot_eval_model_with_entrypoint_hooks(api: Any, request: LoadSnapshotEvalModelRequest) -> Any:
    return api.load_snapshot_eval_model(
        run_dir=request.run_dir,
        snapshot_path=request.snapshot_path,
        observation_dim=request.observation_dim,
        action_dim=request.action_dim,
        stack=request.stack,
        observation_spec=request.observation_spec,
        spec_bundle=request.spec_bundle,
    )


def install_checkpoint_io_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _write_checkpoint(
        *,
        checkpoint_path: Path,
        learner: Any,
        stack: Any,
        device: Any,
        spec_hash256: str | None = None,
        algorithm: str | None = None,
    ) -> dict[str, Any]:
        return write_checkpoint_with_entrypoint_hooks(
            entrypoint_api(),
            WriteCheckpointRequest(
                checkpoint_path=checkpoint_path,
                learner=learner,
                stack=stack,
                device=device,
                spec_hash256=spec_hash256,
                algorithm=algorithm,
            ),
        )

    def _build_checkpoint_record(
        *,
        alias_name: str,
        alias_path: Path,
        source_checkpoint_path: Path,
        artifacts: Any,
        learner: Any,
        metric_kind: str | None = None,
        metric_value: float | None = None,
    ) -> dict[str, Any]:
        return build_checkpoint_record_with_entrypoint_hooks(
            entrypoint_api(),
            BuildCheckpointRecordRequest(
                alias_name=alias_name,
                alias_path=alias_path,
                source_checkpoint_path=source_checkpoint_path,
                artifacts=artifacts,
                learner=learner,
                metric_kind=metric_kind,
                metric_value=metric_value,
            ),
        )

    def _publish_checkpoint_aliases(
        *,
        stack: Any,
        training_paths: Any,
        artifacts: Any,
        checkpoint_path: Path,
        learner: Any,
        latest_metrics: Mapping[str, float] | None,
        dev_eval_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return publish_checkpoint_aliases_with_entrypoint_hooks(
            entrypoint_api(),
            PublishCheckpointAliasesRequest(
                stack=stack,
                training_paths=training_paths,
                artifacts=artifacts,
                checkpoint_path=checkpoint_path,
                learner=learner,
                latest_metrics=latest_metrics,
                dev_eval_summary=dev_eval_summary,
            ),
        )

    namespace.update(
        {
            "_write_checkpoint": _write_checkpoint,
            "_build_checkpoint_record": _build_checkpoint_record,
            "_publish_checkpoint_aliases": _publish_checkpoint_aliases,
        }
    )


def install_learner_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    def _restore_learner_from_checkpoint(
        *,
        checkpoint_path: Path,
        learner: Any,
        stack: Any,
        device: Any,
        expected_spec_hash256: str,
        algorithm: str,
        restore_counters: bool = True,
    ) -> Any:
        return restore_learner_from_checkpoint_with_entrypoint_hooks(
            entrypoint_api(),
            RestoreLearnerCheckpointRequest(
                checkpoint_path=checkpoint_path,
                learner=learner,
                stack=stack,
                device=device,
                expected_spec_hash256=expected_spec_hash256,
                algorithm=algorithm,
                restore_counters=restore_counters,
            ),
        )

    def _initialize_learner_from_checkpoint(
        *,
        checkpoint_path: Path,
        learner: Any,
        device: Any,
        expected_spec_hash256: str,
        algorithm: str,
    ) -> Any:
        return initialize_learner_from_checkpoint_with_entrypoint_hooks(
            entrypoint_api(),
            InitializeLearnerCheckpointRequest(
                checkpoint_path=checkpoint_path,
                learner=learner,
                device=device,
                expected_spec_hash256=expected_spec_hash256,
                algorithm=algorithm,
            ),
        )

    def _build_training_learner(
        *,
        algorithm: str,
        model: Any,
        compiled_model: Any,
        training_config: Any,
        training_paths: Any,
        pass_action_id: int,
        checkpoint_interval_updates: int,
    ) -> Any:
        return build_training_learner_with_entrypoint_hooks(
            entrypoint_api(),
            BuildTrainingLearnerRequest(
                algorithm=algorithm,
                model=model,
                compiled_model=compiled_model,
                training_config=training_config,
                training_paths=training_paths,
                pass_action_id=pass_action_id,
                checkpoint_interval_updates=checkpoint_interval_updates,
            ),
        )

    def _run_structured_warmstart(
        *,
        learner: Any,
        runtime: Any,
        algorithm: str,
        training_config: Any,
        rewards_config: Any,
        training_paths: Any,
        tensorboard_logger: Any | None,
        start_time: float,
        profile_timers: bool = False,
        actor_torch_threads: int | None = None,
        learner_torch_threads: int | None = None,
    ) -> dict[str, float]:
        return run_structured_warmstart_with_entrypoint_hooks(
            entrypoint_api(),
            StructuredWarmstartRequest(
                learner=learner,
                runtime=runtime,
                algorithm=algorithm,
                training_config=training_config,
                rewards_config=rewards_config,
                training_paths=training_paths,
                tensorboard_logger=tensorboard_logger,
                start_time=start_time,
                profile_timers=profile_timers,
                actor_torch_threads=actor_torch_threads,
                learner_torch_threads=learner_torch_threads,
            ),
        )

    namespace.update(
        {
            "_restore_learner_from_checkpoint": _restore_learner_from_checkpoint,
            "_initialize_learner_from_checkpoint": _initialize_learner_from_checkpoint,
            "_build_training_learner": _build_training_learner,
            "_run_structured_warmstart": _run_structured_warmstart,
        }
    )


def install_checkpoint_wrappers(
    namespace: MutableMapping[str, Any],
    *,
    entrypoint_api: Callable[[], Any],
) -> None:
    from weiss_rl.training.train_entrypoint.snapshots import install_snapshot_wrappers

    install_checkpoint_io_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_learner_wrappers(namespace, entrypoint_api=entrypoint_api)
    install_snapshot_wrappers(namespace, entrypoint_api=entrypoint_api)


__all__ = [
    "BuildCheckpointRecordRequest",
    "BuildHeuristicPublicPolicyRequest",
    "BuildTrainingLearnerRequest",
    "EnsureNoLeagueBaselineAnchorRequest",
    "ImportNoLeagueBaselineAnchorRequest",
    "ImportSeedSnapshotPoolRequest",
    "InitializeLearnerCheckpointRequest",
    "LoadSnapshotEvalModelRequest",
    "PublishCheckpointAliasesRequest",
    "RestoreLearnerCheckpointRequest",
    "SeedSnapshotPolicyIdRequest",
    "StructuredWarmstartRequest",
    "ValidateSeedSnapshotImportContractRequest",
    "WriteCheckpointRequest",
    "build_checkpoint_record_with_entrypoint_hooks",
    "build_heuristic_public_policy_with_entrypoint_hooks",
    "build_training_learner_with_entrypoint_hooks",
    "ensure_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_noleague_baseline_anchor_with_entrypoint_hooks",
    "import_seed_snapshot_pool_with_entrypoint_hooks",
    "install_checkpoint_io_wrappers",
    "install_checkpoint_wrappers",
    "install_learner_wrappers",
    "initialize_learner_from_checkpoint_with_entrypoint_hooks",
    "load_snapshot_eval_model_with_entrypoint_hooks",
    "publish_checkpoint_aliases_with_entrypoint_hooks",
    "restore_learner_from_checkpoint_with_entrypoint_hooks",
    "run_structured_warmstart_with_entrypoint_hooks",
    "seed_snapshot_policy_id_with_entrypoint_hooks",
    "validate_seed_snapshot_import_contract_with_entrypoint_hooks",
    "write_checkpoint_with_entrypoint_hooks",
]
