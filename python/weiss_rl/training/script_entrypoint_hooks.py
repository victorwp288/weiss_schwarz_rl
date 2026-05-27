"""Script-level callback assembly for the path-based training entrypoint."""

from __future__ import annotations

from typing import Any


def ensure_current_checkpoint_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    training_paths = kwargs["training_paths"]
    learner = kwargs["learner"]
    stack = kwargs["stack"]
    device = kwargs["device"]
    spec_hash256 = kwargs.get("spec_hash256")
    algorithm = kwargs.get("algorithm")
    return api.ensure_current_checkpoint(
        training_paths=training_paths,
        learner=learner,
        write_checkpoint=lambda checkpoint_path: api._write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
        ),
    )


def periodic_dev_eval_opponents_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    return api.periodic_dev_eval_opponents(
        stack=kwargs["stack"],
        contract=kwargs["contract"],
        run_dir=kwargs["run_dir"],
        observation_dim=kwargs["observation_dim"],
        action_dim=kwargs["action_dim"],
        load_snapshot_model=api._load_snapshot_eval_model,
        build_heuristic_policy=api._build_heuristic_public_policy,
    )


def update_stall_monitor_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    return api._update_stall_monitor_impl(
        stack=kwargs["stack"],
        training_paths=kwargs["training_paths"],
        update_count=kwargs["update_count"],
        summary_payload=kwargs["summary_payload"],
    )


def maybe_rollback_to_best_checkpoint_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    stack = kwargs["stack"]
    learner = kwargs["learner"]
    device = kwargs["device"]
    spec_hash256 = kwargs["spec_hash256"]
    algorithm = kwargs["algorithm"]
    return api.maybe_rollback_to_best_checkpoint(
        stack=stack,
        training_paths=kwargs["training_paths"],
        run_dir=kwargs["artifacts"].run_dir,
        runtime=kwargs["runtime"],
        learner=learner,
        learner_model=kwargs["model"],
        latest_metrics=kwargs["latest_metrics"],
        dev_eval_summary=kwargs["dev_eval_summary"],
        last_rollback_update=kwargs["last_rollback_update"],
        restore_checkpoint=lambda checkpoint_path, *, restore_counters: api._restore_learner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
            restore_counters=restore_counters,
        ),
        write_checkpoint=lambda checkpoint_path: api._write_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
        ),
    )


def maybe_finalize_from_best_checkpoint_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    stack = kwargs["stack"]
    training_paths = kwargs["training_paths"]
    learner = kwargs["learner"]
    device = kwargs["device"]
    spec_hash256 = kwargs["spec_hash256"]
    algorithm = kwargs["algorithm"]
    return api.maybe_finalize_from_best_checkpoint(
        stack=stack,
        training_paths=training_paths,
        run_dir=kwargs["artifacts"].run_dir,
        runtime=kwargs["runtime"],
        learner=learner,
        latest_metrics=kwargs["latest_metrics"],
        dev_eval_summary=kwargs["dev_eval_summary"],
        restore_checkpoint=lambda checkpoint_path, *, restore_counters: api._restore_learner_from_checkpoint(
            checkpoint_path=checkpoint_path,
            learner=learner,
            stack=stack,
            device=device,
            expected_spec_hash256=spec_hash256,
            algorithm=algorithm,
            restore_counters=restore_counters,
        ),
        ensure_current_checkpoint=lambda: api._ensure_current_checkpoint(
            training_paths=training_paths,
            learner=learner,
            stack=stack,
            device=device,
            spec_hash256=spec_hash256,
            algorithm=algorithm,
        ),
    )


def run_periodic_dev_eval_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    return api.run_periodic_dev_eval(
        stack=kwargs["stack"],
        contract=kwargs["contract"],
        artifacts=kwargs["artifacts"],
        training_paths=kwargs["training_paths"],
        learner=kwargs["learner"],
        device=kwargs["device"],
        run_id256=kwargs["run_id256"],
        config_hash256=kwargs["config_hash256"],
        spec_hash256=kwargs["spec_hash256"],
        runner_cls=api._PeriodicDevEvalRunner,
        ensure_current_checkpoint_fn=api._ensure_current_checkpoint,
        current_focal_policy_id_fn=api._current_focal_policy_id,
        artifact_dir_name=kwargs.get("artifact_dir_name", "dev_eval"),
        artifact_scope=kwargs.get("artifact_scope", "periodic_dev_eval"),
        paired_seeds_override=kwargs.get("paired_seeds_override"),
        persist_summary=kwargs.get("persist_summary", True),
        update_stall_monitor_enabled=kwargs.get(
            "update_stall_monitor_enabled",
            kwargs.get("update_stall_monitor", True),
        ),
        spec_dimensions_fn=api._spec_dimensions,
        clone_cpu_eval_model_fn=api._clone_cpu_eval_model,
        periodic_dev_eval_opponents_fn=api._periodic_dev_eval_opponents,
        persist_summary_fn=api._persist_periodic_dev_eval_summary,
        update_stall_monitor_fn=api._update_stall_monitor,
        write_json_fn=api._write_json,
    )


def run_snapshot_promotion_gate_with_script_hooks(api: Any, **kwargs: Any) -> Any:
    return api.run_snapshot_promotion_gate(
        stack=kwargs["stack"],
        contract=kwargs["contract"],
        artifacts=kwargs["artifacts"],
        training_paths=kwargs["training_paths"],
        learner=kwargs["learner"],
        candidate_policy_id=kwargs["candidate_policy_id"],
        update_count=kwargs["update_count"],
        league_reference_update=kwargs["league_reference_update"],
        policy_version=kwargs["policy_version"],
        run_id256=kwargs["run_id256"],
        config_hash256=kwargs["config_hash256"],
        spec_hash256=kwargs["spec_hash256"],
        validate_periodic_dev_eval_contract_fn=api._validate_periodic_dev_eval_contract,
        resolve_promotion_anchor_policy_ids_fn=api._resolve_promotion_anchor_policy_ids,
        spec_dimensions_fn=api._spec_dimensions,
        snapshot_meta_by_policy_id_fn=api._snapshot_meta_by_policy_id,
        load_snapshot_eval_model_fn=api._load_snapshot_eval_model,
        build_heuristic_public_policy_fn=api._build_heuristic_public_policy,
        clone_cpu_eval_model_fn=api._clone_cpu_eval_model,
        promotion_gate_runner_cls=api._PromotionGateRunner,
        run_promotion_gate_fn=api.run_promotion_gate,
        promotion_gate_bootstrap_seed_fn=api._promotion_gate_bootstrap_seed,
        save_snapshot_registry_with_retention_fn=api._save_snapshot_registry_with_retention,
    )
