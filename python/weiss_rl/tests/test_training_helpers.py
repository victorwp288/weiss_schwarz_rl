from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from weiss_rl.config import load_stack_config
from weiss_rl.eval.harness import EvalGameRecord, GameResult
from weiss_rl.league import PromotionGateAnchor
from weiss_rl.league.registry import SnapshotMeta, SnapshotRegistry, snapshot_weights_relpath
from weiss_rl.learners.metric_projection import CUSTOM_LOG_METRIC_KEYS, build_custom_log_metrics
from weiss_rl.learners.vtrace import VtraceMetrics
from weiss_rl.tests._config_paths import canonical_stack_config_path
from weiss_rl.training.anchor_resolution import (
    promotion_anchor_policy_id_candidates,
    resolve_symbolic_promotion_anchor_policy_id,
    true_local_recent_snapshot_ids,
)
from weiss_rl.training.bootstrap import (
    apply_training_flag_overrides,
    build_train_arg_parser,
    expected_sha256,
    format_cuda_unavailable_fallback_message,
    format_loaded_stack_config_message,
    format_manifest_scaffold_only_message,
    format_manifest_written_message,
    format_public_demo_disclaimer_message,
    format_public_demo_staged_message,
    format_resume_run_dir_message,
    format_spec_bundle_status_message,
    format_tensorboard_disabled_message,
    manifest_scaffold_only_reason,
    normalize_sha256,
    require_matching_hash,
    require_positive_int,
    require_positive_optional_float,
    resolve_device,
    resolve_run_label,
    resolve_runtime_profile,
    resolve_seed,
    runtime_training_prerequisite_failure,
)
from weiss_rl.training.checkpoint_guard import (
    checkpoint_guard_rollback_plan,
    format_checkpoint_guard_final_selection_message,
    format_checkpoint_guard_rollback_message,
)
from weiss_rl.training.checkpoints import (
    CHECKPOINT_TRACKER_FORMAT,
    build_checkpoint_record,
    build_secondary_checkpoint_record,
    checkpoint_secondary_records,
    load_checkpoint_tracker,
    publish_checkpoint_aliases,
    write_checkpoint_tracker,
)
from weiss_rl.training.confirmatory_eval import build_confirmatory_dev_eval_plan, format_confirmatory_dev_eval_message
from weiss_rl.training.curriculum_guards import (
    apply_stall_monitor_to_dev_eval_summary,
    early_cutoff_metric_updates,
    format_early_cutoff_triggered_message,
    format_stall_monitor_warning,
    format_training_stopped_by_early_cutoff_message,
    update_early_cutoff,
    update_stall_monitor,
)
from weiss_rl.training.dev_eval_metrics import (
    b2_recent_scores_from_persisted_summaries,
    build_b2_warning_flags,
    extract_anchor_score,
)
from weiss_rl.training.eval_artifacts import (
    b2_disagreement_audit_requests_path,
    build_periodic_dev_eval_checkpoint_summary,
    build_periodic_dev_eval_matchup_context_payload,
    build_periodic_dev_eval_matchup_runtime_payload,
    build_periodic_dev_eval_seed_usage_payload,
    checkpoint_guard_log_path,
    collate_periodic_dev_eval_seed_block_matchup,
    format_b2_disagreement_audit_request_message,
    format_periodic_dev_eval_console_message,
    format_periodic_dev_eval_scheduled_message,
    group_periodic_dev_eval_seed_block_results,
    maybe_log_structured_mainmove_guard,
    maybe_request_b2_disagreement_audit,
    periodic_dev_eval_fast_screens_path,
    periodic_dev_eval_matchup_dir,
    persist_periodic_dev_eval_fast_screen,
    persist_periodic_dev_eval_result,
    persist_periodic_dev_eval_summary,
    promotion_gate_records_by_anchor_index,
    sum_periodic_dev_eval_counter_payloads,
)
from weiss_rl.training.eval_model_cache import get_cached_eval_model, remember_eval_model
from weiss_rl.training.eval_schedule import (
    PeriodicDevEvalOpponentSpec,
    PromotionGateSeedBlockJob,
    build_async_periodic_dev_eval_request,
    build_async_promotion_gate_request,
    build_periodic_dev_eval_seed_block_jobs,
    build_promotion_gate_seed_block_jobs,
    format_league_eval_warmup_gate_message,
    league_eval_warmup_gate_status,
    periodic_dev_eval_anchor_weight_map,
    periodic_dev_eval_duplicate_policy_ids,
    periodic_dev_eval_schedule,
    periodic_dev_eval_schedule_for_seed_items,
    resolved_periodic_dev_eval_worker_devices,
    shard_periodic_dev_eval_seed_block_jobs,
    shard_promotion_gate_seed_block_jobs,
    should_defer_noleague_baseline_alias_refresh,
)
from weiss_rl.training.eval_seeds import (
    expand_periodic_dev_eval_paired_seeds,
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_rng_seed,
    promotion_gate_bootstrap_seed,
    promotion_gate_rng_seed,
)
from weiss_rl.training.guidance_schedules import (
    apply_guidance_schedule_for_next_update,
    counterfactual_positive_coef_for_next_update,
    entropy_coef_for_next_update,
    format_attached_reference_policy_message,
    model_guidance_payload,
    raw_b1_distill_coef_for_next_update,
    reference_policy_top_action_bc_coef_for_next_update,
    restore_model_guidance_from_payload,
)
from weiss_rl.training.learner_setup import (
    format_compile_learner_forward_enabled_message,
    format_compile_learner_missing_trunk_hook_message,
    format_compile_learner_not_cuda_message,
    format_compile_learner_trunk_enabled_message,
    format_compile_learner_trunk_failed_message,
    format_trainable_main_residual_policy_enabled_message,
    maybe_compile_learner_model,
)
from weiss_rl.training.manifest_payloads import (
    evaluation_pinning,
    hardware_summary,
    manifest_actor_device_layout,
    training_controls_payload,
)
from weiss_rl.training.paths import build_training_paths, run_artifacts_from_existing_run_dir
from weiss_rl.training.profiling import (
    build_training_profiler,
    format_torch_profiler_trace_written_message,
    profile_block,
)
from weiss_rl.training.promotion_artifacts import (
    assemble_parallel_promotion_gate_result,
    build_parallel_promotion_gate_plan,
    build_promotion_gate_worker_payloads,
    format_optional_heuristic_public_anchors_skipped_message,
    format_promotion_gate_discarded_after_rollback_message,
    format_promotion_gate_missing_anchors_message,
    format_promotion_gate_skipped_eval_warmup_gate_message,
    format_promotion_gate_skipped_league_warmup_message,
    format_scheduled_async_promotion_gate_message,
    promotion_gate_policy_maps,
)
from weiss_rl.training.provenance import git_commit, load_json_object, manifest_source_path, start_nonce
from weiss_rl.training.session import (
    format_resume_config_hash_mismatch_warning,
    format_resumed_learner_state_message,
    format_seeded_checkpoint_best_alias_message,
    format_seeded_resume_dev_eval_summary_message,
    format_structured_profiling_enabled_message,
    format_training_completed_message,
    format_wall_clock_budget_reached_message,
    wall_clock_budget_metric_updates,
    wall_clock_budget_reached,
    wall_clock_budget_seconds,
)
from weiss_rl.training.snapshot_artifacts import (
    apply_promotion_gate_payload,
    apply_promotion_gate_result,
    format_promotion_gate_registry_update_message,
)
from weiss_rl.training.snapshot_imports import (
    config_marks_noleague_baseline,
    ensure_noleague_baseline_anchor,
    seed_snapshot_policy_id,
    source_snapshot_is_resume_league_snapshot,
)


def test_checkpoint_tracker_helpers_normalize_missing_secondary(tmp_path):
    training_paths = SimpleNamespace(checkpoint_tracker_path=tmp_path / "checkpoint_tracker.json")

    assert load_checkpoint_tracker(training_paths) == {
        "format": CHECKPOINT_TRACKER_FORMAT,
        "latest": None,
        "best": None,
        "secondary": {},
    }

    write_checkpoint_tracker(training_paths, {"latest": {"alias": "latest"}, "secondary": None})

    payload = json.loads(training_paths.checkpoint_tracker_path.read_text(encoding="utf-8"))
    assert payload["format"] == CHECKPOINT_TRACKER_FORMAT
    assert payload["latest"] == {"alias": "latest"}
    assert payload["secondary"] == {}
    assert load_checkpoint_tracker(training_paths)["best"] is None


def test_training_bootstrap_parser_preserves_public_flags_and_alias_warning(capsys):
    parser = build_train_arg_parser()

    args = parser.parse_args(
        [
            "--stack-config",
            "configs/local.yaml",
            "--run-id",
            "legacy_label",
            "--public-demo",
            "--runtime-mode",
            "train_async_fast",
            "--ddp-timeout-seconds",
            "2400",
            "--override",
            "training.optimizer.learning_rate=0.0001",
        ]
    )
    run_label = resolve_run_label(parser, args.run_label, args.run_id_alias)

    assert args.stack_config.as_posix() == "configs/local.yaml"
    assert args.public_demo is True
    assert args.runtime_mode == "train_async_fast"
    assert args.ddp_timeout_seconds == 2400
    assert args.config_override == ["training.optimizer.learning_rate=0.0001"]
    assert run_label == "legacy_label"
    assert "--run-id is deprecated" in capsys.readouterr().err


def test_training_bootstrap_numeric_validators_preserve_error_text():
    assert require_positive_int("--num-envs", "4") == 4
    assert require_positive_optional_float("--max-wall-clock-minutes", None) is None
    assert require_positive_optional_float("--max-wall-clock-minutes", "2.5") == pytest.approx(2.5)

    with pytest.raises(ValueError, match="--num-envs must be >= 1, got 0"):
        require_positive_int("--num-envs", 0)
    with pytest.raises(ValueError, match="--max-wall-clock-minutes must be a finite number > 0, got 0.0"):
        require_positive_optional_float("--max-wall-clock-minutes", 0.0)


def test_training_bootstrap_hash_validators_preserve_error_text():
    sha = "AB" * 32
    assert normalize_sha256(sha) == "ab" * 32
    assert normalize_sha256("abc") == ""
    assert expected_sha256("", flag_name="--config-hash") == ""
    assert expected_sha256(sha, flag_name="--config-hash") == "ab" * 32

    with pytest.raises(
        ValueError, match="--config-hash must be a 64-character lowercase or uppercase SHA-256 hex string"
    ):
        expected_sha256("not-a-hash", flag_name="--config-hash")
    require_matching_hash(flag_name="--config-hash", expected="", actual="anything")
    require_matching_hash(flag_name="--config-hash", expected="abc", actual="abc")
    with pytest.raises(RuntimeError, match="--config-hash mismatch: expected abc, observed def"):
        require_matching_hash(flag_name="--config-hash", expected="abc", actual="def")


def test_training_bootstrap_runtime_resolution_helpers_preserve_contracts(monkeypatch, capsys):
    empty_stack = SimpleNamespace(
        config=SimpleNamespace(
            environment=None,
            model=None,
            reproducibility=None,
            system=None,
            training=None,
        )
    )

    assert resolve_runtime_profile(empty_stack, " debug ") == "debug"
    assert resolve_runtime_profile(empty_stack, "") == "fast"
    assert resolve_seed(empty_stack, 123) == 123
    assert resolve_seed(empty_stack, None) == 7
    assert manifest_scaffold_only_reason(empty_stack) == "missing config blocks: environment, training, model"
    assert runtime_training_prerequisite_failure(empty_stack) is None
    assert format_cuda_unavailable_fallback_message() == (
        "Requested CUDA device is unavailable; falling back to cpu for the canonical single-node run."
    )

    system_stack = SimpleNamespace(
        config=SimpleNamespace(
            reproducibility=SimpleNamespace(seed_derivation=SimpleNamespace(base_seed64=456)),
            system=SimpleNamespace(profile=SimpleNamespace(local_iteration="fast_sim"), learner_device="cuda:7"),
        )
    )
    assert resolve_runtime_profile(system_stack, "") == "fast_sim"
    assert resolve_seed(system_stack, None) == 456
    monkeypatch.setattr("weiss_rl.training.bootstrap.torch.cuda.is_available", lambda: False)
    assert resolve_device(system_stack, "") == torch.device("cpu")
    assert format_cuda_unavailable_fallback_message() in capsys.readouterr().err

    complete_stack = SimpleNamespace(
        config=SimpleNamespace(
            environment=object(),
            model=object(),
            reproducibility=None,
            system=None,
            training=object(),
        )
    )

    def missing_import(name: str):
        assert name == "weiss_sim"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("weiss_rl.training.bootstrap.importlib.import_module", missing_import)
    assert runtime_training_prerequisite_failure(complete_stack) == (
        "weiss_sim is not importable in the active interpreter"
    )


def test_training_bootstrap_flag_overrides_preserve_profile_controls():
    stack = load_stack_config(canonical_stack_config_path())

    updated = apply_training_flag_overrides(
        stack,
        enable_profile_timers=True,
        enable_torch_profiler=True,
    )

    assert updated.config.training.profile_timers is True
    assert updated.config.training.torch_profiler is True
    assert stack.config.training.profile_timers is False
    assert stack.config.training.torch_profiler is False


def test_training_bootstrap_startup_messages_preserve_public_text(tmp_path):
    assert (
        format_spec_bundle_status_message(
            public_demo_enabled=False,
            compatibility_hash="compat123",
            spec_hash256="ab" * 32,
        )
        == f"Verified runtime spec bundle: compat=compat123 sha256={'ab' * 32}"
    )
    assert (
        format_spec_bundle_status_message(
            public_demo_enabled=True,
            compatibility_hash="compat123",
            spec_hash256="cd" * 32,
        )
        == f"Loaded synthetic public-demo spec bundle: compat=compat123 sha256={'cd' * 32}"
    )
    assert format_loaded_stack_config_message(3) == "Loaded stack config with 3 components"
    assert (
        format_manifest_written_message(tmp_path / "manifest.json") == f"Wrote manifest: {tmp_path / 'manifest.json'}"
    )
    assert format_resume_run_dir_message(tmp_path / "run") == f"Resuming existing run directory: {tmp_path / 'run'}"
    assert format_manifest_scaffold_only_message("missing config blocks: training") == (
        "Manifest scaffold only: no learner training or rollout collection was executed.",
        "Reason: missing config blocks: training.",
    )
    assert format_tensorboard_disabled_message(None) == "TensorBoard logging is disabled: SummaryWriter unavailable"
    assert (
        format_tensorboard_disabled_message("torch.utils.tensorboard is missing")
        == "TensorBoard logging is disabled: torch.utils.tensorboard is missing"
    )
    assert format_public_demo_staged_message(
        mode="toy_public_demo_v1",
        policy_count=3,
        catalog_path=tmp_path / "toy_catalog.json",
    ) == (
        "Staged public-demo toy catalog and policy bundle: "
        f"mode=toy_public_demo_v1 policy_count=3 catalog={tmp_path / 'toy_catalog.json'}"
    )
    assert format_public_demo_disclaimer_message() == (
        "Public demo mode is intentionally synthetic and demo-only. "
        "It does not execute simulator training or claim thesis-grade results."
    )


def test_training_paths_create_canonical_run_directories(tmp_path):
    paths = build_training_paths(tmp_path)
    artifacts = run_artifacts_from_existing_run_dir(tmp_path)

    assert paths.training_dir == tmp_path / "training"
    assert paths.checkpoints_dir == tmp_path / "training" / "checkpoints"
    assert paths.latest_checkpoint_path == paths.checkpoints_dir / "latest.pt"
    assert paths.best_checkpoint_path == paths.checkpoints_dir / "best.pt"
    assert paths.checkpoint_tracker_path == paths.checkpoints_dir / "checkpoint_tracker.json"
    assert paths.logs_dir.is_dir()
    assert paths.snapshots_dir.is_dir()
    assert artifacts.run_dir == tmp_path.resolve()
    assert artifacts.manifest_path == tmp_path.resolve() / "manifest.json"


def test_training_manifest_payload_helpers_preserve_schema_keys():
    payload = hardware_summary(
        learner_device=torch.device("cpu"),
        actor_device="cuda:0",
        actor_device_layout=("cuda:0", "cuda:1", "cuda:0"),
    )

    assert payload["learner_device"] == "cpu"
    assert payload["actor_device"] == "cuda:0"
    assert payload["actor_device_layout"] == "cuda:0,cuda:1,cuda:0"
    assert payload["actor_device_unique_count"] == 2
    assert {"platform", "python_version", "machine", "processor", "cpu_count"} <= set(payload)

    stack_without_eval = SimpleNamespace(config=SimpleNamespace(evaluation=None, system=None, training=None))
    assert evaluation_pinning(stack_without_eval) == {}
    assert (
        manifest_actor_device_layout(
            stack=stack_without_eval,
            num_envs=2,
            unroll_length=4,
            profile="fast",
            seed=7,
            pass_action_id=0,
            runtime_mode="train_ordered",
            learner_device=torch.device("cpu"),
        )
        is None
    )

    ddp_stack = SimpleNamespace(
        config=SimpleNamespace(
            evaluation=None,
            system=SimpleNamespace(
                actor_device="cuda:auto",
                learner_device="cuda:auto",
                actor_process_count=32,
                envs_per_actor=64,
                actor_queue_capacity_unrolls=256,
                collection_backend="process",
                actor_torch_threads=1,
            ),
            training=SimpleNamespace(batch_unrolls_per_update=64, max_queue_wait_ms=1000),
        )
    )
    ddp_topology = SimpleNamespace(
        actor_count=8,
        envs_per_actor=64,
        batch_unrolls_per_update=64,
        queue_capacity_unrolls=256,
        learner_gpu_count=4,
    )
    assert manifest_actor_device_layout(
        stack=ddp_stack,
        num_envs=512,
        unroll_length=64,
        profile="fast",
        seed=7,
        pass_action_id=0,
        runtime_mode="train_async_fast",
        learner_device=torch.device("cuda:0"),
        resolved_topology=ddp_topology,
        rank_local_actor_devices=True,
    ) == ("cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:0", "cuda:1", "cuda:2", "cuda:3")

    evaluation = SimpleNamespace(
        eval_device="cpu",
        eval_sampling_algorithm="pinned_cdf_pcg_v1",
        eval_inference_mode="inference_mode",
        seat_swap=True,
        legal_fingerprint_checks=SimpleNamespace(version="legal_fingerprint_v1", mismatch_policy="hard_fail"),
    )
    assert evaluation_pinning(SimpleNamespace(config=SimpleNamespace(evaluation=evaluation))) == {
        "eval_device": "cpu",
        "eval_sampling_algorithm": "pinned_cdf_pcg_v1",
        "eval_inference_mode": "inference_mode",
        "seat_swap": True,
        "legal_fingerprint_version": "legal_fingerprint_v1",
        "legal_fingerprint_mismatch_policy": "hard_fail",
    }
    training_config = SimpleNamespace(
        profile_timers=True,
        torch_profiler=False,
        structured_metrics_mode="cheap",
        teacher_aux_mode="sampled",
        fixed_opponent_backend="snapshot",
        heuristic_native_rollout_enabled=True,
        heuristic_native_rollout_profile="profile_a",
        heuristic_native_rollout_profiles=("profile_a", "profile_b"),
        heuristic_native_rollout_profile_mode="cycle",
    )
    base_controls = training_controls_payload(training_config)
    assert base_controls == {
        "profile_timers": True,
        "torch_profiler": False,
        "structured_metrics_mode": "cheap",
        "teacher_aux_mode": "sampled",
        "fixed_opponent_backend": "snapshot",
        "heuristic_native_rollout_enabled": True,
        "heuristic_native_rollout_profile": "profile_a",
        "heuristic_native_rollout_profiles": ["profile_a", "profile_b"],
        "heuristic_native_rollout_profile_mode": "cycle",
    }
    assert "max_wall_clock_minutes" not in base_controls
    assert training_controls_payload(
        training_config,
        max_wall_clock_minutes=1.5,
        include_wall_clock_budget=True,
    )["max_wall_clock_minutes"] == pytest.approx(1.5)
    assert (
        training_controls_payload(
            training_config,
            max_wall_clock_minutes=None,
            include_wall_clock_budget=True,
        )["max_wall_clock_minutes"]
        is None
    )


def test_training_provenance_helpers_preserve_manifest_path_and_json_contract(tmp_path, monkeypatch):
    inside = tmp_path / "runs" / "run_a" / "manifest.json"
    inside.parent.mkdir(parents=True)
    inside.write_text('{"ok": true}\n', encoding="utf-8")
    outside = tmp_path.parent / "outside_manifest.json"

    assert manifest_source_path(inside, root=tmp_path) == "runs/run_a/manifest.json"
    assert manifest_source_path(outside, root=tmp_path) == str(outside.resolve())
    assert load_json_object(inside, label="manifest") == {"ok": True}

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest JSON must contain an object at the top level"):
        load_json_object(bad_json, label="manifest")

    monkeypatch.setenv("WEISS_RL_GIT_COMMIT", "DEADBEEF" * 5)
    assert git_commit(repo_root=tmp_path) == "deadbeef" * 5
    assert 0 <= start_nonce() < (1 << 64)


def test_training_session_helpers_preserve_loop_event_contracts():
    assert wall_clock_budget_seconds(1.5) == pytest.approx(90.0)
    assert wall_clock_budget_seconds(None) is None
    assert wall_clock_budget_reached(start_time=10.0, max_wall_clock_seconds=5.0, now=15.0) is True
    assert wall_clock_budget_reached(start_time=10.0, max_wall_clock_seconds=5.0, now=14.9) is False
    assert wall_clock_budget_reached(start_time=10.0, max_wall_clock_seconds=None, now=99.0) is False
    assert wall_clock_budget_metric_updates(max_wall_clock_seconds=12.5, elapsed_seconds=13.25) == {
        "wall_clock_budget_reached": pytest.approx(1.0),
        "wall_clock_budget_seconds": pytest.approx(12.5),
        "wall_clock_budget_elapsed_seconds": pytest.approx(13.25),
    }
    assert format_wall_clock_budget_reached_message(elapsed_seconds=13.25, max_wall_clock_seconds=12.5) == (
        "Wall clock budget reached: elapsed=13.25s budget=12.50s"
    )
    assert format_training_completed_message(
        {
            "loss": 1.25,
            "policy_loss": 0.5,
            "value_loss": 0.125,
            "entropy": 0.0625,
        }
    ) == (
        "Completed canonical single-node training run: "
        "loss=1.250000 policy_loss=0.500000 value_loss=0.125000 entropy=0.062500"
    )
    assert (
        format_resumed_learner_state_message(
            checkpoint_path="training/checkpoints/best.pt",
            update_count=12,
            policy_version=14,
        )
        == "Resumed learner state: checkpoint=training/checkpoints/best.pt update=12 policy_version=14"
    )
    assert format_resume_config_hash_mismatch_warning(
        checkpoint_config_hash="abc123",
        current_config_hash="def456",
    ) == (
        "Warning: resuming checkpoint under a different config hash "
        "(checkpoint=abc123, current=def456). "
        "Use this only for explicit research continuations."
    )
    assert (
        format_seeded_checkpoint_best_alias_message(
            {
                "update_count": 12,
                "metric_value": 0.45678,
            }
        )
        == "Seeded checkpoint best alias from resumed dev-eval best: update=12 metric=0.4568"
    )
    assert format_seeded_resume_dev_eval_summary_message(update_count=12, aggregate_score=0.45678) == (
        "Seeded resume dev-eval summary: update=12 aggregate=0.4568"
    )
    assert format_structured_profiling_enabled_message(
        SimpleNamespace(
            profile_timers=True,
            torch_profiler=False,
            structured_metrics_mode="cheap",
            teacher_aux_mode="sampled",
            fixed_opponent_backend="snapshot",
        )
    ) == (
        "Structured profiling enabled: profile_timers=True torch_profiler=False "
        "structured_metrics_mode=cheap teacher_aux_mode=sampled fixed_opponent_backend=snapshot"
    )


def test_learner_setup_compile_helpers_preserve_console_contracts():
    assert format_compile_learner_not_cuda_message() == (
        "Learner compile note: compile_learner is enabled but the learner device is not CUDA; skipping torch.compile."
    )
    assert format_compile_learner_trunk_failed_message("RuntimeError('boom')") == (
        "Learner compile note: structured trunk compile failed; skipping torch.compile (RuntimeError('boom'))."
    )
    assert (
        format_compile_learner_trunk_enabled_message()
        == "Enabled torch.compile for the structured learner trunk (mode=reduce-overhead)."
    )
    assert format_compile_learner_missing_trunk_hook_message() == (
        "Learner compile note: structured legal scoring is enabled but no trunk compile hook exists; skipping torch.compile."
    )
    assert (
        format_compile_learner_forward_enabled_message()
        == "Enabled torch.compile for the learner forward path (mode=reduce-overhead)."
    )
    checkpoint_path = Path("runs") / "base" / "training" / "checkpoints" / "best.pt"
    assert format_trainable_main_residual_policy_enabled_message(
        checkpoint_path=checkpoint_path,
        alpha=0.125,
        hidden_dim=384,
        residual_mode="family_gate",
        initial_state_path_text="runs/residual.pt",
    ) == (
        f"Enabled trainable main residual policy: base={checkpoint_path} alpha=0.125 "
        "hidden_dim=384 mode=family_gate initial_state=runs/residual.pt"
    )
    assert format_trainable_main_residual_policy_enabled_message(
        checkpoint_path=checkpoint_path,
        alpha=0.5,
        hidden_dim=256,
        residual_mode="plain",
        initial_state_path_text="",
    ).endswith("initial_state=<zero>")

    emitted: list[str] = []
    model = torch.nn.Linear(2, 2)
    assert (
        maybe_compile_learner_model(
            model=model,
            training_config=SimpleNamespace(compile_learner=False),
            device=torch.device("cpu"),
            emit=emitted.append,
        )
        is None
    )
    assert emitted == []

    assert (
        maybe_compile_learner_model(
            model=model,
            training_config=SimpleNamespace(compile_learner=True),
            device=torch.device("cpu"),
            emit=emitted.append,
        )
        is None
    )
    assert emitted[-1] == format_compile_learner_not_cuda_message()

    structured_without_hook = torch.nn.Linear(2, 2)
    structured_without_hook.supports_legal_candidate_scoring = True
    assert (
        maybe_compile_learner_model(
            model=structured_without_hook,
            training_config=SimpleNamespace(compile_learner=True),
            device=torch.device("cuda"),
            emit=emitted.append,
        )
        is None
    )
    assert emitted[-1] == format_compile_learner_missing_trunk_hook_message()

    class TrunkCompileModel(torch.nn.Module):
        supports_legal_candidate_scoring = True

        def __init__(self, *, fail: bool = False) -> None:
            super().__init__()
            self.fail = fail
            self.modes: list[str] = []

        def enable_trunk_compile(self, *, mode: str) -> None:
            self.modes.append(mode)
            if self.fail:
                raise RuntimeError("boom")

    trunk_model = TrunkCompileModel()
    assert (
        maybe_compile_learner_model(
            model=trunk_model,
            training_config=SimpleNamespace(compile_learner=True),
            device=torch.device("cuda"),
            emit=emitted.append,
        )
        is trunk_model
    )
    assert trunk_model.modes == ["reduce-overhead"]
    assert emitted[-1] == format_compile_learner_trunk_enabled_message()

    failing_trunk_model = TrunkCompileModel(fail=True)
    assert (
        maybe_compile_learner_model(
            model=failing_trunk_model,
            training_config=SimpleNamespace(compile_learner=True),
            device=torch.device("cuda"),
            emit=emitted.append,
        )
        is None
    )
    assert emitted[-1] == format_compile_learner_trunk_failed_message("RuntimeError('boom')")

    compiled_model = torch.nn.Identity()

    def fake_compile(candidate: torch.nn.Module, *, mode: str) -> torch.nn.Module:
        assert candidate is model
        assert mode == "reduce-overhead"
        return compiled_model

    assert (
        maybe_compile_learner_model(
            model=model,
            training_config=SimpleNamespace(compile_learner=True),
            device=torch.device("cuda"),
            compile_fn=fake_compile,
            emit=emitted.append,
        )
        is compiled_model
    )
    assert emitted[-1] == format_compile_learner_forward_enabled_message()


def test_training_profiling_helpers_preserve_trace_contract(tmp_path):
    with profile_block(False, "disabled"):
        pass

    profiler, profiler_context, profile_dir = build_training_profiler(
        enabled=False,
        run_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert profiler is None
    assert profile_dir is None
    with profiler_context:
        pass

    trace_path = tmp_path / "profiling" / "torch_profiler" / "trace.json"
    assert format_torch_profiler_trace_written_message(trace_path) == f"Wrote torch profiler trace: {trace_path}"


def test_learner_metric_projection_preserves_public_custom_metric_keys():
    update_metrics = {
        "vtrace_rho_p95": 0.95,
        "raw_b1_top_action_ce": 1.25,
        "counterfactual_positive_loss": 0.5,
        "teacher_development_pass_probability": 0.2,
        "not_public": 99.0,
    }
    custom = build_custom_log_metrics(
        update_metrics,
        VtraceMetrics(rho_mean=float("nan"), entropy=0.75),
    )

    assert "raw_b1_top_action_ce" in CUSTOM_LOG_METRIC_KEYS
    assert custom["vtrace_batch_metrics_available"] == 0.0
    assert custom["vtrace_rho_p95"] == pytest.approx(0.95)
    assert custom["vtrace_entropy"] == pytest.approx(0.75)
    assert custom["raw_b1_top_action_ce"] == pytest.approx(1.25)
    assert custom["counterfactual_positive_loss"] == pytest.approx(0.5)
    assert custom["teacher_development_pass_probability"] == pytest.approx(0.2)
    assert "not_public" not in custom


def test_checkpoint_record_helpers_preserve_relative_paths(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    source_checkpoint = checkpoint_dir / "checkpoint_12.pt"
    alias_checkpoint = checkpoint_dir / "best.pt"
    source_checkpoint.write_bytes(b"checkpoint")
    alias_checkpoint.write_bytes(b"alias")

    artifacts = SimpleNamespace(run_dir=run_dir)
    learner = SimpleNamespace(update_count=12, get_policy_version=lambda: 34)

    record = build_checkpoint_record(
        alias_name="best",
        alias_path=alias_checkpoint,
        source_checkpoint_path=source_checkpoint,
        artifacts=artifacts,
        learner=learner,
        metric_kind="dev_eval",
        metric_value=0.75,
    )

    assert record == {
        "alias": "best",
        "alias_path": "training/checkpoints/best.pt",
        "source_checkpoint_path": "training/checkpoints/checkpoint_12.pt",
        "update_count": 12,
        "policy_version": 34,
        "metric_kind": "dev_eval",
        "metric_value": 0.75,
    }

    secondary = build_secondary_checkpoint_record(
        source_checkpoint_path=source_checkpoint,
        artifacts=artifacts,
        update_count=12,
        policy_version=34,
        metric_kind="b2_score",
        metric_value=0.5,
        aggregate_score=0.6,
        dev_eval_ineligibility_reasons=("confidence_ci",),
    )
    assert secondary["source_checkpoint_path"] == "training/checkpoints/checkpoint_12.pt"
    assert secondary["aggregate_score"] == pytest.approx(0.6)
    assert secondary["dev_eval_ineligibility_reasons"] == ["confidence_ci"]


def test_checkpoint_secondary_records_repairs_legacy_shape():
    tracker = {"secondary": None}

    secondary = checkpoint_secondary_records(tracker)

    assert secondary == {}
    assert tracker["secondary"] is secondary


def test_publish_checkpoint_aliases_writes_latest_best_and_secondary_b2(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_20.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    training_paths = SimpleNamespace(
        checkpoint_tracker_path=checkpoint_dir / "checkpoint_tracker.json",
        latest_checkpoint_path=checkpoint_dir / "latest.pt",
        best_checkpoint_path=checkpoint_dir / "best.pt",
    )
    artifacts = SimpleNamespace(run_dir=run_dir)
    learner = SimpleNamespace(update_count=20, get_policy_version=lambda: 21)
    stack = SimpleNamespace(config=SimpleNamespace(curriculum=None, evaluation=None))

    tracker = publish_checkpoint_aliases(
        stack=stack,
        training_paths=training_paths,
        artifacts=artifacts,
        checkpoint_path=checkpoint_path,
        learner=learner,
        latest_metrics={"loss": 1.5},
        dev_eval_summary={
            "aggregate_score": 0.55,
            "anchor_scores": {"B2 HeuristicPublic": 0.4},
        },
        b2_policy_id="B2 HeuristicPublic",
    )

    assert training_paths.latest_checkpoint_path.read_bytes() == b"checkpoint"
    assert training_paths.best_checkpoint_path.read_bytes() == b"checkpoint"
    assert tracker["latest"]["metric_kind"] == "dev_eval_mean"
    assert tracker["best"]["metric_value"] == pytest.approx(0.55)
    assert tracker["secondary"]["best_b2"]["metric_value"] == pytest.approx(0.4)


def test_snapshot_artifacts_apply_promotion_gate_payload_promotes_candidate(tmp_path):
    training_paths = SimpleNamespace(snapshots_dir=tmp_path / "snapshots")
    training_paths.snapshots_dir.mkdir()
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="candidate",
        update=10,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("candidate"),
    )
    registry.save(training_paths.snapshots_dir / "registry.json")
    stack = SimpleNamespace(config=SimpleNamespace(league=None))

    update = apply_promotion_gate_payload(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        payload={
            "passed": True,
            "candidate_policy_id": "candidate",
            "update_count": 10,
            "ordered_opponents": ["B0 RandomLegal", "B1 NoLeague baseline"],
        },
    )

    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert update.passed is True
    assert update.registry_updated is True
    assert update.ordered_opponents == ("B0 RandomLegal", "B1 NoLeague baseline")
    assert format_promotion_gate_registry_update_message(update) == (
        "Promotion gate passed: update=10 candidate=candidate anchors=B0 RandomLegal,B1 NoLeague baseline"
    )
    assert reloaded.champion_snapshots == ["candidate"]


def test_snapshot_artifacts_apply_promotion_gate_payload_rejects_existing_candidate(tmp_path):
    training_paths = SimpleNamespace(snapshots_dir=tmp_path / "snapshots")
    training_paths.snapshots_dir.mkdir()
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="candidate",
        update=10,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("candidate"),
    )
    registry.save(training_paths.snapshots_dir / "registry.json")
    stack = SimpleNamespace(config=SimpleNamespace(league=None))

    update = apply_promotion_gate_payload(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        payload={
            "passed": False,
            "candidate_policy_id": "candidate",
            "update_count": 10,
            "ordered_opponents": [],
            "reasons": [{"code": "low_score"}, {}],
        },
    )

    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert update.passed is False
    assert update.reason_codes == "low_score,unknown"
    assert update.registry_updated is True
    assert format_promotion_gate_registry_update_message(update) == (
        "Promotion gate failed: update=10 candidate=candidate reasons=low_score,unknown"
    )
    assert reloaded.rejected_snapshots == ["candidate"]


def test_snapshot_artifacts_apply_promotion_gate_result_updates_registry_like_inline_gate(tmp_path):
    training_paths = SimpleNamespace(snapshots_dir=tmp_path / "snapshots")
    training_paths.snapshots_dir.mkdir()
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="candidate",
        update=10,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("candidate"),
    )
    stack = SimpleNamespace(config=SimpleNamespace(league=None))

    passed_update = apply_promotion_gate_result(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        registry=registry,
        candidate_policy_id="candidate",
        update_count=10,
        result=SimpleNamespace(
            passed=True,
            ordered_opponents=("B0 RandomLegal",),
            reasons=(),
        ),
    )
    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert passed_update.passed is True
    assert passed_update.ordered_opponents == ("B0 RandomLegal",)
    assert reloaded.champion_snapshots == ["candidate"]

    failed_update = apply_promotion_gate_result(
        stack=stack,
        training_paths=training_paths,
        run_dir=tmp_path,
        registry=reloaded,
        candidate_policy_id="candidate",
        update_count=12,
        result=SimpleNamespace(
            passed=False,
            ordered_opponents=("B0 RandomLegal",),
            reasons=({"code": "low_score"}, {}),
        ),
    )
    reloaded = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    assert failed_update.passed is False
    assert failed_update.reason_codes == "low_score,unknown"
    assert reloaded.rejected_snapshots == ["candidate"]


def test_snapshot_import_helpers_preserve_public_policy_ids(tmp_path):
    canonical = {"config": {"experiment": {"role": "baseline_noleague_contract"}, "model": {}}}
    legacy = {"training_family_a": {"mode": "b1_no_league"}}
    source_run_dir = tmp_path / "source" / "run"
    expected_prefix = seed_snapshot_policy_id(source_run_dir=source_run_dir, source_policy_id="x")[:15]

    assert config_marks_noleague_baseline(canonical)
    assert config_marks_noleague_baseline(legacy)
    assert seed_snapshot_policy_id(source_run_dir=source_run_dir, source_policy_id="policy/001") == (
        f"{expected_prefix}_policy_001"
    )

    local_snapshot = SnapshotMeta(
        policy_id="policy_000100",
        update=100,
        weights_sha256="ab" * 32,
        path=snapshot_weights_relpath("policy_000100"),
        source_kind="local",
    )
    seed_snapshot = SnapshotMeta(
        policy_id="seed_abc_policy_000100",
        update=100,
        weights_sha256="ab" * 32,
        path=snapshot_weights_relpath("seed_abc_policy_000100"),
        source_kind="seed_import",
    )

    assert source_snapshot_is_resume_league_snapshot(local_snapshot, rejected_policy_ids=set())
    assert not source_snapshot_is_resume_league_snapshot(seed_snapshot, rejected_policy_ids=set())
    assert not source_snapshot_is_resume_league_snapshot(local_snapshot, rejected_policy_ids={"policy_000100"})


def test_snapshot_imports_current_run_baseline_anchor_service_writes_alias(tmp_path):
    run_dir = tmp_path / "run"
    training_paths = SimpleNamespace(
        snapshots_dir=run_dir / "training" / "snapshots",
        checkpoints_dir=run_dir / "training" / "checkpoints",
    )
    training_paths.snapshots_dir.mkdir(parents=True)
    training_paths.checkpoints_dir.mkdir(parents=True)
    SnapshotRegistry().save(training_paths.snapshots_dir / "registry.json")
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=None,
            training=SimpleNamespace(
                reference_policy_id="b1_noleague_baseline",
                reference_policy_top_action_bc_coef=0.0,
                reference_policy_top_action_family_bc_coef=0.0,
                b1_opponent_reference_policy_top_action_bc_coef=0.0,
                raw_b1_distill=SimpleNamespace(enabled=False, coef=0.0, final_coef=0.0),
            ),
        )
    )

    result = ensure_noleague_baseline_anchor(
        stack=stack,
        training_paths=training_paths,
        run_dir=run_dir,
        model_state_dict={"weight": torch.ones(1)},
        learner_update_count=7,
        device=torch.device("cpu"),
        config_hash256="ab" * 32,
        expected_config_canonical={},
        permit_current_run_alias=True,
        write_checkpoint=lambda path: path.write_bytes(b"checkpoint"),
        guidance_payload={
            "public_heuristic_logit_bias_scale": 0.25,
            "public_heuristic_actor_logit_bias_scale": 0.5,
        },
        experiment_role="baseline_noleague",
    )

    assert result.policy_id == "b1_noleague_baseline"
    assert result.message is not None
    registry = SnapshotRegistry.load(training_paths.snapshots_dir / "registry.json")
    snapshot = next(snapshot for snapshot in registry.snapshots if snapshot.policy_id == "b1_noleague_baseline")
    assert snapshot is not None
    assert snapshot.update == 7
    assert "b1_noleague_baseline" in registry.pinned_snapshots
    payload = torch.load(run_dir / snapshot.path, map_location="cpu", weights_only=True)
    assert payload["public_heuristic_logit_bias_scale"] == pytest.approx(0.25)
    assert payload["public_heuristic_actor_logit_bias_scale"] == pytest.approx(0.5)


def test_anchor_resolution_helpers_preserve_symbolic_anchor_semantics():
    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=10,
        weights_sha256="b" * 64,
        path=snapshot_weights_relpath("b1_noleague_baseline"),
    )
    registry.add_snapshot(
        policy_id="policy_000100",
        update=100,
        weights_sha256="c" * 64,
        path=snapshot_weights_relpath("policy_000100"),
    )
    registry.add_snapshot(
        policy_id="policy_000200",
        update=200,
        weights_sha256="d" * 64,
        path=snapshot_weights_relpath("policy_000200"),
    )
    registry.add_champion("policy_000100")
    registry.add_champion("policy_000200")

    assert promotion_anchor_policy_id_candidates("B1 NoLeague baseline") == (
        "b1_noleague_baseline",
        "B1 NoLeague baseline",
    )
    assert true_local_recent_snapshot_ids(registry) == ("policy_000100", "policy_000200")
    assert (
        resolve_symbolic_promotion_anchor_policy_id(
            "Latest champion snapshot",
            registry=registry,
            promotion_gate_enabled=True,
        )
        == "policy_000200"
    )


def test_guidance_schedule_helpers_update_learner_and_model_metrics() -> None:
    training = SimpleNamespace(
        entropy_coef=0.02,
        entropy_anneal_to=0.002,
        entropy_anneal_steps_updates=100,
        teacher_public_heuristic_coef=1.0,
        teacher_public_heuristic_final_coef=0.0,
        teacher_public_heuristic_start_updates=0,
        teacher_public_heuristic_end_updates=100,
        reference_policy_top_action_bc_coef=0.5,
        reference_policy_top_action_bc_final_coef=0.1,
        reference_policy_top_action_bc_start_updates=0,
        reference_policy_top_action_bc_end_updates=100,
        reference_policy_top_action_family_bc_coef=0.4,
        reference_policy_top_action_family_bc_final_coef=0.0,
        reference_policy_top_action_family_bc_start_updates=0,
        reference_policy_top_action_family_bc_end_updates=100,
        raw_b1_distill=SimpleNamespace(enabled=True, coef=0.3, final_coef=0.0, start_updates=0, end_updates=100),
        counterfactual_positive=SimpleNamespace(
            enabled=True, coef=0.2, final_coef=0.0, start_updates=0, end_updates=100
        ),
    )
    model_config = SimpleNamespace(
        public_heuristic_logit_bias_scale=0.6,
        public_heuristic_logit_bias_final_scale=0.2,
        public_heuristic_logit_bias_start_updates=0,
        public_heuristic_logit_bias_end_updates=100,
        public_heuristic_actor_logit_bias_scale=-1.0,
    )
    learner_calls: dict[str, object] = {}

    class _Learner:
        def set_teacher_aux_coefs(self, **kwargs):
            learner_calls["teacher"] = kwargs

        def set_reference_policy_bc_coefs(self, **kwargs):
            learner_calls["reference"] = kwargs

        def set_raw_b1_distill_coef(self, value):
            learner_calls["raw_b1"] = value

        def set_counterfactual_positive_coef(self, value):
            learner_calls["counterfactual"] = value

    class _Model:
        def __init__(self) -> None:
            self.learner_scale = 0.0
            self.actor_scale = 0.0

        def set_public_heuristic_logit_bias_scale(self, learner_value, *, actor_value=None):
            self.learner_scale = float(learner_value)
            self.actor_scale = float(actor_value)

        def get_public_heuristic_logit_bias_scale(self, *, scoring_mode):
            return self.actor_scale if scoring_mode == "actor" else self.learner_scale

    model = _Model()
    stack = SimpleNamespace(config=SimpleNamespace(training=training, model=model_config))

    assert entropy_coef_for_next_update(training, update_count=50) == pytest.approx(0.011)
    assert reference_policy_top_action_bc_coef_for_next_update(training, update_count=50) == pytest.approx(0.3)
    assert raw_b1_distill_coef_for_next_update(training, update_count=50) == pytest.approx(0.15)
    assert counterfactual_positive_coef_for_next_update(training, update_count=50) == pytest.approx(0.1)

    metrics = apply_guidance_schedule_for_next_update(
        learner=_Learner(),
        model=model,
        stack=stack,
        update_count=50,
    )

    assert learner_calls["teacher"] == {"public_heuristic": pytest.approx(0.5)}
    assert learner_calls["reference"] == {"top_action": pytest.approx(0.3), "top_action_family": pytest.approx(0.2)}
    assert learner_calls["raw_b1"] == pytest.approx(0.15)
    assert learner_calls["counterfactual"] == pytest.approx(0.1)
    assert metrics["public_heuristic_logit_bias_scale_active"] == pytest.approx(0.4)
    assert metrics["public_heuristic_actor_logit_bias_scale_active"] == pytest.approx(0.4)
    assert model_guidance_payload(model) == {
        "public_heuristic_actor_logit_bias_scale": pytest.approx(0.4),
        "public_heuristic_logit_bias_scale": pytest.approx(0.4),
    }

    restore_model_guidance_from_payload(
        model,
        {"public_heuristic_logit_bias_scale": 0.7, "public_heuristic_actor_logit_bias_scale": 0.25},
    )
    assert model_guidance_payload(model) == {
        "public_heuristic_actor_logit_bias_scale": pytest.approx(0.25),
        "public_heuristic_logit_bias_scale": pytest.approx(0.7),
    }
    assert format_attached_reference_policy_message(
        policy_id="b1_noleague_baseline",
        coef=0.5,
        family_coef=0.25,
        raw_b1_distill_enabled=True,
        weights_path="training/snapshots/b1/weights.pt",
    ) == (
        "Attached frozen reference policy: policy_id=b1_noleague_baseline coef=0.5 family_coef=0.25 "
        "raw_b1_distill=True weights=training/snapshots/b1/weights.pt"
    )


def test_checkpoint_guard_rollback_plan_preserves_score_drop_and_confidence_reasons():
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    cooldown_updates=20,
                    min_best_score=0.2,
                    rollback_score_margin=0.05,
                    rollback_truncation_rate_threshold=0.25,
                    rollback_max_prob_lt_half=0.6,
                )
            )
        )
    )

    plan = checkpoint_guard_rollback_plan(
        stack=stack,
        learner_update_count=120,
        last_rollback_update=None,
        best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.7, "update_count": 100},
        dev_eval_summary={
            "aggregate_score": 0.6,
            "evaluation_surface": {"authoritative": True},
            "anchors": {
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.1, "prob_lt_half": 0.8, "ci_half_width": 0.2},
                }
            },
        },
    )

    assert plan is not None
    assert plan.best_update_count == 100
    assert plan.best_score == pytest.approx(0.7)
    assert plan.current_score == pytest.approx(0.6)
    assert plan.reasons == ("score_drop", "confidence")
    assert plan.max_prob_lt_half == pytest.approx(0.8)


def test_checkpoint_guard_rollback_message_preserves_train_console_contract():
    guard_event = {
        "update_count": 120,
        "best_update_count": 80,
        "current_score": 0.123456,
        "best_score": 0.654321,
        "reasons": ["score_drop", "confidence"],
    }

    assert format_checkpoint_guard_rollback_message(guard_event) == (
        "Checkpoint guard rollback: update=120 best_update=80 "
        "current_score=0.1235 best_score=0.6543 reasons=score_drop,confidence"
    )
    assert format_checkpoint_guard_final_selection_message(guard_event) == (
        "Checkpoint guard final selection: update=120 best_update=80 current_score=0.1235 best_score=0.6543"
    )


def test_checkpoint_guard_rollback_plan_honors_cooldown_and_improving_scores():
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    cooldown_updates=20,
                    min_best_score=0.2,
                    rollback_score_margin=0.05,
                    rollback_truncation_rate_threshold=0.25,
                    rollback_max_prob_lt_half=0.6,
                )
            )
        )
    )
    best_record = {"metric_kind": "dev_eval_mean", "metric_value": 0.7, "update_count": 100}
    improving_summary = {"aggregate_score": 0.75, "evaluation_surface": {"authoritative": True}}
    declining_summary = {"aggregate_score": 0.6, "evaluation_surface": {"authoritative": True}}

    assert (
        checkpoint_guard_rollback_plan(
            stack=stack,
            learner_update_count=110,
            last_rollback_update=100,
            best_record=best_record,
            dev_eval_summary=declining_summary,
        )
        is None
    )
    assert (
        checkpoint_guard_rollback_plan(
            stack=stack,
            learner_update_count=130,
            last_rollback_update=100,
            best_record=best_record,
            dev_eval_summary=improving_summary,
        )
        is None
    )


def test_dev_eval_b2_helpers_preserve_warning_contract():
    summaries = {
        "policy_000010": {"update_count": 10, "anchor_scores": {"B2 HeuristicPublic": 0.21}},
        "policy_000020": {"update_count": 20, "b2": {"score": 0.22}},
        "policy_000030": {"update_count": 30, "anchor_scores": {"B2 HeuristicPublic": 0.9}},
    }

    recent = b2_recent_scores_from_persisted_summaries(
        summaries,
        current_policy_id="policy_000030",
        heuristic_public_policy_id="B2 HeuristicPublic",
    )
    warnings = build_b2_warning_flags(
        current_score=0.22,
        current_summary={
            "total_actions": 20,
            "main_move_actions": 10,
            "pass_with_nonpass_available": 0,
            "max_consecutive_main_moves": 4,
        },
        recent_scores=recent,
    )

    assert recent == [0.21, 0.22]
    assert [warning["kind"] for warning in warnings] == [
        "b2_flatline_v1",
        "b2_action_family_warning_v1",
    ]
    assert warnings[0]["recent_eval_count"] == 3
    assert warnings[1]["main_move_rate"] == pytest.approx(0.5)

    assert extract_anchor_score(
        {
            "anchors": {
                "B2 HeuristicPublic": {
                    "uncertainty": {
                        "mean": 0.42,
                    }
                }
            }
        },
        "B2 HeuristicPublic",
    ) == pytest.approx(0.42)


def test_curriculum_guard_helpers_write_stall_and_early_cutoff_state(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(
                    enabled=True,
                    truncation_rate_threshold=0.25,
                    consecutive_evals=1,
                ),
                early_cutoff=SimpleNamespace(
                    enabled=True,
                    warmup_updates=0,
                    patience_updates=5,
                    min_improvement=0.01,
                    stall_patience_evals=1,
                    stall_rate_threshold=0.25,
                ),
            )
        )
    )
    summary = {
        "aggregate_score": 0.4,
        "anchors": {
            "B2 HeuristicPublic": {
                "summary": {
                    "games": 4,
                    "truncations": 2,
                    "no_progress_timeouts": 2,
                    "natural_timeouts": 0,
                }
            }
        },
    }

    stall = update_stall_monitor(
        stack=stack,
        training_paths=training_paths,
        update_count=10,
        summary_payload=summary,
    )
    cutoff = update_early_cutoff(
        stack=stack,
        training_paths=training_paths,
        update_count=10,
        summary_payload=summary,
    )

    assert stall is not None
    assert stall["stall_risk"] is True
    assert stall["worst_no_progress_timeout_rate"] == pytest.approx(0.5)
    assert cutoff is not None
    assert cutoff["should_stop"] is True
    assert "stall" in cutoff["reasons"]
    assert (training_paths.logs_dir / "stall_monitor.json").is_file()
    assert (training_paths.logs_dir / "early_cutoff_events.jsonl").is_file()


def test_curriculum_guard_early_cutoff_console_and_metric_contracts():
    payload = {
        "best_score": 0.61,
        "current_score": 0.42,
        "best_update_count": 20,
        "no_improvement_updates": 12,
        "consecutive_stall_evals": 3,
        "reasons": ["no_improvement", "stall"],
    }

    updates = early_cutoff_metric_updates(payload)

    assert updates == {
        "early_cutoff_triggered": 1.0,
        "early_cutoff_best_score": pytest.approx(0.61),
        "early_cutoff_current_score": pytest.approx(0.42),
        "early_cutoff_no_improvement_updates": pytest.approx(12.0),
        "early_cutoff_consecutive_stall_evals": pytest.approx(3.0),
    }
    assert format_early_cutoff_triggered_message(payload, update_count=32) == (
        "Early cutoff triggered: update=32 best_update=20 best_score=0.6100 "
        "current_score=0.4200 reasons=no_improvement,stall"
    )
    assert format_training_stopped_by_early_cutoff_message(updates) == (
        "Training stopped by early cutoff: best_score=0.6100 current_score=0.4200 "
        "no_improvement_updates=12 consecutive_stall_evals=3"
    )


def test_curriculum_guard_applies_stall_monitor_to_authoritative_summary(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(
                    enabled=True,
                    truncation_rate_threshold=0.25,
                    consecutive_evals=1,
                )
            )
        )
    )
    summary_path = tmp_path / "eval" / "dev_eval" / "update_10" / "summary.json"
    summary = {
        "format": "periodic_dev_eval_summary_v2",
        "update_count": 10,
        "anchors": {
            "B2 HeuristicPublic": {
                "summary": {
                    "games": 4,
                    "truncations": 2,
                    "no_progress_timeouts": 1,
                    "natural_timeouts": 0,
                }
            }
        },
    }

    stall = apply_stall_monitor_to_dev_eval_summary(
        stack=stack,
        training_paths=training_paths,
        summary_payload=summary,
        summary_path=summary_path,
    )

    assert stall is not None
    assert summary["stall_monitor"] == stall
    assert json.loads(summary_path.read_text(encoding="utf-8"))["stall_monitor"] == stall
    assert format_stall_monitor_warning(stall, update_count=10) == (
        "Stall monitor warning: update=10 worst_anchor=B2 HeuristicPublic "
        "stall_rate=0.250 no_progress_rate=0.250 truncation_rate=0.500 consecutive=1"
    )


def test_eval_seed_helpers_are_deterministic_and_surface_specific():
    scheduled_game = SimpleNamespace(
        pair_index=3,
        swap_index=1,
        episode_seed=12345,
        seat0_policy_id="candidate",
        seat1_policy_id="anchor",
    )

    periodic_seed = periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=0)
    promotion_seed = promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=0)
    periodic_bootstrap_seed = periodic_dev_eval_bootstrap_seed(update_count=10, policy_version=11)
    promotion_bootstrap_seed = promotion_gate_bootstrap_seed(update_count=10, policy_version=11)
    expanded = expand_periodic_dev_eval_paired_seeds(
        [1, 2],
        requested_pairs=4,
        seed_file_sha256="ab" * 32,
        update_count=10,
        policy_version=11,
        scope="periodic_dev_eval_confirmatory",
    )

    assert periodic_seed == periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=0)
    assert promotion_seed == promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=0)
    assert periodic_seed != promotion_seed
    assert periodic_bootstrap_seed == periodic_dev_eval_bootstrap_seed(update_count=10, policy_version=11)
    assert promotion_bootstrap_seed == promotion_gate_bootstrap_seed(update_count=10, policy_version=11)
    assert periodic_bootstrap_seed != promotion_bootstrap_seed
    assert expanded[:2] == [1, 2]
    assert len(expanded) == 4
    assert len(set(expanded)) == 4


def test_confirmatory_eval_plan_expands_seed_file_with_request_reasons(tmp_path):
    seed_file = tmp_path / "dev_eval_seeds.txt"
    seed_file.write_text("11\n22\n", encoding="utf-8")
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"dev_eval": seed_file},
        config=SimpleNamespace(
            reproducibility=None,
            evaluation=SimpleNamespace(
                seat_swap=True,
                eval_inference_mode=True,
                eval_sampling_algorithm="pinned_cdf_pcg_v1",
                seed_files={},
                periodic_dev_eval_paired_seeds=2,
                final_matrix_stage2_adaptive_max_paired_seeds=32,
            ),
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(enabled=False, truncation_rate_threshold=0.25),
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    min_best_score=0.0,
                    rollback_score_margin=0.1,
                    promote_min_prob_gt_half=0.8,
                    promote_max_ci_half_width=0.1,
                ),
            ),
        ),
    )

    plan = build_confirmatory_dev_eval_plan(
        stack=stack,
        existing_best_record={"metric_kind": "dev_eval_mean", "metric_value": 0.62},
        dev_eval_summary={
            "update_count": 12,
            "policy_version": 13,
            "aggregate_score": 0.625,
            "evaluation_surface": {"authoritative": True},
            "anchors": {
                "B0 RandomLegal": {
                    "uncertainty": {"prob_gt_half": 1.0, "prob_lt_half": 0.0, "ci_half_width": 0.05},
                },
                "B1 NoLeague baseline": {
                    "uncertainty": {"prob_gt_half": 0.9, "prob_lt_half": 0.1, "ci_half_width": 0.109},
                },
            },
            "stall_monitor": {"worst_truncation_rate": 0.0},
        },
    )

    assert plan is not None
    assert plan.seed_file == seed_file
    assert plan.reasons == ("confidence_ci",)
    assert plan.target_pairs == 32
    assert plan.paired_seeds[:2] == (11, 22)
    assert len(plan.paired_seeds) == 32
    assert len(set(plan.paired_seeds)) == 32


def test_confirmatory_eval_message_preserves_train_console_contract(tmp_path):
    seed_file = tmp_path / "dev_eval_seeds.txt"

    assert format_confirmatory_dev_eval_message(
        update_count=12,
        paired_seed_count=8,
        aggregate_score=0.123456,
        reasons=("confidence_prob", "confidence_ci"),
        seed_file=seed_file,
    ) == (
        "Confirmatory dev eval: update=12 paired_seeds=8 aggregate=0.1235 "
        "reasons=confidence_prob,confidence_ci seed_file=dev_eval_seeds.txt"
    )


def _confidence_gate_stack():
    return SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(enabled=False, truncation_rate_threshold=0.25),
                checkpoint_guard=SimpleNamespace(
                    enabled=True,
                    promote_min_prob_gt_half=0.8,
                    promote_max_ci_half_width=0.1,
                ),
            )
        )
    )


def _authoritative_b2_summary(*, update_count: int = 40, policy_version: int = 3) -> dict[str, object]:
    return {
        "policy_id": f"train_u{update_count}_p{policy_version}",
        "update_count": update_count,
        "policy_version": policy_version,
        "aggregate_score": 0.58,
        "evaluation_surface": {"authoritative": True},
        "anchor_scores": {
            "B0 RandomLegal": 1.0,
            "B1 NoLeague baseline": 0.6,
            "B2 HeuristicPublic": 0.32,
        },
        "anchors": {
            "B0 RandomLegal": {
                "uncertainty": {"mean": 1.0, "ci_half_width": 0.01, "prob_gt_half": 1.0},
            },
            "B1 NoLeague baseline": {
                "uncertainty": {"mean": 0.6, "ci_half_width": 0.02, "prob_gt_half": 0.95},
            },
            "B2 HeuristicPublic": {
                "summary": {
                    "games": 16,
                    "total_actions": 100,
                    "main_move_actions": 60,
                    "pass_with_nonpass_available": 10,
                    "max_consecutive_main_moves": 8,
                },
                "uncertainty": {"mean": 0.32, "ci_half_width": 0.25, "prob_gt_half": 0.2},
                "evaluation_context": {
                    "episodes_path": f"eval/dev_eval/update_{update_count}/B2 HeuristicPublic/episodes.jsonl",
                },
            },
        },
        "b2": {"warning_flags": [{"kind": "b2_flatline_v1"}]},
    }


def test_eval_artifacts_persist_summary_and_fast_screen_records(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    payload_a = _authoritative_b2_summary(update_count=10, policy_version=1)
    payload_a["anchor_scores"]["B2 HeuristicPublic"] = 0.21
    payload_b = _authoritative_b2_summary(update_count=20, policy_version=2)
    payload_b["policy_id"] = "train_u20_p2"
    payload_b["anchor_scores"]["B2 HeuristicPublic"] = 0.22
    payload_c = _authoritative_b2_summary(update_count=30, policy_version=3)
    payload_c["policy_id"] = "train_u30_p3"
    payload_c["anchor_scores"]["B2 HeuristicPublic"] = 0.22
    payload_c["periodic_dev_eval_parallel"] = {"workers": 2}
    payload_c["periodic_dev_eval_runtime"] = {"elapsed_sec": 1.5}

    for payload in (payload_a, payload_b, payload_c):
        persist_periodic_dev_eval_summary(
            training_paths=training_paths,
            payload=payload,
            b2_policy_id="B2 HeuristicPublic",
        )
    persist_periodic_dev_eval_fast_screen(training_paths=training_paths, payload=payload_c)

    summaries = json.loads((training_paths.logs_dir / "periodic_dev_eval_summaries.json").read_text(encoding="utf-8"))
    latest = summaries["train_u30_p3"]
    assert latest["format"] == "periodic_dev_eval_summary_v2"
    assert latest["b2"]["score"] == pytest.approx(0.22)
    assert {warning["kind"] for warning in latest["warning_flags"]} == {
        "b2_flatline_v1",
        "b2_action_family_warning_v1",
    }

    fast_screens = json.loads(periodic_dev_eval_fast_screens_path(training_paths).read_text(encoding="utf-8"))
    assert fast_screens["train_u30_p3"]["format"] == "periodic_dev_eval_fast_screen_v1"
    assert fast_screens["train_u30_p3"]["periodic_dev_eval_parallel"] == {"workers": 2}
    assert fast_screens["train_u30_p3"]["periodic_dev_eval_runtime"] == {"elapsed_sec": 1.5}


def test_eval_model_cache_helpers_preserve_lru_and_eval_touch() -> None:
    class FakeEvalModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1
            return self

    cache: OrderedDict[tuple[str], FakeEvalModel] = OrderedDict()
    model_a = FakeEvalModel("a")
    model_b = FakeEvalModel("b")
    model_c = FakeEvalModel("c")

    remember_eval_model(cache, ("a",), model_a, max_entries=2)
    remember_eval_model(cache, ("b",), model_b, max_entries=2)

    assert list(cache.keys()) == [("a",), ("b",)]
    assert get_cached_eval_model(cache, ("a",)) is model_a
    assert model_a.eval_calls == 1
    assert list(cache.keys()) == [("b",), ("a",)]

    remember_eval_model(cache, ("c",), model_c, max_entries=2)

    assert list(cache.keys()) == [("a",), ("c",)]
    assert get_cached_eval_model(cache, ("b",)) is None


def test_eval_artifacts_periodic_dev_eval_console_message_preserves_train_contract() -> None:
    assert format_periodic_dev_eval_console_message(
        label="Periodic dev eval complete",
        update_count=12,
        aggregate_score=0.123456,
        anchor_names=["B0 RandomLegal", "B2 HeuristicPublic"],
        opponent_slug="b0_randomlegal",
    ) == (
        "Periodic dev eval complete: update=12 opponent=b0_randomlegal "
        "aggregate=0.1235 anchors=B0 RandomLegal,B2 HeuristicPublic"
    )
    assert format_periodic_dev_eval_scheduled_message(
        update_count=12,
        worker_devices=("cuda:1", "cuda:2"),
        fallback_eval_device="cpu",
        anchor_names=("B0 RandomLegal", "B2 HeuristicPublic"),
    ) == ("Periodic dev eval scheduled: update=12 devices=cuda:1,cuda:2 anchors=B0 RandomLegal,B2 HeuristicPublic")
    assert (
        format_periodic_dev_eval_scheduled_message(
            update_count=12,
            worker_devices=(),
            fallback_eval_device="cpu",
            anchor_names=("B0 RandomLegal",),
        )
        == "Periodic dev eval scheduled: update=12 devices=cpu anchors=B0 RandomLegal"
    )


def test_eval_artifacts_matchup_dirs_disambiguate_duplicate_policy_ids(tmp_path):
    specs = (
        PeriodicDevEvalOpponentSpec(
            policy_id="b1_noleague_baseline",
            display_name="B1 NoLeague baseline",
            kind="snapshot",
            snapshot_path="snapshots/b1/weights.pt",
        ),
        PeriodicDevEvalOpponentSpec(
            policy_id="b1_noleague_baseline",
            display_name="Previous recent snapshot",
            kind="snapshot",
            snapshot_path="snapshots/recent/weights.pt",
        ),
        PeriodicDevEvalOpponentSpec(
            policy_id="b2_heuristic_public",
            display_name="B2 HeuristicPublic",
            kind="heuristic",
            heuristic_profile="public",
        ),
    )

    duplicate_policy_ids = periodic_dev_eval_duplicate_policy_ids(specs)
    dirs = [
        periodic_dev_eval_matchup_dir(
            update_dir=tmp_path,
            opponent_spec=spec,
            duplicate_policy_ids=duplicate_policy_ids,
        )
        for spec in specs
    ]

    assert duplicate_policy_ids == {"b1_noleague_baseline"}
    assert dirs[0] != dirs[1]
    assert all(path.name.startswith("b1_noleague_baseline__") for path in dirs[:2])
    assert dirs[2] == tmp_path / "b2_heuristic_public"


def test_eval_artifacts_sum_periodic_dev_eval_counter_payloads_sorts_and_coerces_values():
    payload = sum_periodic_dev_eval_counter_payloads(
        [
            {"seconds": {"run": "1.5", "load": 2}, "counts": {"episodes": "3"}},
            {"seconds": {"run": 0.25, "write": 4}, "counts": {"episodes": 5, "faults": False}},
        ]
    )

    assert list(payload["seconds"]) == ["load", "run", "write"]
    assert payload["seconds"] == {"load": 2.0, "run": 1.75, "write": 4.0}
    assert payload["counts"] == {"episodes": 8, "faults": 0}


def test_eval_artifacts_checkpoint_summary_preserves_parallel_and_weighting_contract():
    summary = build_periodic_dev_eval_checkpoint_summary(
        focal_policy_id="policy_000012",
        update_count=12,
        policy_version=13,
        matchup_results=[
            {
                "policy_id": "b1",
                "display_name": "B1",
                "matchup_payload": {
                    "uncertainty": {"mean": 0.25},
                    "seat_diagnostics": {"delta": 0.1},
                    "evaluation_runtime": {"game_count": 4, "seed_block_count": 2},
                },
            },
            {
                "policy_id": "b0_randomlegal",
                "display_name": "B0 RandomLegal",
                "matchup_payload": {
                    "uncertainty": {"mean": 0.75},
                    "seat_diagnostics": {"delta": 0.0},
                    "evaluation_runtime": {"game_count": 6},
                },
            },
        ],
        anchor_weight_config={"B0 RandomLegal": 3.0},
        effective_parallel_workers=2,
        worker_devices=("cuda:0", "cuda:1"),
        seed_block_job_count=4,
        batched_inference_enabled=True,
        total_eval_wall_clock_seconds=2.0,
    )

    assert summary["policy_id"] == "policy_000012"
    assert summary["aggregate_score"] == pytest.approx((0.25 + 0.75 * 3.0) / 4.0)
    assert summary["unweighted_aggregate_score"] == pytest.approx(0.5)
    assert summary["anchor_scores"] == {"B1": 0.25, "B0 RandomLegal": 0.75}
    assert summary["periodic_dev_eval_parallel"] == {
        "enabled": True,
        "worker_count": 2,
        "worker_devices": ["cuda:0", "cuda:1"],
        "job_count": 4,
        "batched_inference_enabled": True,
        "seed_block_sharding_enabled": True,
    }
    assert summary["periodic_dev_eval_runtime"]["game_count"] == 10
    assert summary["periodic_dev_eval_runtime"]["games_per_sec"] == pytest.approx(5.0)
    assert summary["evaluation_surface"] == {
        "kind": "fast_batched_screen",
        "authoritative": False,
        "batched_inference_enabled": True,
    }


def test_eval_artifacts_seed_usage_payload_preserves_serial_and_parallel_schema(tmp_path):
    seed_file = tmp_path / "seeds" / "dev.txt"
    checkpoint_path = tmp_path / "run" / "training" / "checkpoints" / "latest.pt"

    payload = build_periodic_dev_eval_seed_usage_payload(
        seed_file=seed_file,
        seed_root=tmp_path,
        seed_file_sha256="ab" * 32,
        validated_sources={"source": "ok"},
        artifact_scope="periodic_dev_eval",
        scheduled_paired_seed_count=1,
        paired_seeds=(11, 22),
        seat_swap=True,
        eval_device="cuda:0",
        eval_inference_mode=True,
        eval_sampling_algorithm="pinned_cdf_pcg_v1",
        eval_assert_sorted_legal_ids=True,
        focal_policy_id="policy_000012",
        update_count=12,
        policy_version=13,
        checkpoint_path=checkpoint_path,
        run_dir=tmp_path / "run",
        opponent_policy_id="b0_randomlegal",
        opponent_display_name="B0 RandomLegal",
        parallel_seed_blocks=({"block_index": 0, "worker_index": 1},),
    )

    assert payload["seed_file"] == {
        "path": "seeds/dev.txt",
        "sha256": "ab" * 32,
        "validated_sources": {"source": "ok"},
    }
    assert payload["seed_schedule"] == {
        "configured_paired_seed_count": 1,
        "requested_paired_seed_count": 2,
        "expanded_beyond_seed_file": True,
    }
    assert payload["paired_seeds"] == [11, 22]
    assert payload["protocol"]["eval_device"] == "cuda:0"
    assert payload["focal_policy"]["checkpoint_path"] == "training/checkpoints/latest.pt"
    assert payload["opponent_policy"] == {
        "policy_id": "b0_randomlegal",
        "display_name": "B0 RandomLegal",
    }
    assert payload["parallel_seed_blocks"] == [{"block_index": 0, "worker_index": 1}]


def test_eval_artifacts_matchup_context_and_runtime_payloads_preserve_schema(tmp_path):
    run_dir = tmp_path / "run"
    matchup_dir = run_dir / "eval" / "dev_eval" / "update_12" / "b0_randomlegal"
    checkpoint_path = run_dir / "training" / "checkpoints" / "latest.pt"

    context = build_periodic_dev_eval_matchup_context_payload(
        artifact_scope="periodic_dev_eval",
        update_count=12,
        policy_version=13,
        checkpoint_path=checkpoint_path,
        matchup_dir=matchup_dir,
        run_dir=run_dir,
        anchor_display_name="B0 RandomLegal",
    )
    serial_runtime = build_periodic_dev_eval_matchup_runtime_payload(
        wall_clock_seconds=2.0,
        game_count=8,
        runner_counters={"seconds": {"run": 1.5}, "counts": {"games": 8}},
        batched_model_inference=False,
    )
    parallel_runtime = build_periodic_dev_eval_matchup_runtime_payload(
        wall_clock_seconds=0.0,
        game_count=8,
        runner_counters={"seconds": {}, "counts": {}},
        batched_model_inference=True,
        seed_block_count=2,
        serial_worker_wall_clock_seconds_sum=3.25,
    )

    assert context == {
        "artifact_scope": "periodic_dev_eval",
        "update_count": 12,
        "policy_version": 13,
        "checkpoint_path": "training/checkpoints/latest.pt",
        "matchup_dir": "eval/dev_eval/update_12/b0_randomlegal",
        "episodes_path": "eval/dev_eval/update_12/b0_randomlegal/episodes.jsonl",
        "seed_usage_path": "eval/dev_eval/update_12/b0_randomlegal/seed_usage.json",
        "anchor_display_name": "B0 RandomLegal",
    }
    assert serial_runtime == {
        "wall_clock_seconds": 2.0,
        "games_per_sec": 4.0,
        "game_count": 8,
        "persistent_env_reuse": True,
        "batched_model_inference": False,
        "runner_counters": {"seconds": {"run": 1.5}, "counts": {"games": 8}},
    }
    assert parallel_runtime == {
        "wall_clock_seconds": 0.0,
        "games_per_sec": 0.0,
        "game_count": 8,
        "persistent_env_reuse": True,
        "seed_block_count": 2,
        "batched_model_inference": True,
        "serial_worker_wall_clock_seconds_sum": 3.25,
        "runner_counters": {"seconds": {}, "counts": {}},
    }


def test_eval_artifacts_collates_periodic_seed_block_matchup_by_opponent_and_block_order():
    block_results = [
        {
            "opponent_index": 0,
            "block_index": 1,
            "paired_seed_items": ((1, 22),),
            "records": (SimpleNamespace(pair_index=1, swap_index=1),),
            "wall_clock_seconds": 0.5,
            "runner_counters": {"seconds": {"run": 0.4}, "counts": {"games": 1}},
            "worker_index": 1,
            "worker_device": "cuda:1",
        },
        {
            "opponent_index": 0,
            "block_index": 0,
            "paired_seed_items": ((0, 11),),
            "records": (
                SimpleNamespace(pair_index=0, swap_index=1),
                SimpleNamespace(pair_index=0, swap_index=0),
            ),
            "wall_clock_seconds": 0.75,
            "runner_counters": {"seconds": {"run": 0.7}, "counts": {"games": 2}},
            "worker_index": 0,
            "worker_device": "cuda:0",
        },
    ]

    grouped = group_periodic_dev_eval_seed_block_results(block_results)
    collated = collate_periodic_dev_eval_seed_block_matchup(
        block_results_by_opponent=grouped,
        opponent_index=0,
        opponent_display_name="B0 RandomLegal",
    )

    assert [record.pair_index for record in collated["records"]] == [0, 0, 1]
    assert [record.swap_index for record in collated["records"]] == [0, 1, 1]
    assert collated["parallel_seed_blocks"] == [
        {
            "block_index": 0,
            "paired_seed_items": [{"pair_index": 0, "seed": 11}],
            "worker_index": 0,
            "worker_device": "cuda:0",
        },
        {
            "block_index": 1,
            "paired_seed_items": [{"pair_index": 1, "seed": 22}],
            "worker_index": 1,
            "worker_device": "cuda:1",
        },
    ]
    assert collated["wall_clock_seconds"] == 0.75
    assert collated["serial_worker_wall_clock_seconds_sum"] == pytest.approx(1.25)
    assert collated["seed_block_count"] == 2
    assert collated["runner_counters"] == {"seconds": {"run": 1.1}, "counts": {"games": 3}}


def test_eval_artifacts_persist_periodic_result_routes_authoritative_and_fast_screen(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    authoritative_payload = {
        "policy_id": "train_u12_p13",
        "aggregate_score": 0.25,
        "anchor_scores": {"B0 RandomLegal": 0.25},
        "update_count": 12,
        "policy_version": 13,
        "evaluation_surface": {
            "kind": "canonical_scalar",
            "authoritative": True,
            "batched_inference_enabled": False,
        },
    }
    fast_payload = {
        **authoritative_payload,
        "policy_id": "train_u14_p15",
        "update_count": 14,
        "policy_version": 15,
        "evaluation_surface": {
            "kind": "fast_batched_screen",
            "authoritative": False,
            "batched_inference_enabled": True,
        },
    }

    summary_kind = persist_periodic_dev_eval_result(
        training_paths=training_paths,
        payload=authoritative_payload,
        b2_policy_id="b2_heuristic_public",
    )
    fast_kind = persist_periodic_dev_eval_result(
        training_paths=training_paths,
        payload=fast_payload,
        b2_policy_id="b2_heuristic_public",
    )
    forced_kind = persist_periodic_dev_eval_result(
        training_paths=training_paths,
        payload={**fast_payload, "policy_id": "train_u16_p17"},
        b2_policy_id="b2_heuristic_public",
        force_summary=True,
    )

    summaries = json.loads((training_paths.logs_dir / "periodic_dev_eval_summaries.json").read_text(encoding="utf-8"))
    fast_screens = json.loads(
        (training_paths.logs_dir / "periodic_dev_eval_fast_screens.json").read_text(encoding="utf-8")
    )
    assert summary_kind == "summary"
    assert fast_kind == "fast_screen"
    assert forced_kind == "summary"
    assert set(summaries) == {"train_u12_p13", "train_u16_p17"}
    assert set(fast_screens) == {"train_u14_p15"}


def test_eval_artifacts_promotion_gate_worker_records_group_by_anchor_then_block():
    payloads = [
        {"anchor_index": 1, "block_index": 0, "records": ("b1-block0",)},
        {"anchor_index": 0, "block_index": 1, "records": ("b0-block1",)},
        {"anchor_index": 0, "block_index": 0, "records": ("b0-block0",)},
    ]

    records_by_anchor = promotion_gate_records_by_anchor_index(worker_payloads=payloads, anchor_count=3)

    assert records_by_anchor == {
        0: ["b0-block0", "b0-block1"],
        1: ["b1-block0"],
        2: [],
    }


def _promotion_gate_eval_record(
    *,
    pair_index: int,
    swap_index: int,
    focal_policy_id: str = "candidate",
    opponent_policy_id: str = "anchor_a",
    truncated: bool = False,
) -> EvalGameRecord:
    focal_seat = 0 if swap_index == 0 else 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=pair_index * 2 + swap_index,
        episode_seed=100 + pair_index,
        episode_key=f"episode-{pair_index}-{swap_index}",
        episode_key64=pair_index * 2 + swap_index + 1,
        config_hash256="ab" * 32,
        spec_hash256="cd" * 32,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=focal_policy_id if focal_seat == 0 else opponent_policy_id,
        seat1_policy_id=opponent_policy_id if focal_seat == 0 else focal_policy_id,
        focal_seat=focal_seat,
        outcome="T" if truncated else "W",
        terminated=not truncated,
        truncated=truncated,
        engine_status=0,
        run_id256="ef" * 32,
    )


def test_promotion_artifacts_assemble_parallel_result_writes_anchor_episodes_and_record(tmp_path):
    stack = load_stack_config(canonical_stack_config_path())
    artifacts = SimpleNamespace(run_dir=tmp_path)
    anchors = (PromotionGateAnchor(name="B0 RandomLegal", policy_id="anchor_a"),)
    records_by_anchor = {
        0: [
            _promotion_gate_eval_record(pair_index=1, swap_index=1),
            _promotion_gate_eval_record(pair_index=0, swap_index=1),
            _promotion_gate_eval_record(pair_index=1, swap_index=0),
            _promotion_gate_eval_record(pair_index=0, swap_index=0),
        ]
    }

    result = assemble_parallel_promotion_gate_result(
        stack=stack,
        artifacts=artifacts,
        update_count=12,
        policy_version=13,
        focal_policy_id="candidate",
        anchors=anchors,
        records_by_anchor_index=records_by_anchor,
        paired_seeds=(101,),
        sample_count=64,
    )

    episodes_path = tmp_path / result.anchors[0].episodes_path
    record_path = tmp_path / "eval" / "promotion_gate" / "update_12" / stack.config.league.promotion_gate.record_file
    episode_rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    record_payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert [row["pair_index"] for row in episode_rows] == [0, 0, 1, 1]
    assert [row["swap_index"] for row in episode_rows] == [0, 1, 0, 1]
    assert result.ordered_opponents == ("B0 RandomLegal",)
    assert (
        result.anchors[0].episodes_path
        == "eval/promotion_gate/update_12/promotion_gate_episodes/00_b0_randomlegal.jsonl"
    )
    assert result.anchors[0].truncation.denominator == 4
    assert record_payload["focal_policy_id"] == "candidate"
    assert record_payload["ordered_opponents"] == ["B0 RandomLegal"]


def test_promotion_artifacts_policy_maps_split_snapshot_and_heuristic_opponents():
    snapshot_model = object()
    heuristic_policy = object()

    anchor_models, heuristic_policies = promotion_gate_policy_maps(
        (
            ("snapshot_a", "Snapshot A", snapshot_model, None),
            ("heuristic_b", "Heuristic B", None, heuristic_policy),
        )
    )

    assert anchor_models == {"snapshot_a": snapshot_model}
    assert heuristic_policies == {"heuristic_b": heuristic_policy}


def test_promotion_artifacts_console_messages_preserve_public_text():
    assert format_promotion_gate_discarded_after_rollback_message(
        candidate_policy_id="candidate",
        candidate_update=12,
        rollback_best_update=10,
    ) == (
        "Promotion gate result discarded after rollback: "
        "candidate=candidate candidate_update=12 rollback_best_update=10"
    )
    assert format_promotion_gate_skipped_league_warmup_message(
        update_count=7,
        effective_update=6,
        threshold=20,
        candidate_policy_id="candidate",
    ) == ("Promotion gate skipped during league warmup: update=7 effective_update=6 threshold=20 candidate=candidate")
    assert format_promotion_gate_skipped_eval_warmup_gate_message(
        update_count=21,
        effective_update=20,
        candidate_policy_id="candidate",
    ) == ("Promotion gate skipped during league eval warmup gate: update=21 effective_update=20 candidate=candidate")
    assert format_promotion_gate_missing_anchors_message(
        update_count=21,
        candidate_policy_id="candidate",
        missing_anchors=("B1 NoLeague baseline", "B2 HeuristicPublic"),
    ) == (
        "Promotion gate skipped: update=21 candidate=candidate missing_anchors=B1 NoLeague baseline,B2 HeuristicPublic"
    )
    assert format_scheduled_async_promotion_gate_message(
        update_count=24,
        candidate_policy_id="candidate",
        anchor_names=("B0 RandomLegal", "B1 NoLeague baseline"),
    ) == ("Scheduled async promotion gate: update=24 candidate=candidate anchors=B0 RandomLegal,B1 NoLeague baseline")
    assert format_optional_heuristic_public_anchors_skipped_message(RuntimeError("missing public ids")) == (
        "Promotion gate note: skipping optional heuristic-public anchors because the active simulator contract "
        "does not expose the required public action/observation metadata (missing public ids)."
    )


class _PromotionWorkerRunner:
    def run_game(self, scheduled_game):
        return GameResult(
            episode_seed=scheduled_game.episode_seed,
            terminated=True,
            truncated=False,
            winner_seat=scheduled_game.focal_seat,
        )


def test_promotion_artifacts_worker_payloads_preserve_seed_block_schema():
    opponent_spec = PeriodicDevEvalOpponentSpec(
        policy_id="anchor_a",
        display_name="Anchor A",
        kind="snapshot",
        snapshot_path="snapshots/anchor_a.pt",
    )
    job = PromotionGateSeedBlockJob(
        anchor_index=2,
        block_index=3,
        anchor_spec=opponent_spec,
        paired_seed_items=((0, 111), (1, 222)),
    )

    payloads = build_promotion_gate_worker_payloads(
        seed_block_jobs=(job,),
        runner=_PromotionWorkerRunner(),
        candidate_policy_id="candidate",
        run_id256="ab" * 32,
        config_hash256="cd" * 32,
        spec_hash256="ef" * 32,
    )

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["anchor_index"] == 2
    assert payload["block_index"] == 3
    assert payload["anchor_policy_id"] == "anchor_a"
    assert payload["anchor_display_name"] == "Anchor A"
    assert payload["paired_seed_items"] == ((0, 111), (1, 222))
    records = payload["records"]
    assert len(records) == 4
    assert [record.pair_index for record in records] == [0, 0, 1, 1]
    assert [record.swap_index for record in records] == [0, 1, 0, 1]
    assert {record.run_id256 for record in records} == {"ab" * 32}


def test_promotion_artifacts_parallel_plan_resolves_anchors_seeds_shards_and_devices(tmp_path):
    seed_file = tmp_path / "promotion_seeds.txt"
    seed_file.write_text("101\n202\n303\n", encoding="utf-8")
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"promotion_gate": seed_file},
        config=SimpleNamespace(
            league=SimpleNamespace(
                promotion_seed_file=seed_file.as_posix(),
                promotion_gate_paired_seeds=3,
                promotion_anchor_set_v1=SimpleNamespace(
                    required=("B0 RandomLegal", "B1 NoLeague baseline"),
                    optional_if_available=(),
                ),
                promotion_gate=SimpleNamespace(
                    parallel_workers=4,
                    parallel_worker_devices=(),
                ),
            )
        ),
    )
    anchor_specs = (
        PeriodicDevEvalOpponentSpec(policy_id="b0", display_name="B0 RandomLegal", kind="random_legal"),
        PeriodicDevEvalOpponentSpec(policy_id="b1", display_name="B1 NoLeague baseline", kind="snapshot"),
    )

    plan = build_parallel_promotion_gate_plan(
        stack=stack,
        anchor_policy_ids={"B0 RandomLegal": "b0", "B1 NoLeague baseline": "b1"},
        anchor_specs=anchor_specs,
        eval_device="cpu",
    )

    assert [anchor.name for anchor in plan.ordered_anchors] == ["B0 RandomLegal", "B1 NoLeague baseline"]
    assert plan.paired_seeds == (101, 202, 303)
    assert plan.worker_devices == ("cpu", "cpu", "cpu", "cpu")
    assert [[(job.anchor_index, job.block_index) for job in shard] for shard in plan.job_shards] == [
        [(0, 0)],
        [(0, 1)],
        [(1, 0)],
        [(1, 1)],
    ]


def test_eval_artifacts_b2_audit_request_prefers_canonical_config_and_logs_both_streams(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    episodes_path = tmp_path / "eval" / "dev_eval" / "update_40" / "B2 HeuristicPublic" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("{}\n", encoding="utf-8")
    canonical_config = tmp_path / "config_canonical.json"
    canonical_config.write_text("{}", encoding="utf-8")
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(json.dumps({"stack_config_path": "fallback.yaml"}), encoding="utf-8")
    artifacts = SimpleNamespace(run_dir=tmp_path, run_summary_path=run_summary_path)

    payload = maybe_request_b2_disagreement_audit(
        stack=_confidence_gate_stack(),
        training_paths=training_paths,
        artifacts=artifacts,
        dev_eval_summary=_authoritative_b2_summary(),
        b2_policy_id="B2 HeuristicPublic",
    )

    assert payload is not None
    assert payload["trigger_reasons"] == ["b2_flatline", "confidence_only_gate"]
    assert payload["command"][3] == canonical_config.as_posix()
    request_entries = [
        json.loads(line)
        for line in b2_disagreement_audit_requests_path(training_paths).read_text(encoding="utf-8").splitlines()
    ]
    guard_entries = [
        json.loads(line) for line in checkpoint_guard_log_path(training_paths).read_text(encoding="utf-8").splitlines()
    ]
    assert request_entries[-1] == payload
    assert guard_entries[-1] == payload
    assert format_b2_disagreement_audit_request_message(payload) == (
        "B2 disagreement audit requested: update=40 reasons=b2_flatline,confidence_only_gate "
        "episodes=eval/dev_eval/update_40/B2 HeuristicPublic/episodes.jsonl"
    )


def test_eval_artifacts_b2_audit_request_negative_cases_do_not_write(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(json.dumps({"stack_config_path": "fallback.yaml"}), encoding="utf-8")
    artifacts = SimpleNamespace(run_dir=tmp_path, run_summary_path=run_summary_path)

    missing_file_summary = _authoritative_b2_summary()
    assert (
        maybe_request_b2_disagreement_audit(
            stack=_confidence_gate_stack(),
            training_paths=training_paths,
            artifacts=artifacts,
            dev_eval_summary=missing_file_summary,
            b2_policy_id="B2 HeuristicPublic",
        )
        is None
    )

    non_authoritative_summary = _authoritative_b2_summary()
    non_authoritative_summary["evaluation_surface"] = {"authoritative": False}
    assert (
        maybe_request_b2_disagreement_audit(
            stack=_confidence_gate_stack(),
            training_paths=training_paths,
            artifacts=artifacts,
            dev_eval_summary=non_authoritative_summary,
            b2_policy_id="B2 HeuristicPublic",
        )
        is None
    )
    assert not b2_disagreement_audit_requests_path(training_paths).exists()


def test_eval_artifacts_structured_mainmove_guard_logs_low_b2_warning(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    learner = SimpleNamespace(update_count=12, get_policy_version=lambda: 13)

    payload = maybe_log_structured_mainmove_guard(
        training_paths=training_paths,
        learner=learner,
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.5,
            "structured_main_move_share_when_play_available": 0.6,
        },
        dev_eval_summary={
            "aggregate_score": 0.6,
            "anchor_scores": {"B2 HeuristicPublic": 0.05},
        },
    )

    assert payload is not None
    assert payload["event_kind"] == "structured_mainmove_warning_v1"
    assert payload["b2_anchor_score"] == pytest.approx(0.05)
    entries = [
        json.loads(line) for line in checkpoint_guard_log_path(training_paths).read_text(encoding="utf-8").splitlines()
    ]
    assert entries == [payload]


def test_eval_artifacts_structured_mainmove_guard_skips_when_b2_is_healthy(tmp_path):
    training_paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    learner = SimpleNamespace(update_count=12, get_policy_version=lambda: 13)

    payload = maybe_log_structured_mainmove_guard(
        training_paths=training_paths,
        learner=learner,
        latest_metrics={
            "structured_main_move_0_2_top1_rate": 0.5,
            "structured_main_move_share_when_play_available": 0.6,
        },
        dev_eval_summary={
            "aggregate_score": 0.6,
            "anchor_scores": {"B2 HeuristicPublic": 0.2},
        },
    )

    assert payload is None
    assert not checkpoint_guard_log_path(training_paths).exists()


def test_eval_schedule_helpers_resolve_seeds_and_filter_anchor_weights(tmp_path):
    seed_file = tmp_path / "dev_eval.txt"
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"dev_eval": seed_file},
        config=SimpleNamespace(
            evaluation=SimpleNamespace(
                seat_swap=True,
                eval_inference_mode=True,
                eval_sampling_algorithm="pinned_cdf_pcg_v1",
                periodic_dev_eval_paired_seeds=2,
                periodic_dev_eval_interval_updates=10,
                seed_files={"dev_eval": "dev_eval.txt"},
                periodic_dev_eval_anchor_weights={
                    "B1 NoLeague baseline": 3,
                    "": 1,
                    "bad": "nope",
                    "negative": -1,
                    "bool": True,
                },
            ),
            reproducibility=SimpleNamespace(seed_files={"dev_eval": "dev_eval.txt"}),
            league=None,
        ),
    )

    resolved_seed_file, sources, paired_seeds, seed_hash = periodic_dev_eval_schedule(stack)

    assert resolved_seed_file == seed_file
    assert sources == {
        "stack.seed_sets.dev_eval": "dev_eval.txt",
        "evaluation.seed_files.dev_eval": "dev_eval.txt",
        "reproducibility.seed_files.dev_eval": "dev_eval.txt",
    }
    assert paired_seeds == [11, 22]
    assert len(seed_hash) == 64
    assert periodic_dev_eval_anchor_weight_map(stack) == {"B1 NoLeague baseline": 3.0}
    assert (
        should_defer_noleague_baseline_alias_refresh(
            stack=stack,
            experiment_role="baseline_noleague_variant",
            update_count=20,
        )
        is True
    )


def test_eval_schedule_helpers_report_warmup_gate_failures():
    stack = SimpleNamespace(
        config=SimpleNamespace(
            league=SimpleNamespace(
                enabled=True,
                warmup=SimpleNamespace(
                    eval_gate_enabled=True,
                    eval_gate_min_aggregate_score=0.5,
                    eval_gate_min_anchor_scores={
                        "B1 NoLeague baseline": 0.45,
                        "Previous recent snapshot": 0.9,
                    },
                ),
            )
        )
    )

    status = league_eval_warmup_gate_status(
        stack,
        {
            "aggregate_score": 0.6,
            "anchor_scores": {
                "B1 NoLeague baseline": 0.4,
            },
        },
    )

    assert status["enabled"] is True
    assert status["open"] is False
    assert status["reasons"] == ["anchor_scores"]
    assert status["failed_anchors"] == {
        "B1 NoLeague baseline": {
            "score": 0.4,
            "min_score": 0.45,
        }
    }
    assert format_league_eval_warmup_gate_message(status) == (
        "League eval warmup gate: open=False reasons=anchor_scores"
    )


def test_eval_schedule_seed_block_jobs_and_schedules_are_stable():
    opponents = (
        PeriodicDevEvalOpponentSpec(policy_id="anchor_a", display_name="Anchor A", kind="snapshot"),
        PeriodicDevEvalOpponentSpec(policy_id="anchor_a", display_name="Anchor A duplicate", kind="snapshot"),
    )

    jobs = build_periodic_dev_eval_seed_block_jobs(
        opponent_specs=opponents,
        paired_seeds=[101, 202, 303],
        configured_parallel_workers=4,
    )
    promotion_jobs = build_promotion_gate_seed_block_jobs(
        anchor_specs=opponents,
        paired_seeds=[101, 202, 303],
        configured_parallel_workers=4,
    )
    schedule = periodic_dev_eval_schedule_for_seed_items(
        focal_policy_id="candidate",
        opponent_policy_id="anchor_a",
        paired_seed_items=jobs[0].paired_seed_items,
    )

    assert periodic_dev_eval_duplicate_policy_ids(opponents) == {"anchor_a"}
    assert [(job.opponent_index, job.block_index, job.paired_seed_items) for job in jobs] == [
        (0, 0, ((0, 101), (1, 202))),
        (0, 1, ((2, 303),)),
        (1, 0, ((0, 101), (1, 202))),
        (1, 1, ((2, 303),)),
    ]
    assert [(job.anchor_index, job.block_index, job.paired_seed_items) for job in promotion_jobs] == [
        (0, 0, ((0, 101), (1, 202))),
        (0, 1, ((2, 303),)),
        (1, 0, ((0, 101), (1, 202))),
        (1, 1, ((2, 303),)),
    ]
    assert [
        [job.block_index for job in shard]
        for shard in shard_periodic_dev_eval_seed_block_jobs(jobs=jobs, shard_count=2)
    ] == [
        [0, 0],
        [1, 1],
    ]
    assert [
        [job.block_index for job in shard]
        for shard in shard_promotion_gate_seed_block_jobs(jobs=promotion_jobs, shard_count=2)
    ] == [
        [0, 0],
        [1, 1],
    ]
    assert [(game.swap_index, game.seat0_policy_id, game.seat1_policy_id) for game in schedule] == [
        (0, "candidate", "anchor_a"),
        (1, "anchor_a", "candidate"),
        (0, "candidate", "anchor_a"),
        (1, "anchor_a", "candidate"),
    ]


def test_eval_schedule_async_request_builders_normalize_payloads(tmp_path):
    stack = SimpleNamespace(config=SimpleNamespace())
    opponents = (PeriodicDevEvalOpponentSpec(policy_id="anchor", display_name="Anchor", kind="heuristic_public"),)

    periodic = build_async_periodic_dev_eval_request(
        stack=stack,
        checkpoint_path=tmp_path / "checkpoint.pt",
        focal_policy_id="candidate",
        update_count="12",
        policy_version="13",
        run_dir=tmp_path,
        run_id256=123,
        config_hash256=456,
        spec_hash256=789,
        artifact_dir_name="dev_eval",
        artifact_scope="periodic_dev_eval",
        paired_seeds=["1", 2],
        opponents=opponents,
        eval_device_override=None,
        parallel_workers=0,
        parallel_worker_devices=("cpu",),
    )
    promotion = build_async_promotion_gate_request(
        stack=stack,
        run_dir=tmp_path,
        candidate_policy_id="candidate",
        candidate_snapshot_path="training/snapshots/candidate/weights.pt",
        update_count="12",
        policy_version="13",
        run_id256=123,
        config_hash256=456,
        spec_hash256=789,
        anchor_policy_ids={"Anchor": "anchor"},
        anchor_specs=opponents,
        eval_device_override="cpu",
    )

    assert periodic.update_count == 12
    assert periodic.policy_version == 13
    assert periodic.run_id256 == "123"
    assert periodic.paired_seeds == (1, 2)
    assert periodic.parallel_workers == 1
    assert periodic.parallel_worker_devices == ("cpu",)
    assert periodic.opponents == opponents
    assert promotion.update_count == 12
    assert promotion.policy_version == 13
    assert promotion.anchor_policy_ids == {"Anchor": "anchor"}
    assert promotion.anchor_specs == opponents


def test_eval_schedule_periodic_worker_devices_use_actor_layout(monkeypatch):
    stack = SimpleNamespace(config=SimpleNamespace(system=SimpleNamespace(actor_process_count=4)))
    monkeypatch.setattr("weiss_rl.training.eval_schedule.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("weiss_rl.training.eval_schedule.torch.cuda.device_count", lambda: 3)

    resolved = resolved_periodic_dev_eval_worker_devices(
        stack=stack,
        parallel_workers=4,
        explicit_worker_devices=(),
        eval_device="cuda:auto",
        learner_device=SimpleNamespace(type="cuda"),
        actor_device_layout_resolver=lambda *args, **kwargs: ("cuda:1", "cuda:2"),
    )

    assert resolved == ("cuda:1", "cuda:2", "cuda:1", "cuda:2")
