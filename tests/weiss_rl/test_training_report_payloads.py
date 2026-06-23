from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.report_payloads import (
    augment_determinism_payload,
    augment_environment_payload,
    augment_run_summary_payload,
    profiling_enabled_message,
    training_controls_payload,
)


def _training_config() -> SimpleNamespace:
    return SimpleNamespace(
        profile_timers=True,
        torch_profiler=False,
        structured_metrics_mode="compact",
        teacher_aux_mode="teacher",
        fixed_opponent_backend="heuristic",
        actor_policy_backend="model",
        actor_heuristic_fraction=0.25,
        actor_heuristic_final_fraction=0.0,
        actor_sampling_temperature=0.25,
        train_on_heuristic_actor_rows=False,
    )


def test_training_controls_payload_preserves_report_field_names_and_string_modes() -> None:
    assert training_controls_payload(None) is None

    payload = training_controls_payload(_training_config())

    assert payload == {
        "profile_timers": True,
        "torch_profiler": False,
        "structured_metrics_mode": "compact",
        "teacher_aux_mode": "teacher",
        "fixed_opponent_backend": "heuristic",
        "actor_policy_backend": "model",
        "actor_heuristic_fraction": 0.25,
        "actor_heuristic_final_fraction": 0.0,
        "actor_sampling_temperature": 0.25,
        "train_on_heuristic_actor_rows": False,
    }


def test_profiling_enabled_message_matches_train_entrypoint_text() -> None:
    assert (
        profiling_enabled_message(
            SimpleNamespace(
                profile_timers=False,
                torch_profiler=False,
                structured_metrics_mode="compact",
                teacher_aux_mode="teacher",
                fixed_opponent_backend="heuristic",
                actor_policy_backend="model",
                actor_heuristic_fraction=0.25,
                actor_heuristic_final_fraction=0.0,
                actor_sampling_temperature=0.25,
                train_on_heuristic_actor_rows=False,
            )
        )
        is None
    )

    assert profiling_enabled_message(_training_config()) == (
        "Structured profiling enabled: "
        "profile_timers=True "
        "torch_profiler=False "
        "structured_metrics_mode=compact "
        "teacher_aux_mode=teacher "
        "fixed_opponent_backend=heuristic"
    )


def test_augment_run_summary_payload_records_runtime_policy_inputs_and_resume_paths(tmp_path: Path) -> None:
    b1_dir = tmp_path / "b1"
    seed_dir = tmp_path / "seed"
    init_checkpoint = tmp_path / "seed" / "best.pt"
    resume_dir = tmp_path / "resume"
    checkpoint_path = resume_dir / "checkpoints" / "latest.pt"

    payload = augment_run_summary_payload(
        {"existing": True},
        public_demo_enabled=False,
        runtime_mode="train_ordered",
        policy_set_selection_details={"mode": "deterministic_v1"},
        training_config=_training_config(),
        b1_baseline_run_dir=b1_dir,
        seed_snapshot_run_dir=seed_dir,
        init_from_checkpoint_path=init_checkpoint,
        resume_run_dir=resume_dir,
        resume_checkpoint_path=checkpoint_path,
    )

    assert payload["existing"] is True
    assert payload["runtime_mode"] == "train_ordered"
    assert payload["policy_set_selection_mode"] == "deterministic_v1"
    assert payload["training_controls"] == training_controls_payload(_training_config())
    assert payload["b1_baseline_run_dir"] == b1_dir.resolve().as_posix()
    assert payload["seed_snapshot_run_dir"] == seed_dir.resolve().as_posix()
    assert payload["init_from_checkpoint_path"] == init_checkpoint.resolve().as_posix()
    assert payload["resume"] == {
        "enabled": True,
        "resume_run_dir": resume_dir.as_posix(),
        "resume_checkpoint_path": checkpoint_path.as_posix(),
    }


def test_augment_run_summary_payload_uses_public_demo_runtime_and_default_selection_mode() -> None:
    payload = augment_run_summary_payload(
        {},
        public_demo_enabled=True,
        runtime_mode="train_ordered",
        policy_set_selection_details={},
        training_config=None,
        b1_baseline_run_dir=None,
        seed_snapshot_run_dir=None,
        init_from_checkpoint_path=None,
        resume_run_dir=None,
        resume_checkpoint_path=None,
    )

    assert payload == {
        "runtime_mode": "public_demo",
        "policy_set_selection_mode": "unresolved",
    }


def test_augment_determinism_payload_uses_historical_policy_selection_key(tmp_path: Path) -> None:
    b1_dir = tmp_path / "b1"
    seed_dir = tmp_path / "seed"
    init_checkpoint = tmp_path / "seed" / "best.pt"
    checkpoint_path = tmp_path / "run" / "checkpoints" / "latest.pt"

    payload = augment_determinism_payload(
        {},
        public_demo_enabled=False,
        runtime_mode="train_ordered",
        policy_set_selection_details={"mode": "registry"},
        training_config=_training_config(),
        b1_baseline_run_dir=b1_dir,
        seed_snapshot_run_dir=seed_dir,
        init_from_checkpoint_path=init_checkpoint,
        resume_checkpoint_path=checkpoint_path,
    )

    assert payload["runtime_mode"] == "train_ordered"
    assert payload["policy_selection_mode"] == "registry"
    assert payload["training_controls"] == training_controls_payload(_training_config())
    assert payload["b1_baseline_run_dir"] == b1_dir.resolve().as_posix()
    assert payload["seed_snapshot_run_dir"] == seed_dir.resolve().as_posix()
    assert payload["init_from_checkpoint_path"] == init_checkpoint.resolve().as_posix()
    assert payload["resume_checkpoint_path"] == checkpoint_path.as_posix()


def test_augment_environment_payload_records_cwd_argv_hardware_and_resume_path(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "latest.pt"
    hardware = {"device": "cpu"}

    payload = augment_environment_payload(
        {},
        root=tmp_path,
        argv=("python", "train.py"),
        hardware=hardware,
        init_from_checkpoint_path=checkpoint_path,
        resume_checkpoint_path=checkpoint_path,
    )

    assert payload == {
        "cwd": tmp_path.as_posix(),
        "argv": ["python", "train.py"],
        "hardware": hardware,
        "init_from_checkpoint_path": checkpoint_path.as_posix(),
        "resume_checkpoint_path": checkpoint_path.as_posix(),
    }
