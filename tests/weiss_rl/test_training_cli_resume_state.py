from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.train_entrypoint.cli import _require_explicit_resume_geometry, resolve_train_cli_state


def test_resume_requires_explicit_runtime_geometry() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    args = SimpleNamespace(
        resume_run_dir=object(),
        resume_from="latest",
        num_envs=None,
        unroll_length=None,
        runtime_mode=None,
        profile=None,
    )

    try:
        _require_explicit_resume_geometry(parser, args)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("resume without explicit geometry should fail")

    assert "--num-envs" in message
    assert "--unroll-length" in message
    assert "--runtime-mode" in message
    assert "--profile" in message


def test_resume_accepts_explicit_runtime_geometry() -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    args = SimpleNamespace(
        resume_run_dir=object(),
        resume_from="latest",
        num_envs=32,
        unroll_length=16,
        runtime_mode="train_async_fast",
        profile="fast",
    )

    _require_explicit_resume_geometry(parser, args)


def test_train_cli_state_rejects_checkpoint_init_with_resume(tmp_path: Path) -> None:
    parser = SimpleNamespace(error=lambda message: (_ for _ in ()).throw(RuntimeError(message)))
    stack = SimpleNamespace(
        config=SimpleNamespace(training=object()),
    )
    args = SimpleNamespace(
        run_label="run_a",
        run_id_alias="",
        resume_run_dir=tmp_path / "runs" / "source",
        resume_from="",
        num_envs=2,
        unroll_length=4,
        runtime_mode="train_ordered",
        profile="default",
        max_updates=1,
        stack_config=tmp_path / "stack.yaml",
        config_override=(),
        profile_timers=False,
        torch_profiler=False,
        init_from_checkpoint=tmp_path / "checkpoint.pt",
        init_schedule_offset_updates=None,
        public_demo=False,
    )
    api = SimpleNamespace(
        QueueRuntimeMode=str,
        _resolve_run_label=lambda parser, run_label, run_id_alias: run_label,
        _require_positive_int=lambda _flag, value: int(value),
        load_stack_config=lambda _path: stack,
        apply_stack_overrides=lambda loaded_stack, _overrides: loaded_stack,
        parse_override_tokens=lambda _tokens: (),
        _apply_training_flag_overrides=lambda loaded_stack, **_kwargs: loaded_stack,
        _manifest_scaffold_only_reason=lambda _stack: None,
        _resolve_resume_checkpoint_path=lambda *, resume_from, resume_run_dir: None,
    )

    try:
        resolve_train_cli_state(parser=parser, args=args, api=api)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("checkpoint init combined with resume should fail")

    assert "--init-from-checkpoint starts a fresh run" in message
