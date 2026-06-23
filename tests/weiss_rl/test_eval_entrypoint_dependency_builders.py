from __future__ import annotations

from .entrypoints_test_support import (
    Path,
)


def test_eval_entrypoint_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    class FakeTensorBoardLogger:
        pass

    def fake_run_final_eval(**_kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(eval_script, "TensorBoardLogger", FakeTensorBoardLogger)
    monkeypatch.setattr(eval_script, "run_final_eval", fake_run_final_eval)

    dependencies = eval_script._canonical_eval_dependencies()

    assert dependencies.tensorboard_logger_cls is FakeTensorBoardLogger
    assert dependencies.run_final_eval_fn is fake_run_final_eval


def test_eval_dispatch_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    def fake_public_demo_eval_mode(**_kwargs: object) -> None:
        return None

    def fake_summary_json(_path: Path, _payload: object) -> None:
        return None

    monkeypatch.setattr(eval_script, "run_public_demo_eval_mode", fake_public_demo_eval_mode)
    monkeypatch.setattr(eval_script, "write_matchup_summary_json", fake_summary_json)

    dependencies = eval_script._eval_dispatch_dependencies()

    assert dependencies.run_public_demo_eval_mode_fn is fake_public_demo_eval_mode
    assert dependencies.write_matchup_summary_json_fn is fake_summary_json


def test_eval_startup_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    def fake_load_stack_config(_path: Path) -> object:
        return object()

    def fake_banner(
        _reported_spec_hash: str,
        _config_hash256: str,
        *,
        run_label: str,
        spec_mismatch_policy: str,
    ) -> None:
        assert run_label
        assert spec_mismatch_policy

    monkeypatch.setattr(eval_script, "load_stack_config", fake_load_stack_config)
    monkeypatch.setattr(eval_script, "print_startup_banner", fake_banner)

    dependencies = eval_script._eval_startup_dependencies()

    assert dependencies.load_stack_config_fn is fake_load_stack_config
    assert dependencies.print_startup_banner_fn is fake_banner
