from __future__ import annotations


def test_eval_entrypoint_main_runner_threads_startup_and_dispatch() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_entrypoint_support.main import run_eval_entrypoint_main

    calls: list[str] = []
    parsed_args = SimpleNamespace(kind="args")
    validated_state = SimpleNamespace(run_label="eval_label")
    startup_state = SimpleNamespace(kind="startup")
    startup_dependencies = SimpleNamespace(kind="startup_deps")
    dispatch_dependencies = SimpleNamespace(kind="dispatch_deps")

    class FakeParser:
        def parse_args(self) -> object:
            calls.append("parse_args")
            return parsed_args

    parser_obj = FakeParser()

    def fake_build_parser() -> FakeParser:
        calls.append("build_parser")
        return parser_obj

    def fake_validate(*, parser: object, args: object) -> object:
        calls.append("validate")
        assert parser is parser_obj
        assert args is parsed_args
        return validated_state

    def fake_startup_dependencies() -> object:
        calls.append("startup_dependencies")
        return startup_dependencies

    def fake_prepare_startup(*, args: object, run_label: str, dependencies: object) -> object:
        calls.append("prepare_startup")
        assert args is parsed_args
        assert run_label == "eval_label"
        assert dependencies is startup_dependencies
        return startup_state

    def fake_dispatch_dependencies() -> object:
        calls.append("dispatch_dependencies")
        return dispatch_dependencies

    def fake_dispatch(
        *,
        parser: object,
        args: object,
        validated: object,
        startup: object,
        dependencies: object,
    ) -> None:
        calls.append("dispatch")
        assert parser is parser_obj
        assert args is parsed_args
        assert validated is validated_state
        assert startup is startup_state
        assert dependencies is dispatch_dependencies

    run_eval_entrypoint_main(
        build_eval_parser_fn=fake_build_parser,
        validate_eval_args_fn=fake_validate,
        prepare_eval_startup_fn=fake_prepare_startup,
        run_eval_dispatch_fn=fake_dispatch,
        eval_startup_dependencies_fn=fake_startup_dependencies,
        eval_dispatch_dependencies_fn=fake_dispatch_dependencies,
    )

    assert calls == [
        "build_parser",
        "parse_args",
        "validate",
        "startup_dependencies",
        "prepare_startup",
        "dispatch_dependencies",
        "dispatch",
    ]


def test_eval_entrypoint_runtime_threads_facade_globals(monkeypatch) -> None:
    from types import SimpleNamespace

    import weiss_rl.workflows.eval_entrypoint_support.runtime as eval_entrypoint_runtime

    calls: list[str] = []
    parser_obj = SimpleNamespace(kind="parser")
    startup_dependencies = SimpleNamespace(kind="startup_deps")
    dispatch_dependencies = SimpleNamespace(kind="dispatch_deps")

    def fake_main(**kwargs: object) -> None:
        calls.append("main")
        assert kwargs["build_eval_parser_fn"]() is parser_obj
        assert kwargs["eval_startup_dependencies_fn"]() is startup_dependencies
        assert kwargs["eval_dispatch_dependencies_fn"]() is dispatch_dependencies

    def fake_startup_dependencies(entrypoint_globals: object) -> object:
        calls.append("startup_dependencies")
        assert entrypoint_globals is globals_map
        return startup_dependencies

    def fake_dispatch_dependencies(entrypoint_globals: object) -> object:
        calls.append("dispatch_dependencies")
        assert entrypoint_globals is globals_map
        return dispatch_dependencies

    monkeypatch.setattr(eval_entrypoint_runtime, "run_eval_entrypoint_main", fake_main)
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_startup_dependencies",
        fake_startup_dependencies,
    )
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_dispatch_dependencies",
        fake_dispatch_dependencies,
    )
    globals_map = {
        "build_eval_parser": lambda: parser_obj,
        "validate_eval_args": object(),
        "prepare_eval_startup": object(),
        "run_eval_dispatch": object(),
    }

    eval_entrypoint_runtime.run_eval_entrypoint(entrypoint_globals=globals_map)

    assert calls == ["main", "startup_dependencies", "dispatch_dependencies"]
