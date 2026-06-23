from __future__ import annotations

from types import SimpleNamespace

from .entrypoints_test_support import Path
from .eval_dispatch_test_support import dispatch_dependencies, parser, public_demo_startup, seed_stack, validated_args


def test_eval_dispatch_routes_public_demo_with_resolved_paths(tmp_path: Path, capsys) -> None:
    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch

    observed: dict[str, object] = {}
    stack = seed_stack(tmp_path)
    args = SimpleNamespace(
        public_demo=True,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "custom_eval",
        public_demo_paired_seeds=7,
        public_demo_bootstrap_samples=11,
        run_label="demo",
    )

    run_eval_dispatch(
        parser=parser(),
        args=args,
        validated=validated_args("demo"),
        startup=public_demo_startup(stack),
        dependencies=dispatch_dependencies(
            public_demo_stop_rules_fn=lambda: "stop_rules",
            run_public_demo_final_eval_fn=lambda **_kwargs: {"policy_ids": []},
            run_public_demo_eval_mode_fn=lambda **kwargs: observed.setdefault("public_demo", kwargs),
        ),
    )

    call = observed["public_demo"]
    assert call["stack"] is stack
    assert call["run_dir"] == (tmp_path / "run").resolve()
    assert call["final_eval_dir"] == (tmp_path / "custom_eval").resolve()
    assert call["paired_seed_limit"] == 7
    assert call["bootstrap_samples"] == 11
    assert call["config_hash256"] == "c" * 64
    assert call["spec_hash256"] == "d" * 64
    assert "Verified public-demo spec bundle" in capsys.readouterr().out
