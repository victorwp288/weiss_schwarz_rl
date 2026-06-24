from __future__ import annotations

from .entrypoints_test_support import (
    Path,
    pytest,
)


def test_eval_startup_validation_preserves_mode_errors() -> None:
    from weiss_rl.workflows.eval_support.parser.eval_parser import build_eval_parser
    from weiss_rl.workflows.eval_support.startup.eval_startup import validate_eval_args

    parser = build_eval_parser()
    args = parser.parse_args(
        [
            "--stack-config",
            "configs/presets/structured_acceptance_standard_thesis_eval.yaml",
            "--run-dir",
            "runs/demo",
            "--episodes-jsonl",
            "runs/demo/eval/final_eval/episodes.jsonl",
        ]
    )

    with pytest.raises(SystemExit):
        validate_eval_args(parser=parser, args=args)


def test_eval_startup_route_payload_names_identity_handoff_steps() -> None:
    from weiss_rl.workflows.eval_support.startup.eval_startup import eval_startup_route_payload

    route = eval_startup_route_payload()
    step_ids = {step["step_id"] for step in route}

    assert {"load_stack", "verify_config_identity", "select_spec_source", "announce_startup"} <= step_ids
    for step in route:
        assert step["title"]
        assert step["route"]
        assert step["output"]


def test_eval_startup_preparation_uses_public_demo_contract_and_banner() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.startup.eval_startup import EvalStartupDependencies, prepare_eval_startup

    observed: dict[str, object] = {}
    stack = SimpleNamespace(root=Path("repo"))
    args = SimpleNamespace(
        stack_config=Path("configs/demo.yaml"),
        config_hash="",
        public_demo=True,
        spec_hash="",
    )

    def fake_banner(
        reported_spec_hash: str,
        config_hash256: str,
        *,
        run_label: str,
        spec_mismatch_policy: str,
    ) -> None:
        observed["banner"] = (reported_spec_hash, config_hash256, run_label, spec_mismatch_policy)

    startup = prepare_eval_startup(
        args=args,
        run_label="demo_eval",
        dependencies=EvalStartupDependencies(
            load_stack_config_fn=lambda path: stack,
            compute_config_hash256_fn=lambda loaded_stack: "c" * 64,
            expected_sha256_fn=lambda value, *, flag_name: "",
            require_matching_hash_fn=lambda **kwargs: observed.setdefault("hash", kwargs),
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            assert_spec_bundle_contract_fn=lambda expected, bundle: observed.setdefault("spec", (expected, bundle)),
            public_demo_spec_hash256_fn=lambda: "d" * 64,
            load_verified_simulator_contract_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("public demo must not load simulator contract")
            ),
            print_startup_banner_fn=fake_banner,
        ),
    )

    assert startup.stack is stack
    assert startup.config_hash256 == "c" * 64
    assert startup.reported_spec_hash == "d" * 64
    assert startup.contract is None
    assert observed["hash"] == {"flag_name": "--config-hash", "expected": "", "actual": "c" * 64}
    assert observed["spec"] == ("", {"spec_hash": "public_demo"})
    assert observed["banner"] == ("d" * 64, "c" * 64, "demo_eval", "hard_fail")


def test_eval_startup_preparation_uses_verified_simulator_contract() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.startup.eval_startup import EvalStartupDependencies, prepare_eval_startup

    observed: dict[str, object] = {}
    stack = SimpleNamespace(root=Path("repo"))
    contract = SimpleNamespace(spec_hash256="e" * 64)
    args = SimpleNamespace(
        stack_config=Path("configs/thesis.yaml"),
        config_hash="c" * 64,
        public_demo=False,
        spec_hash="e" * 64,
    )

    def fake_load_verified_simulator_contract(*args: object, **kwargs: object) -> object:
        observed["contract_call"] = (args, kwargs)
        return contract

    startup = prepare_eval_startup(
        args=args,
        run_label="canonical_eval",
        dependencies=EvalStartupDependencies(
            load_stack_config_fn=lambda path: stack,
            compute_config_hash256_fn=lambda loaded_stack: "c" * 64,
            expected_sha256_fn=lambda value, *, flag_name: value,
            require_matching_hash_fn=lambda **kwargs: observed.setdefault("hash", kwargs),
            public_demo_spec_bundle_fn=lambda: (_ for _ in ()).throw(
                AssertionError("canonical startup must not load public-demo spec")
            ),
            assert_spec_bundle_contract_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("canonical startup must not assert public-demo contract")
            ),
            public_demo_spec_hash256_fn=lambda: (_ for _ in ()).throw(
                AssertionError("canonical startup must not report public-demo hash")
            ),
            load_verified_simulator_contract_fn=fake_load_verified_simulator_contract,
            print_startup_banner_fn=lambda *args, **kwargs: observed.setdefault("banner", (args, kwargs)),
        ),
    )

    assert startup.stack is stack
    assert startup.config_hash256 == "c" * 64
    assert startup.reported_spec_hash == "e" * 64
    assert startup.contract is contract
    assert observed["hash"] == {"flag_name": "--config-hash", "expected": "c" * 64, "actual": "c" * 64}
    assert observed["contract_call"] == ((Path("repo"),), {"expected_spec_hash": "e" * 64})
    assert observed["banner"] == (
        ("e" * 64, "c" * 64),
        {"run_label": "canonical_eval", "spec_mismatch_policy": "hard_fail"},
    )
