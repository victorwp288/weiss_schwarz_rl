from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.train_entrypoint.cli import TrainCliState, prepare_train_startup_state
from weiss_rl.training.train_entrypoint.startup_state import train_startup_route_payload


def test_train_startup_route_payload_names_identity_handoff_steps() -> None:
    route = train_startup_route_payload()
    step_ids = {step["step_id"] for step in route}

    assert {
        "select_spec_source",
        "verify_config_identity",
        "capture_run_provenance",
        "assign_run_identity",
        "announce_startup",
    } <= step_ids
    for step in route:
        assert step["title"]
        assert step["route"]
        assert step["output"]


def test_train_startup_state_uses_public_demo_contract_without_runtime_load(tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    stack = SimpleNamespace(root=tmp_path, components=[object(), object()])
    cli = TrainCliState(
        run_label="toy_demo",
        num_envs=2,
        unroll_length=4,
        max_updates=1,
        runtime_mode="train_ordered",
        stack=stack,
        training_config=object(),
        manifest_only_reason=None,
        public_demo_enabled=True,
        resume_run_dir=None,
        resume_checkpoint_path=None,
        init_from_checkpoint_path=None,
        init_schedule_offset_override_updates=None,
    )
    args = SimpleNamespace(spec_hash="", config_hash="")
    run_identity = SimpleNamespace(run_id256="a" * 64, run_id64="a" * 16, run_dir_name="toy_demo")

    def fake_banner(
        spec_hash256: str,
        config_hash256: str,
        *,
        run_id64: str,
        run_id256: str,
        run_label: str,
        run_dir_name: str,
        spec_mismatch_policy: str,
    ) -> None:
        calls["banner"] = (
            spec_hash256,
            config_hash256,
            run_id64,
            run_id256,
            run_label,
            run_dir_name,
            spec_mismatch_policy,
        )

    def fake_new_run_identity(**kwargs: object) -> object:
        calls["identity"] = kwargs
        return run_identity

    api = SimpleNamespace(
        public_demo_spec_bundle=lambda: {"action": {"pass_action_id": 8}, "spec_hash": "toy"},
        assert_spec_bundle_contract=lambda expected, bundle: calls.setdefault("spec", (expected, bundle)),
        public_demo_spec_hash256=lambda: "b" * 64,
        public_demo_simulator_info=lambda: {"compatibility_hash": "public_demo"},
        load_verified_simulator_contract=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public demo startup must not load the runtime simulator contract")
        ),
        compute_config_hash256=lambda loaded_stack: "c" * 64,
        _expected_sha256=lambda value, *, flag_name: "",
        _require_matching_hash=lambda **kwargs: calls.setdefault("hash", kwargs),
        _git_commit=lambda: "d" * 40,
        _start_nonce=lambda: "nonce",
        new_run_identity=fake_new_run_identity,
        _run_artifacts_from_existing_run_dir=lambda _path: (_ for _ in ()).throw(
            AssertionError("fresh public demo startup should not resolve resume artifacts")
        ),
        resume_run_identity=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fresh public demo startup should not resume identity")
        ),
        _load_json_object=lambda *_args, **_kwargs: {},
        print_startup_banner=fake_banner,
        _spec_mismatch_policy=lambda loaded_stack: "hard_fail",
    )

    startup = prepare_train_startup_state(parser=SimpleNamespace(), args=args, api=api, cli=cli)

    assert startup.simulator_contract is None
    assert startup.spec_hash256 == "b" * 64
    assert startup.config_hash256 == "c" * 64
    assert startup.run_id256 == "a" * 64
    assert startup.run_id64 == "a" * 16
    assert startup.resume_artifacts is None
    assert calls["spec"] == ("", {"action": {"pass_action_id": 8}, "spec_hash": "toy"})
    assert calls["hash"] == {"flag_name": "--config-hash", "expected": "", "actual": "c" * 64}
    assert calls["identity"]["run_label"] == "toy_demo"
    assert calls["banner"] == ("b" * 64, "c" * 64, "a" * 16, "a" * 64, "toy_demo", "toy_demo", "hard_fail")
