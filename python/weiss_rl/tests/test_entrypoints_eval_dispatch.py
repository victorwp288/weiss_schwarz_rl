from __future__ import annotations

from .test_entrypoints import (
    Path,
    _copy_repo_configs,
    _mismatched_sha256,
    _run_entrypoint,
    _run_public_demo_train,
    _write_manifest_only_stack_config,
    _write_stub_weiss_sim,
    argparse,
    compute_config_hash256,
    json,
    load_stack_config,
    public_demo_spec_hash256,
    pytest,
    spec_bundle_hash,
)


def test_eval_startup_validation_preserves_mode_errors() -> None:
    from weiss_rl.workflows.eval_support.eval_parser import build_eval_parser
    from weiss_rl.workflows.eval_support.eval_startup import validate_eval_args

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


def test_eval_startup_preparation_uses_public_demo_contract_and_banner() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_startup import EvalStartupDependencies, prepare_eval_startup

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

    from weiss_rl.workflows.eval_support.eval_startup import EvalStartupDependencies, prepare_eval_startup

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


def test_eval_dispatch_routes_public_demo_with_resolved_paths(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    observed: dict[str, object] = {}
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=True,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "custom_eval",
        public_demo_paired_seeds=7,
        public_demo_bootstrap_samples=11,
        run_label="demo",
    )

    run_eval_dispatch(
        parser=argparse.ArgumentParser(),
        args=args,
        validated=EvalValidatedArgs(
            run_label="demo",
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
        ),
        startup=EvalStartup(
            stack=stack,
            config_hash256="c" * 64,
            reported_spec_hash="d" * 64,
            contract=None,
        ),
        dependencies=EvalDispatchDependencies(
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            public_demo_stop_rules_fn=lambda: "stop_rules",
            run_public_demo_final_eval_fn=lambda **_kwargs: {"policy_ids": []},
            run_public_demo_eval_mode_fn=lambda **kwargs: observed.setdefault("public_demo", kwargs),
            run_canonical_eval_pipeline_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("canonical mode should not run")
            ),
            run_summary_only_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("summary mode should not run")
            ),
            load_eval_game_records_fn=None,
            build_matchup_export_fn=None,
            build_seat_advantage_diagnostics_fn=None,
            write_matchup_diagnostics_json_fn=None,
            write_matchup_summary_csv_fn=None,
            write_matchup_summary_json_fn=None,
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


def test_eval_dispatch_request_preserves_route_payloads(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_dispatch_request import eval_dispatch_request
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final_eval",
        policy_id=("B0 RandomLegal", "policy_000100"),
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples="13",
        skip_metagame=1,
        study_config=tmp_path / "study.yaml",
        skip_figures=0,
        skip_readiness=True,
        git_commit_override=123,
        episodes_jsonl=tmp_path / "episodes.jsonl",
        summary_json=tmp_path / "summary.json",
        summary_csv=tmp_path / "summary.csv",
        diagnostics_json=tmp_path / "diagnostics.json",
        bootstrap_seed="23",
        public_demo_paired_seeds="7",
        public_demo_bootstrap_samples="11",
    )
    validated = EvalValidatedArgs(
        run_label="canonical",
        paired_seed_limit=5,
        stage1_paired_seeds=3,
        max_paired_seeds=9,
    )
    startup = EvalStartup(
        stack=stack,
        config_hash256="c" * 64,
        reported_spec_hash="e" * 64,
        contract=SimpleNamespace(simulator={"compatibility_hash": "compat123"}, spec_hash256="e" * 64),
    )
    dependencies = EvalDispatchDependencies(
        public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
        public_demo_stop_rules_fn="stop-rules",
        run_public_demo_final_eval_fn="public-final",
        run_public_demo_eval_mode_fn="public-mode",
        run_canonical_eval_pipeline_fn="canonical",
        run_summary_only_eval_mode_fn="summary",
        load_eval_game_records_fn="load-records",
        build_matchup_export_fn="build-export",
        build_seat_advantage_diagnostics_fn="seat-diagnostics",
        write_matchup_diagnostics_json_fn="write-diagnostics",
        write_matchup_summary_csv_fn="write-csv",
        write_matchup_summary_json_fn="write-json",
    )

    request = eval_dispatch_request(
        parser=parser,
        args=args,
        validated=validated,
        startup=startup,
        dependencies=dependencies,
    )

    assert request.is_public_demo is False
    assert request.has_run_dir is True
    assert request.has_episodes_jsonl is True
    assert request.public_demo_kwargs() == {
        "stack": stack,
        "run_dir": (tmp_path / "run").resolve(),
        "final_eval_dir": (tmp_path / "final_eval").resolve(),
        "paired_seed_limit": 7,
        "bootstrap_samples": 11,
        "config_hash256": "c" * 64,
        "spec_hash256": "e" * 64,
        "public_demo_stop_rules_fn": "stop-rules",
        "run_public_demo_final_eval_fn": "public-final",
    }
    assert request.canonical_kwargs() == {
        "parser": parser,
        "stack": stack,
        "run_dir": (tmp_path / "run").resolve(),
        "final_eval_dir": (tmp_path / "final_eval").resolve(),
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "snapshot_registry_path": (tmp_path / "registry.json").resolve(),
        "dev_eval_summaries_path": (tmp_path / "dev_eval.json").resolve(),
        "b1_baseline_run_dir": (tmp_path / "b1").resolve(),
        "bootstrap_samples": 13,
        "paired_seed_limit": 5,
        "stage1_paired_seeds": 3,
        "max_paired_seeds": 9,
        "skip_metagame": True,
        "study_config_path": (tmp_path / "study.yaml").resolve(),
        "skip_figures": False,
        "skip_readiness": True,
        "git_commit_override": "123",
    }
    assert request.summary_only_kwargs() == {
        "stack": stack,
        "episodes_jsonl": tmp_path / "episodes.jsonl",
        "summary_json": tmp_path / "summary.json",
        "summary_csv": tmp_path / "summary.csv",
        "diagnostics_json": tmp_path / "diagnostics.json",
        "bootstrap_samples": 13,
        "bootstrap_seed": 23,
        "load_eval_game_records_fn": "load-records",
        "build_matchup_export_fn": "build-export",
        "build_seat_advantage_diagnostics_fn": "seat-diagnostics",
        "write_matchup_diagnostics_json_fn": "write-diagnostics",
        "write_matchup_summary_csv_fn": "write-csv",
        "write_matchup_summary_json_fn": "write-json",
    }


def test_eval_dispatch_routes_canonical_with_normalized_args(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    observed: dict[str, object] = {}
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        policy_id=["B0 RandomLegal", "policy_000100"],
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=13,
        skip_metagame=True,
        study_config=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        episodes_jsonl=None,
    )
    contract = SimpleNamespace(
        simulator={"compatibility_hash": "compat123"},
        spec_hash256="e" * 64,
    )

    def fake_canonical(**kwargs: object) -> int:
        observed["canonical"] = kwargs
        return 23

    with pytest.raises(SystemExit) as exc_info:
        run_eval_dispatch(
            parser=argparse.ArgumentParser(),
            args=args,
            validated=EvalValidatedArgs(
                run_label="canonical",
                paired_seed_limit=5,
                stage1_paired_seeds=3,
                max_paired_seeds=9,
            ),
            startup=EvalStartup(
                stack=stack,
                config_hash256="c" * 64,
                reported_spec_hash="e" * 64,
                contract=contract,
            ),
            dependencies=EvalDispatchDependencies(
                public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
                public_demo_stop_rules_fn=None,
                run_public_demo_final_eval_fn=None,
                run_public_demo_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("public demo should not run")
                ),
                run_canonical_eval_pipeline_fn=fake_canonical,
                run_summary_only_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("summary mode should not run")
                ),
                load_eval_game_records_fn=None,
                build_matchup_export_fn=None,
                build_seat_advantage_diagnostics_fn=None,
                write_matchup_diagnostics_json_fn=None,
                write_matchup_summary_csv_fn=None,
                write_matchup_summary_json_fn=None,
            ),
        )

    assert exc_info.value.code == 23
    call = observed["canonical"]
    assert call["stack"] is stack
    assert call["run_dir"] == (tmp_path / "run").resolve()
    assert call["final_eval_dir"] == (tmp_path / "run" / "eval" / "final_eval").resolve()
    assert call["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert call["snapshot_registry_path"] == (tmp_path / "registry.json").resolve()
    assert call["dev_eval_summaries_path"] == (tmp_path / "dev_eval.json").resolve()
    assert call["b1_baseline_run_dir"] == (tmp_path / "b1").resolve()
    assert call["bootstrap_samples"] == 13
    assert call["paired_seed_limit"] == 5
    assert call["stage1_paired_seeds"] == 3
    assert call["max_paired_seeds"] == 9
    assert call["skip_metagame"] is True
    assert call["study_config_path"] == (tmp_path / "study.yaml").resolve()
    assert call["skip_figures"] is True
    assert call["skip_readiness"] is True
    assert call["git_commit_override"] == "abc123"


def test_eval_dispatch_routes_summary_only_with_dependency_bundle(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    observed: dict[str, object] = {}
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=None,
        episodes_jsonl=tmp_path / "episodes.jsonl",
        summary_json=tmp_path / "summary.json",
        summary_csv=tmp_path / "summary.csv",
        diagnostics_json=tmp_path / "diagnostics.json",
        bootstrap_samples=17,
        bootstrap_seed=23,
    )

    def fake_summary(**kwargs: object) -> None:
        observed["summary"] = kwargs

    run_eval_dispatch(
        parser=argparse.ArgumentParser(),
        args=args,
        validated=EvalValidatedArgs(
            run_label="summary",
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
        ),
        startup=EvalStartup(
            stack=stack,
            config_hash256="c" * 64,
            reported_spec_hash="e" * 64,
            contract=SimpleNamespace(
                simulator={"compatibility_hash": "compat123"},
                spec_hash256="e" * 64,
            ),
        ),
        dependencies=EvalDispatchDependencies(
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            public_demo_stop_rules_fn=None,
            run_public_demo_final_eval_fn=None,
            run_public_demo_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("public demo should not run")
            ),
            run_canonical_eval_pipeline_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("canonical mode should not run")
            ),
            run_summary_only_eval_mode_fn=fake_summary,
            load_eval_game_records_fn="load-records",
            build_matchup_export_fn="build-export",
            build_seat_advantage_diagnostics_fn="seat-diagnostics",
            write_matchup_diagnostics_json_fn="write-diagnostics",
            write_matchup_summary_csv_fn="write-csv",
            write_matchup_summary_json_fn="write-json",
        ),
    )

    call = observed["summary"]
    assert call["stack"] is stack
    assert call["episodes_jsonl"] == tmp_path / "episodes.jsonl"
    assert call["summary_json"] == tmp_path / "summary.json"
    assert call["summary_csv"] == tmp_path / "summary.csv"
    assert call["diagnostics_json"] == tmp_path / "diagnostics.json"
    assert call["bootstrap_samples"] == 17
    assert call["bootstrap_seed"] == 23
    assert call["load_eval_game_records_fn"] == "load-records"
    assert call["build_matchup_export_fn"] == "build-export"
    assert call["build_seat_advantage_diagnostics_fn"] == "seat-diagnostics"
    assert call["write_matchup_diagnostics_json_fn"] == "write-diagnostics"
    assert call["write_matchup_summary_csv_fn"] == "write-csv"
    assert call["write_matchup_summary_json_fn"] == "write-json"
    assert "Verified runtime spec bundle" in capsys.readouterr().out


def test_eval_dispatch_contract_check_only_skips_route_adapters(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "report_eval.txt", "dev_eval": tmp_path / "dev.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=None,
        episodes_jsonl=None,
    )

    run_eval_dispatch(
        parser=argparse.ArgumentParser(),
        args=args,
        validated=EvalValidatedArgs(
            run_label="contract_check",
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
        ),
        startup=EvalStartup(
            stack=stack,
            config_hash256="c" * 64,
            reported_spec_hash="e" * 64,
            contract=SimpleNamespace(
                simulator={"compatibility_hash": "compat123"},
                spec_hash256="e" * 64,
            ),
        ),
        dependencies=EvalDispatchDependencies(
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            public_demo_stop_rules_fn=None,
            run_public_demo_final_eval_fn=None,
            run_public_demo_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("public demo should not run")
            ),
            run_canonical_eval_pipeline_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("canonical mode should not run")
            ),
            run_summary_only_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("summary mode should not run")
            ),
        ),
    )

    output = capsys.readouterr().out
    assert "Verified runtime spec bundle" in output
    assert "Evaluation contract check complete; no episodes were summarized." in output
    assert "Seed sets: ['dev_eval', 'report_eval']" in output


def test_eval_entrypoint_honors_explicit_spec_hash_without_reproducibility_config(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash=_mismatched_sha256(spec_bundle_hash(bundle)),
    )

    assert result.returncode != 0
    assert "Spec bundle hash mismatch" in result.stderr


def test_eval_entrypoint_accepts_spec_bundle_sha256(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash=spec_bundle_hash(bundle),
    )

    assert result.returncode == 0, result.stderr
    assert "Verified runtime spec bundle" in result.stdout
    assert "run_label:              (default)" in result.stdout
    assert "computed_run_id64:" not in result.stdout


def test_eval_entrypoint_reports_run_label_without_claiming_run_identity(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        run_label="eval_report_label",
    )

    assert result.returncode == 0, result.stderr
    assert "run_label:              eval_report_label" in result.stdout
    assert "Verified runtime spec bundle" in result.stdout
    assert "computed_run_id64:" not in result.stdout
    assert "computed_run_id256:" not in result.stdout


def test_eval_entrypoint_fails_fast_on_config_hash_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    config_hash256 = compute_config_hash256(load_stack_config(stack_config))

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--config-hash", _mismatched_sha256(config_hash256)],
    )

    assert result.returncode != 0
    assert "--config-hash mismatch" in result.stderr


def test_eval_entrypoint_requires_skip_readiness_when_skipping_required_outputs(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--run-dir", str(tmp_path / "runs" / "candidate"), "--skip-metagame"],
    )

    assert result.returncode != 0
    assert "--skip-metagame or --skip-figures requires --skip-readiness" in result.stderr


def test_eval_entrypoint_exports_summary_json_and_csv(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    diagnostics_json = tmp_path / "diagnostics.json"
    episodes_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "pair_index": 0,
                        "swap_index": 0,
                        "episode_index": 0,
                        "episode_seed": 7,
                        "episode_key": "01" * 32,
                        "episode_key64": 1,
                        "config_hash256": "ab" * 32,
                        "spec_hash256": "cd" * 32,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "champion",
                        "seat1_policy_id": "baseline",
                        "focal_seat": 0,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "pair_index": 0,
                        "swap_index": 1,
                        "episode_index": 1,
                        "episode_seed": 7,
                        "episode_key": "02" * 32,
                        "episode_key64": 2,
                        "config_hash256": "ab" * 32,
                        "spec_hash256": "cd" * 32,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "baseline",
                        "seat1_policy_id": "champion",
                        "focal_seat": 1,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                    },
                    sort_keys=True,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash="",
        extra_args=[
            "--episodes-jsonl",
            str(episodes_path),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
            "--diagnostics-json",
            str(diagnostics_json),
            "--bootstrap-samples",
            "16",
            "--bootstrap-seed",
            "7",
        ],
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "decisive"
    assert payload["summary"]["wins"] == 2
    assert diagnostics["seat_results"]["seat0_wins"] == 1
    assert diagnostics["seat_results"]["seat1_wins"] == 1
    assert summary_csv.read_text(encoding="utf-8").splitlines()[0].startswith("focal_policy_id,")


def test_eval_entrypoint_public_demo_generates_demo_only_final_eval_artifacts(tmp_path: Path) -> None:
    train_result, run_dir = _run_public_demo_train(tmp_path, run_label="toy_public_demo_eval")
    assert train_result.returncode == 0, train_result.stderr
    stack_config = tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"

    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.workflows.eval_entrypoint",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        extra_args=[
            "--public-demo",
            "--run-dir",
            str(run_dir),
            "--public-demo-paired-seeds",
            "4",
            "--public-demo-bootstrap-samples",
            "8",
        ],
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((run_dir / "eval" / "final_eval" / "summary.json").read_text(encoding="utf-8"))
    metadata = summary["metadata"]

    assert summary["policy_ids"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "toy_policy_000100",
        "toy_policy_000200",
    ]
    assert metadata["demo_only"] is True
    assert metadata["public_safe"] is True
    assert metadata["catalog_path"] == "public_demo/catalog.json"
    assert metadata["policy_manifest_path"] == "public_demo/policy_manifest.json"
    assert metadata["paired_seed_budget"] == 4
    assert metadata["recommended_focal_policy_id"] == "toy_policy_000200"
    assert len(summary["matchups"]) == 10
    assert "Public-demo final_eval summary JSON" in result.stdout
