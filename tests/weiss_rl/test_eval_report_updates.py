from __future__ import annotations

from .entrypoints_test_support import (
    ArtifactLayout,
    Path,
    json,
)


def test_eval_report_update_payloads_preserve_summary_and_determinism_fields(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.reports.eval_report_update_payloads import (
        RunLevelReportUpdateInputs,
        build_canonical_eval_evidence_summary,
        build_determinism_report_update_fields,
        build_run_summary_update_fields,
    )

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    selection_details = {"mode": "deterministic_v1", "status": "resolved"}
    inputs = RunLevelReportUpdateInputs(
        layout=layout,
        run_dir=layout.run_dir,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details=selection_details,
        final_eval_payload={
            "summary_sections": [{"key": "metadata.selection"}, {"key": "matrices"}],
            "matchups": [{"a": 1}, {"a": 2}],
        },
        metagame_payload={"kind": "metagame"},
        figure_paths=(layout.figures_paper_dir / "seat_bias.pdf", tmp_path / "external.pdf"),
        readiness_payload={"passed": True},
    )
    evidence = build_canonical_eval_evidence_summary(inputs)

    assert build_run_summary_update_fields(inputs) == {
        "final_eval_dir": "eval/final_eval",
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "policy_set_selection_mode": "deterministic_v1",
        "metagame_dir": "eval/metagame",
        "figure_outputs": ["figures/paper/seat_bias.pdf", (tmp_path / "external.pdf").as_posix()],
        "paper_readiness_summary_path": "paper_readiness_summary.json",
        "paper_grade": True,
        "canonical_eval_completed": True,
        "canonical_eval_evidence": evidence,
    }
    assert evidence == {
        "final_eval_summary_path": "eval/final_eval/summary.json",
        "policy_set_path": "eval/final_eval/policy_set.json",
        "matchups_path": "eval/final_eval/matchups.csv",
        "mean_matrix_path": "eval/final_eval/matrices/mean.csv",
        "paper_readiness_summary_path": "paper_readiness_summary.json",
        "policy_count": 2,
        "matchup_count": 2,
        "policy_selection_mode": "deterministic_v1",
        "selection_status": "resolved",
        "summary_section_keys": ["metadata.selection", "matrices"],
        "paper_grade": True,
    }

    determinism_fields = build_determinism_report_update_fields(
        inputs,
        replay_verification={
            "status": "verified",
            "sampled_episode_count": 5,
            "verified_episode_count": 4,
            "failed_episode_count": 1,
        },
        artifact_hashes={"artifacts": {"summary.json": "ab" * 32}},
    )

    assert determinism_fields == {
        "run_dir": layout.run_dir.as_posix(),
        "policy_selection_mode": "deterministic_v1",
        "replay_verification": {
            "path": "eval/diagnostics/replay_verification.json",
            "status": "verified",
            "sampled_episode_count": 5,
            "verified_episode_count": 4,
            "failed_episode_count": 1,
        },
        "canonical_artifact_hashes": {"summary.json": "ab" * 32},
        "canonical_eval_evidence": evidence,
        "final_eval": {
            "path": "eval/final_eval/summary.json",
            "policy_ids": ["B0 RandomLegal", "policy_000100"],
            "selection": selection_details,
            "matchup_count": 2,
        },
    }


def test_eval_report_update_writes_summary_and_determinism_artifacts(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.reports.eval_report_updates import _update_run_level_reports

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.ensure_directories()
    layout.run_summary_path.write_text(
        json.dumps({"kind": "run_summary_v1", "preexisting": "summary"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    layout.determinism_report_path.write_text(
        json.dumps({"kind": "determinism_report_v1", "preexisting": "determinism"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    layout.replay_verification_json().write_text(
        json.dumps(
            {
                "status": "verified",
                "sampled_episode_count": 3,
                "verified_episode_count": 3,
                "failed_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    layout.final_eval_aggregate_hashes_json().write_text(
        json.dumps({"artifacts": {"summary.json": "cd" * 32}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _update_run_level_reports(
        layout=layout,
        run_dir=layout.run_dir,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"mode": "deterministic_v1", "status": "resolved"},
        final_eval_payload={"summary_sections": [{"key": "matrices"}], "matchups": [{"winner": "a"}]},
        metagame_payload=None,
        figure_paths=(),
        readiness_payload={"passed": False},
    )

    run_summary = json.loads(layout.run_summary_path.read_text(encoding="utf-8"))
    assert run_summary["preexisting"] == "summary"
    assert run_summary["final_eval_dir"] == "eval/final_eval"
    assert run_summary["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert run_summary["policy_set_selection_mode"] == "deterministic_v1"
    assert run_summary["metagame_dir"] is None
    assert run_summary["figure_outputs"] == []
    assert run_summary["paper_grade"] is False
    assert run_summary["canonical_eval_completed"] is True
    assert run_summary["canonical_eval_evidence"]["summary_section_keys"] == ["matrices"]
    assert run_summary["canonical_eval_evidence"]["paper_grade"] is False

    determinism = json.loads(layout.determinism_report_path.read_text(encoding="utf-8"))
    assert determinism["preexisting"] == "determinism"
    assert determinism["policy_selection_mode"] == "deterministic_v1"
    assert determinism["replay_verification"] == {
        "path": "eval/diagnostics/replay_verification.json",
        "status": "verified",
        "sampled_episode_count": 3,
        "verified_episode_count": 3,
        "failed_episode_count": 0,
    }
    assert determinism["canonical_artifact_hashes"] == {"summary.json": "cd" * 32}
    assert determinism["canonical_eval_evidence"]["final_eval_summary_path"] == "eval/final_eval/summary.json"
    assert determinism["final_eval"]["matchup_count"] == 1


def test_eval_report_helpers_create_defaults_for_interpolated_runs(tmp_path: Path) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    run_dir = tmp_path / "runs" / "interpolated_eval"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_id256": "ab" * 32,
                "run_id64": "ab" * 8,
                "evaluation_pinning": {"eval_device": "cpu"},
                "seed_derivation": {"base_seed": 7},
                "seed_files": {"report_eval": {"path": "seeds/report.txt", "sha256": "cd" * 32}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_summary = eval_script._load_run_summary_or_default(layout)
    determinism = eval_script._load_determinism_report_or_default(layout)
    environment = eval_script._load_environment_or_default(layout)

    assert run_summary["runtime_mode"] == "interpolated_checkpoint"
    assert run_summary["run_id256"] == "ab" * 32
    assert determinism["device_policy"]["learner"] == "interpolated_checkpoint"
    assert determinism["device_policy"]["evaluation"] == "cpu"
    assert determinism["seed_derivation"] == {"base_seed": 7}
    assert environment["kind"] == "environment_manifest_v1"
    assert environment["run_id256"] == "ab" * 32


def test_eval_git_commit_override_does_not_mutate_manifest_payload() -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    manifest = {"run_id256": "ab" * 32}

    effective = eval_script._effective_manifest_git_commit(
        manifest=manifest,
        git_commit_override="deadbeef" * 5,
    )

    assert effective == "deadbeef" * 5
    assert "git_commit" not in manifest
