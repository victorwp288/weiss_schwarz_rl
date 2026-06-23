from __future__ import annotations

from .entrypoints_test_support import (
    Path,
    _copy_repo_configs,
    _run_entrypoint,
    _run_public_demo_train,
    _write_stub_weiss_sim,
    json,
    public_demo_spec_hash256,
)


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
