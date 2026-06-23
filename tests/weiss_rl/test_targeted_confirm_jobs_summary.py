from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_targeted_confirm_jobs_write_into_requested_output_subdir() -> None:
    from weiss_rl.eval.targeted_confirm.jobs import targeted_eval_job

    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=Path("runs/demo"),
        snapshot_registry_json=Path("runs/demo/training/snapshots/registry.json"),
        b1_baseline_run_dir=Path("runs/demo"),
        paired_seeds=16,
        bootstrap_samples=1000,
        focal_policy_id="policy_000003",
    )

    job = targeted_eval_job(
        args=args,
        paired_seeds=[1, 2, 3],
        opponent_index=2,
        opponent="B2 HeuristicPublic",
        output_dir=Path("runs/demo/eval/confirm16_argmax"),
    )

    assert job["output_dir"] == "runs/demo/eval/confirm16_argmax"
    assert job["opponent_policy_id"] == "B2 HeuristicPublic"
    assert job["paired_seeds"] == [1, 2, 3]


def test_targeted_confirm_summary_helpers_preserve_worker_and_summary_payloads(tmp_path: Path) -> None:
    from weiss_rl.eval.targeted_confirm import core as targeted_confirm_core
    from weiss_rl.eval.targeted_confirm import summary as targeted_confirm_summary

    matchup_dir = tmp_path / "eval" / "confirm16" / "matchups" / "00_policy__vs__01_b2"
    worker_result = targeted_confirm_summary.targeted_worker_summary_from_result(
        {
            "focal_policy_id": "policy_000021",
            "opponent_policy_id": "B2 HeuristicPublic",
            "matchup_dir": matchup_dir,
            "used_paired_seeds": (1, 2, 3),
            "summary": {
                "summary": {
                    "games": 6,
                    "wins": 4,
                    "losses": 1,
                    "draws": 1,
                    "truncations": 2,
                    "engine_errors": 0,
                },
                "uncertainty": {
                    "paired_seed_count": 3,
                    "mean": 0.75,
                    "ci_low": 0.6,
                    "ci_high": 0.9,
                    "prob_gt_half": 0.95,
                },
            },
        }
    )
    seed_file = tmp_path / "report_eval.txt"
    seed_file.write_text("1\n2\n3\n", encoding="utf-8")
    args = SimpleNamespace(
        focal_policy_id="policy_000021",
        paired_seeds=16,
        stack_config=Path("configs/thesis/final_eval.yaml"),
        god_search_mode="same_world_prefix_rollout",
        god_search_top_k=3,
        god_search_rollouts_per_action=2,
        god_search_max_rollout_decisions=4,
        god_search_max_search_decisions_per_game=5,
        god_search_rollout_policy="argmax",
        god_search_no_prefix_verify=True,
        god_search_soft_prefix_fail=True,
        god_search_trace_limit=7,
    )
    plan = SimpleNamespace(
        args=args,
        out_dir=tmp_path / "eval" / "confirm16",
        seed_file_path=seed_file,
        seed_source="report_eval",
        eval_sampling_algorithm="argmax",
        model_sampling_temperature=0.5,
        jobs=[
            {"opponent_policy_id": "B2 HeuristicPublic"},
            {"opponent_policy_id": "B4 HeuristicPublicControl"},
            {"opponent_policy_id": "seed_1"},
            {"opponent_policy_id": "seed_2"},
            {"opponent_policy_id": "seed_3"},
            {"opponent_policy_id": "seed_4"},
        ],
    )
    results_by_opp = {
        "B2 HeuristicPublic": worker_result,
        "B4 HeuristicPublicControl": {**worker_result, "wins": 3, "games": 6},
        "seed_1": {**worker_result, "wins": 2, "games": 6},
        "seed_2": {**worker_result, "wins": 1, "games": 6},
        "seed_3": {**worker_result, "wins": 0, "games": 6},
        "seed_4": {**worker_result, "wins": 6, "games": 6},
    }

    summary = targeted_confirm_summary.build_targeted_confirm_summary(
        plan=plan,
        results_by_opp=results_by_opp,
        started_unix=0.0,
    )

    assert targeted_confirm_core.targeted_worker_summary_from_result is (
        targeted_confirm_summary.targeted_worker_summary_from_result
    )
    assert worker_result == {
        "focal_policy_id": "policy_000021",
        "opponent_policy_id": "B2 HeuristicPublic",
        "paired_seeds": 3,
        "games": 6,
        "wins": 4,
        "losses": 1,
        "draws": 1,
        "mean": 0.75,
        "ci_low": 0.6,
        "ci_high": 0.9,
        "prob_gt_half": 0.95,
        "truncations": 2,
        "engine_errors": 0,
        "summary_path": (matchup_dir / "matchup_summary.json").as_posix(),
        "diagnostics_path": (matchup_dir / "diagnostics.json").as_posix(),
    }
    assert summary["games_per_row"] == 32
    assert summary["seed_file"]["source"] == "report_eval"
    assert summary["overall"] == {"games": 36, "mean": 16 / 36, "wins": 16}
    assert summary["anchor_subset"] == {"games": 30, "mean": 10 / 30, "wins": 10}
    assert summary["legacy_subset"] == {"games": 6, "mean": 1.0, "wins": 6}
    assert summary["god_search"] == {
        "mode": "same_world_prefix_rollout",
        "top_k": 3,
        "rollouts_per_action": 2,
        "max_rollout_decisions": 4,
        "max_search_decisions_per_game": 5,
        "rollout_policy": "argmax",
        "apply_to_focal_only": True,
        "verify_prefix_replay": False,
        "fail_on_prefix_mismatch": False,
        "trace_limit": 7,
    }


def test_targeted_confirm_plan_builds_ordered_jobs_with_shared_output_dir() -> None:
    from weiss_rl.eval.targeted_confirm.jobs import build_targeted_confirm_jobs

    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=Path("runs/demo"),
        snapshot_registry_json=Path("runs/demo/training/snapshots/registry.json"),
        b1_baseline_run_dir=Path("runs/b1"),
        paired_seeds=32,
        bootstrap_samples=1000,
        focal_policy_id="policy_000021",
        god_search_mode="disabled",
    )

    jobs = build_targeted_confirm_jobs(
        args=args,
        paired_seeds=[1, 2, 3],
        opponents=["B2 HeuristicPublic", "B4 HeuristicPublicControl"],
        output_dir=Path("runs/demo/eval/targeted_confirm32"),
    )

    assert [(job["opponent_index"], job["opponent_policy_id"]) for job in jobs] == [
        (1, "B2 HeuristicPublic"),
        (2, "B4 HeuristicPublicControl"),
    ]
    assert jobs[0]["focal_index"] == 0
    assert jobs[0]["focal_policy_id"] == "policy_000021"
    assert jobs[0]["output_dir"] == "runs/demo/eval/targeted_confirm32"
    assert jobs[0]["stage1_paired_seeds"] == 32
    assert jobs[0]["max_paired_seeds"] == 32


def test_targeted_confirm_plan_preparation_uses_seed_source_output_dir_and_eval_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from weiss_rl.eval.targeted_confirm import jobs as targeted_confirm_jobs

    seed_file = tmp_path / "custom_seeds.txt"
    seed_file.write_text("10\n11\n12\n13\n", encoding="utf-8")
    fake_stack = SimpleNamespace(
        seed_sets={"custom_eval": seed_file},
        config=SimpleNamespace(
            evaluation=SimpleNamespace(
                eval_sampling_algorithm="model_argmax_pinned_v1",
                model_sampling_temperature=0.25,
            )
        ),
    )
    monkeypatch.setattr(targeted_confirm_jobs, "load_stack_config", lambda _path: fake_stack)
    args = SimpleNamespace(
        stack_config=Path("configs/thesis/final_eval.yaml"),
        run_dir=tmp_path / "run",
        snapshot_registry_json=Path("runs/demo/training/snapshots/registry.json"),
        b1_baseline_run_dir=Path("runs/b1"),
        paired_seed_file=None,
        seed_set="custom_eval",
        paired_seeds=3,
        output_subdir="confirm3_custom",
        bootstrap_samples=99,
        focal_policy_id="policy_000021",
        god_search_mode="disabled",
    )

    plan = targeted_confirm_jobs.prepare_targeted_confirm_plan(
        args,
        opponents=["B2 HeuristicPublic", "B4 HeuristicPublicControl"],
    )

    assert plan.__class__ is targeted_confirm_jobs.TargetedConfirmPlan
    assert plan.opponents == ["B2 HeuristicPublic", "B4 HeuristicPublicControl"]
    assert plan.paired_seeds == [10, 11, 12]
    assert plan.seed_file_path == seed_file
    assert plan.seed_source == "custom_eval"
    assert plan.out_dir == tmp_path / "run" / "eval" / "confirm3_custom"
    assert plan.out_dir.is_dir()
    assert plan.eval_sampling_algorithm == "model_argmax_pinned_v1"
    assert plan.model_sampling_temperature == 0.25
    assert [(job["opponent_index"], job["opponent_policy_id"]) for job in plan.jobs] == [
        (1, "B2 HeuristicPublic"),
        (2, "B4 HeuristicPublicControl"),
    ]
    assert plan.jobs[0]["paired_seeds"] == [10, 11, 12]
    assert plan.jobs[0]["bootstrap_samples"] == 99
    assert plan.jobs[0]["output_dir"] == (tmp_path / "run" / "eval" / "confirm3_custom").as_posix()
