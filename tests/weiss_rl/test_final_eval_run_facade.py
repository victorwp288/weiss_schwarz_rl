from __future__ import annotations

from pathlib import Path
from typing import Any

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import final_eval as final_eval_module
from weiss_rl.eval.final import run as final_eval_run

from .final_eval_test_support import (
    _CONFIG_HASH256,
    _RUN_ID256,
    _SPEC_HASH256,
    _FakeMatrixRunner,
)


def test_final_eval_run_split_preserves_facade_and_upper_triangle_jobs(tmp_path: Path) -> None:
    jobs = final_eval_run.build_final_eval_matchup_jobs(["policy_a", "policy_b", "policy_c"])

    assert final_eval_module.run_final_eval is final_eval_run.run_final_eval
    assert final_eval_module._build_matchup_jobs is final_eval_run.build_final_eval_matchup_jobs
    assert final_eval_module._build_run_payload is final_eval_run.build_final_eval_run_payload
    assert final_eval_module._resolve_run_policy_ids is final_eval_run.resolve_final_eval_run_policy_ids
    assert final_eval_module._run_matchup_jobs is final_eval_run.run_final_eval_matchup_jobs
    assert final_eval_module._validate_run_seed_budget is final_eval_run.validate_final_eval_run_seed_budget
    assert final_eval_module._write_run_artifacts is final_eval_run.write_final_eval_run_artifacts
    assert [(job.focal_index, job.opponent_index, job.focal_policy_id, job.opponent_policy_id) for job in jobs] == [
        (0, 0, "policy_a", "policy_a"),
        (0, 1, "policy_a", "policy_b"),
        (0, 2, "policy_a", "policy_c"),
        (1, 1, "policy_b", "policy_b"),
        (1, 2, "policy_b", "policy_c"),
        (2, 2, "policy_c", "policy_c"),
    ]

    calls: list[dict[str, Any]] = []

    def fake_run_matchup(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "focal_index": kwargs["focal_index"],
            "opponent_index": kwargs["opponent_index"],
            "focal_policy_id": kwargs["focal_policy_id"],
            "opponent_policy_id": kwargs["opponent_policy_id"],
            "matchup_dir": tmp_path / "matchup",
            "summary": {},
            "posterior_samples": (),
            "records": (),
            "replay_samples": (),
        }

    results = final_eval_run.run_final_eval_matchup_jobs(
        output_dir=tmp_path / "final_eval",
        jobs=jobs[:2],
        runner=_FakeMatrixRunner({}),
        paired_seeds=[11, 22],
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        scheme="S1",
        sample_count=7,
        run_matchup_fn=fake_run_matchup,
    )

    assert (tmp_path / "final_eval").is_dir()
    assert [(result["focal_index"], result["opponent_index"]) for result in results] == [(0, 0), (0, 1)]
    assert [(call["focal_policy_id"], call["opponent_policy_id"]) for call in calls] == [
        ("policy_a", "policy_a"),
        ("policy_a", "policy_b"),
    ]
    assert calls[0]["paired_seeds"] == [11, 22]
    assert calls[0]["stage1_paired_seeds"] == 1
    assert calls[0]["max_paired_seeds"] == 2
    assert calls[0]["scheme"] == "S1"
    assert calls[0]["sample_count"] == 7


def test_eval_root_does_not_export_final_eval_run_module_alias() -> None:
    import weiss_rl.eval as eval_package

    assert not hasattr(eval_package, "final_eval_run")
