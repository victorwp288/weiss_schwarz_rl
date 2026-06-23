from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.config.models import StopRulesConfig
from weiss_rl.eval import final_eval as final_eval_module
from weiss_rl.eval.final import matchup_outputs as final_eval_matchup_outputs
from weiss_rl.eval.final import matchup_schedule as final_eval_matchup_schedule
from weiss_rl.eval.final import matchups as final_eval_matchups
from weiss_rl.eval.final.matchups import (
    matchup_dir_name,
    run_final_eval_matchup,
    scheduled_game,
)

from .final_eval_test_support import (
    _CONFIG_HASH256,
    _RUN_ID256,
    _SPEC_HASH256,
    _AlwaysWinRunner,
)


def test_final_eval_matchup_runner_preserves_schedule_artifacts_and_early_stop(tmp_path: Path) -> None:
    runner = _AlwaysWinRunner()
    output_dir = tmp_path / "final_eval"
    first_swap = scheduled_game(
        pair_index=2,
        swap_index=1,
        episode_seed=33,
        focal_policy_id="policy_000021",
        opponent_policy_id="B3 HeuristicPublicAggro",
    )

    result = run_final_eval_matchup(
        output_dir=output_dir,
        focal_index=0,
        opponent_index=1,
        focal_policy_id="policy_000021",
        opponent_policy_id="B3 HeuristicPublicAggro",
        paired_seeds=[11, 22, 33, 44],
        stage1_paired_seeds=2,
        max_paired_seeds=4,
        stop_rules=StopRulesConfig(stop_delta_ci_half_width=0.05, stop_confidence=0.95),
        runner=runner,
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
        scheme="S0",
        sample_count=8,
    )

    assert final_eval_module._run_matchup is run_final_eval_matchup
    assert final_eval_module._scheduled_game is scheduled_game
    assert final_eval_module._matchup_dir_name is matchup_dir_name
    assert first_swap.seat0_policy_id == "B3 HeuristicPublicAggro"
    assert first_swap.seat1_policy_id == "policy_000021"
    assert first_swap.focal_seat == 1
    assert first_swap.seat0_deck == "preset:aggro_deck_5hy_nino_v1"
    assert first_swap.seat1_deck == "preset:main_deck_5hy_yotsuba_v1"
    assert len(runner.calls) == 4
    assert [(call.pair_index, call.swap_index, call.focal_seat) for call in runner.calls] == [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 1, 1),
    ]
    assert result["matchup_dir"].name == "00_policy_000021__vs__01_b3_heuristicpublicaggro"
    assert result["used_paired_seeds"] == (11, 22)
    assert len(result["records"]) == 4
    assert len(result["replay_samples"]) == 1
    assert result["summary"]["evaluation_context"] == {
        "artifact_scope": "final_eval",
        "focal_policy_index": 0,
        "opponent_policy_index": 1,
        "stage1_paired_seeds": 2,
        "max_paired_seeds": 4,
        "used_paired_seeds": [11, 22],
    }
    assert result["diagnostics"]["god_search"] == {"enabled": True, "checked": 4}
    assert len(result["episodes_path"].read_text(encoding="utf-8").splitlines()) == 4
    assert (result["matchup_dir"] / "matchup_summary.json").is_file()
    assert (result["matchup_dir"] / "matchup_summary.csv").is_file()
    assert (result["matchup_dir"] / "diagnostics.json").is_file()
    posterior_payload = json.loads((result["matchup_dir"] / "posterior_samples.json").read_text(encoding="utf-8"))
    assert posterior_payload["requested_sample_count"] == 8
    assert posterior_payload["sample_count"] == 8
    assert posterior_payload["has_payoff_samples"] is True


def test_final_eval_matchup_split_modules_preserve_legacy_exports(tmp_path: Path) -> None:
    matchup_dir = tmp_path / "matchups" / "00_policy_alpha__vs__01_policy_beta"

    assert final_eval_matchups.scheduled_game is final_eval_matchup_schedule.scheduled_game
    assert final_eval_matchups.matchup_dir_name is final_eval_matchup_schedule.matchup_dir_name
    assert final_eval_matchups.slug is final_eval_matchup_schedule.slug
    assert final_eval_matchups.bootstrap_seed is final_eval_matchup_schedule.bootstrap_seed
    assert final_eval_matchups.build_matchup_payload is final_eval_matchup_outputs.build_matchup_payload
    assert final_eval_matchups.matchup_posterior_samples is final_eval_matchup_outputs.matchup_posterior_samples
    assert final_eval_matchups.build_matchup_diagnostics is final_eval_matchup_outputs.build_matchup_diagnostics
    assert final_eval_matchups.write_matchup_artifacts is final_eval_matchup_outputs.write_matchup_artifacts
    assert final_eval_matchups.matchup_artifact_paths is final_eval_matchup_outputs.matchup_artifact_paths

    paths = final_eval_matchup_outputs.matchup_artifact_paths(matchup_dir)
    assert paths.summary_json == matchup_dir / "matchup_summary.json"
    assert paths.summary_csv == matchup_dir / "matchup_summary.csv"
    assert paths.diagnostics_json == matchup_dir / "diagnostics.json"
    assert paths.posterior_samples_json == matchup_dir / "posterior_samples.json"
    assert final_eval_matchup_outputs.build_posterior_samples_payload(
        focal_policy_id="policy_alpha",
        opponent_policy_id="policy_beta",
        sample_count=8,
        posterior_samples=(0.25, 0.75),
        summary_payload={"has_payoff_samples": True},
    ) == {
        "focal_policy_id": "policy_alpha",
        "opponent_policy_id": "policy_beta",
        "requested_sample_count": 8,
        "sample_count": 2,
        "has_payoff_samples": True,
        "samples": [0.25, 0.75],
    }


def test_eval_root_does_not_export_final_eval_matchup_module_aliases() -> None:
    import weiss_rl.eval as eval_package

    assert not hasattr(eval_package, "final_eval_matchup_outputs")
    assert not hasattr(eval_package, "final_eval_matchup_schedule")
    assert not hasattr(eval_package, "final_eval_matchups")


def test_eval_root_does_not_export_final_eval_helper_module_aliases() -> None:
    import weiss_rl.eval as eval_package

    assert not hasattr(eval_package, "final_eval_artifacts")
    assert not hasattr(eval_package, "final_eval_matrices")
    assert not hasattr(eval_package, "final_eval_payload")
    assert not hasattr(eval_package, "final_eval_policy_selection")
