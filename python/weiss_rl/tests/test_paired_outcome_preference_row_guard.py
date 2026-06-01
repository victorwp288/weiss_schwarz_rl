from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.experiments.paired_outcome_preference_row_guard import (
    PairedOutcomePreferenceRowGuardConfig,
    evaluate_row_guard_rows,
)
from weiss_rl.experiments.paired_outcome_preference_row_guard_reporting import (
    paired_outcome_preference_row_guard_output_line,
    paired_outcome_preference_row_guard_output_payload,
)
from weiss_rl.experiments.paired_outcome_preference_row_guard_runtime import (
    PairedOutcomePreferenceRowGuardRunResult,
    paired_outcome_preference_row_guard_config_from_args,
)


def test_row_guard_entrypoint_facade_reexports_cli_runtime_and_core_helpers() -> None:
    from weiss_rl.experiments import (
        paired_outcome_preference_row_guard,
        paired_outcome_preference_row_guard_cli,
        paired_outcome_preference_row_guard_entrypoint,
        paired_outcome_preference_row_guard_runtime,
    )

    assert paired_outcome_preference_row_guard_entrypoint._build_parser is (
        paired_outcome_preference_row_guard_cli.build_paired_outcome_preference_row_guard_parser
    )
    assert paired_outcome_preference_row_guard_entrypoint.run_paired_outcome_preference_row_guard is (
        paired_outcome_preference_row_guard_runtime.run_paired_outcome_preference_row_guard
    )
    assert paired_outcome_preference_row_guard_entrypoint.PairedOutcomePreferenceRowGuardConfig is (
        paired_outcome_preference_row_guard.PairedOutcomePreferenceRowGuardConfig
    )
    assert paired_outcome_preference_row_guard_entrypoint.build_paired_outcome_preference_row_guard is (
        paired_outcome_preference_row_guard.build_paired_outcome_preference_row_guard
    )
    assert paired_outcome_preference_row_guard_entrypoint.write_paired_outcome_preference_row_guard is (
        paired_outcome_preference_row_guard.write_paired_outcome_preference_row_guard
    )


def test_row_guard_parser_preserves_defaults(tmp_path: Path) -> None:
    from weiss_rl.experiments.paired_outcome_preference_row_guard_cli import (
        build_paired_outcome_preference_row_guard_parser,
    )

    args = build_paired_outcome_preference_row_guard_parser().parse_args(
        [
            "--dataset",
            str(tmp_path / "preference.npz"),
            "--stack-config",
            str(tmp_path / "stack.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--reference-checkpoint",
            str(tmp_path / "reference.pt"),
            "--output-json",
            str(tmp_path / "row_guard.json"),
        ]
    )

    assert args.dataset == tmp_path / "preference.npz"
    assert args.stack_config == tmp_path / "stack.yaml"
    assert args.run_dir == tmp_path / "run"
    assert args.checkpoint == tmp_path / "checkpoint.pt"
    assert args.reference_checkpoint == tmp_path / "reference.pt"
    assert args.protected_group == []
    assert args.required_group == []
    assert args.min_required_group_mean_logp_delta == 0.0
    assert args.min_protected_mean_logp_delta == 0.0
    assert args.max_protected_row_worsened_fraction == 0.0
    assert args.max_protected_rank_worsened_fraction == 0.0
    assert args.max_protected_top_family_changed_rate == 0.0
    assert args.top_action_near_tie_margin == 1e-5
    assert args.max_protected_lost_target_non_near_tie_rate == 0.0
    assert args.allow_missing_context is False
    assert args.max_examples == 25
    assert args.output_json == tmp_path / "row_guard.json"


def test_row_guard_runtime_maps_args(tmp_path: Path) -> None:
    args = SimpleNamespace(
        dataset=tmp_path / "preference.npz",
        stack_config=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint=tmp_path / "checkpoint.pt",
        reference_checkpoint=tmp_path / "reference.pt",
        protected_group=["fixed"],
        required_group=["fixed", "learned"],
        min_required_group_mean_logp_delta=0.01,
        min_protected_mean_logp_delta=0.02,
        max_protected_row_worsened_fraction=0.03,
        max_protected_rank_worsened_fraction=0.04,
        max_protected_top_family_changed_rate=0.05,
        top_action_near_tie_margin=0.06,
        max_protected_lost_target_non_near_tie_rate=0.07,
        allow_missing_context=True,
        max_examples=9,
    )

    config = paired_outcome_preference_row_guard_config_from_args(args)

    assert config == PairedOutcomePreferenceRowGuardConfig(
        dataset_path=tmp_path / "preference.npz",
        stack_config_path=tmp_path / "stack.yaml",
        run_dir=tmp_path / "run",
        checkpoint_path=tmp_path / "checkpoint.pt",
        reference_checkpoint_path=tmp_path / "reference.pt",
        protected_groups=("fixed",),
        required_groups=("fixed", "learned"),
        min_required_group_mean_logp_delta=0.01,
        min_protected_mean_logp_delta=0.02,
        max_protected_row_worsened_fraction=0.03,
        max_protected_rank_worsened_fraction=0.04,
        max_protected_top_family_changed_rate=0.05,
        top_action_near_tie_margin=0.06,
        max_protected_lost_target_non_near_tie_rate=0.07,
        require_context=False,
        max_examples=9,
    )


def test_row_guard_reporting_preserves_compact_group_payload(tmp_path: Path) -> None:
    report = {
        "passed": False,
        "failures": ["protected_row_worsened_fraction_above:fixed:0.5>0"],
        "row_count": 2,
        "current_context_episode_count": 2,
        "reference_context_episode_count": 1,
        "groups": [
            {
                "label": "fixed",
                "protected": True,
                "required": True,
                "row_count": 2,
                "mean_target_logp_delta": -0.1,
                "row_worsened_fraction": 0.5,
                "rank_worsened_fraction": 0.25,
                "top_family_changed_rate": 0.1,
                "lost_target_non_near_tie_rate": 0.2,
                "ignored": "not exported",
            }
        ],
    }

    assert paired_outcome_preference_row_guard_output_payload(
        output_json=tmp_path / "row_guard.json",
        report=report,
    ) == {
        "output_json": (tmp_path / "row_guard.json").as_posix(),
        "passed": False,
        "failures": ["protected_row_worsened_fraction_above:fixed:0.5>0"],
        "row_count": 2,
        "current_context_episode_count": 2,
        "reference_context_episode_count": 1,
        "groups": [
            {
                "label": "fixed",
                "protected": True,
                "required": True,
                "row_count": 2,
                "mean_target_logp_delta": -0.1,
                "row_worsened_fraction": 0.5,
                "rank_worsened_fraction": 0.25,
                "top_family_changed_rate": 0.1,
                "lost_target_non_near_tie_rate": 0.2,
            }
        ],
    }
    assert paired_outcome_preference_row_guard_output_line(
        output_json=tmp_path / "row_guard.json",
        report=report,
    ) == (
        '{"current_context_episode_count": 2, "failures": '
        '["protected_row_worsened_fraction_above:fixed:0.5>0"], "groups": '
        '[{"label": "fixed", "lost_target_non_near_tie_rate": 0.2, '
        '"mean_target_logp_delta": -0.1, "protected": true, "rank_worsened_fraction": 0.25, '
        '"required": true, "row_count": 2, "row_worsened_fraction": 0.5, '
        '"top_family_changed_rate": 0.1}], '
        f'"output_json": "{(tmp_path / "row_guard.json").as_posix()}", '
        '"passed": false, "reference_context_episode_count": 1, "row_count": 2}'
    )


def test_row_guard_run_result_exit_code_tracks_guard_status(tmp_path: Path) -> None:
    assert (
        PairedOutcomePreferenceRowGuardRunResult(
            output_json=tmp_path / "row_guard.json",
            report={"passed": True},
        ).exit_code
        == 0
    )
    assert (
        PairedOutcomePreferenceRowGuardRunResult(
            output_json=tmp_path / "row_guard.json",
            report={"passed": False},
        ).exit_code
        == 1
    )


def test_row_guard_passes_clean_protected_rows(tmp_path: Path) -> None:
    report = evaluate_row_guard_rows(
        [
            _row(group="fixed", logp_delta=0.2, ref_rank=2, cur_rank=1),
            _row(group="learned", logp_delta=0.1, ref_rank=3, cur_rank=2),
        ],
        config=_config(tmp_path, protected_groups=("fixed",), required_groups=("fixed", "learned")),
        episode_count=2,
        train_rows=2,
        current_context_coverage=_coverage(2),
        reference_context_coverage=_coverage(2),
    )

    assert report["passed"] is True
    fixed = {group["label"]: group for group in report["groups"]}["fixed"]
    assert fixed["row_worsened_fraction"] == 0.0
    assert fixed["rank_worsened_fraction"] == 0.0


def test_row_guard_fails_protected_logp_and_rank_worsening(tmp_path: Path) -> None:
    report = evaluate_row_guard_rows(
        [
            _row(group="fixed", logp_delta=-0.01, ref_rank=1, cur_rank=2),
            _row(group="fixed", logp_delta=0.02, ref_rank=1, cur_rank=1),
        ],
        config=_config(tmp_path, protected_groups=("fixed",), required_groups=("fixed",)),
        episode_count=2,
        train_rows=2,
        current_context_coverage=_coverage(2),
        reference_context_coverage=_coverage(2),
    )

    assert report["passed"] is False
    assert any(item.startswith("protected_row_worsened_fraction_above:fixed") for item in report["failures"])
    assert any(item.startswith("protected_rank_worsened_fraction_above:fixed") for item in report["failures"])


def test_row_guard_allows_near_tie_lost_target_but_fails_non_near_tie(tmp_path: Path) -> None:
    near_tie_report = evaluate_row_guard_rows(
        [
            _row(
                group="fixed",
                logp_delta=0.0,
                ref_top=104,
                cur_top=124,
                target=104,
                top_margin=1e-6,
            )
        ],
        config=_config(tmp_path, protected_groups=("fixed",), required_groups=("fixed",)),
        episode_count=1,
        train_rows=1,
        current_context_coverage=_coverage(1),
        reference_context_coverage=_coverage(1),
    )
    non_near_tie_report = evaluate_row_guard_rows(
        [
            _row(
                group="fixed",
                logp_delta=0.0,
                ref_top=104,
                cur_top=124,
                target=104,
                top_margin=0.1,
            )
        ],
        config=_config(tmp_path, protected_groups=("fixed",), required_groups=("fixed",)),
        episode_count=1,
        train_rows=1,
        current_context_coverage=_coverage(1),
        reference_context_coverage=_coverage(1),
    )

    assert near_tie_report["passed"] is True
    assert non_near_tie_report["passed"] is False
    assert any(
        item.startswith("protected_lost_target_non_near_tie_rate_above:fixed")
        for item in non_near_tie_report["failures"]
    )


def test_row_guard_fails_top_family_change_on_protected_rows(tmp_path: Path) -> None:
    report = evaluate_row_guard_rows(
        [
            _row(
                group="fixed",
                logp_delta=0.01,
                ref_family="play",
                cur_family="attack",
            )
        ],
        config=_config(tmp_path, protected_groups=("fixed",), required_groups=("fixed",)),
        episode_count=1,
        train_rows=1,
        current_context_coverage=_coverage(1),
        reference_context_coverage=_coverage(1),
    )

    assert report["passed"] is False
    assert any(item.startswith("protected_top_family_changed_rate_above:fixed") for item in report["failures"])


def _config(
    tmp_path: Path,
    *,
    protected_groups: tuple[str, ...],
    required_groups: tuple[str, ...],
) -> PairedOutcomePreferenceRowGuardConfig:
    return PairedOutcomePreferenceRowGuardConfig(
        dataset_path=tmp_path / "dataset.npz",
        stack_config_path=tmp_path / "stack.yaml",
        run_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoint.pt",
        reference_checkpoint_path=tmp_path / "reference.pt",
        protected_groups=protected_groups,
        required_groups=required_groups,
        top_action_near_tie_margin=1e-5,
    )


def _row(
    *,
    group: str,
    logp_delta: float,
    ref_rank: int = 1,
    cur_rank: int = 1,
    ref_top: int = 104,
    cur_top: int = 104,
    target: int = 104,
    ref_family: str = "play",
    cur_family: str = "play",
    top_margin: float = 0.0,
) -> dict[str, object]:
    return {
        "group_label": group,
        "target_action": target,
        "target_family": "play",
        "target_logp_delta": logp_delta,
        "target_probability_delta": logp_delta,
        "current_target_rank": cur_rank,
        "reference_target_rank": ref_rank,
        "rank_worsened": cur_rank > ref_rank,
        "row_worsened": logp_delta < 0.0,
        "current_top_action": cur_top,
        "reference_top_action": ref_top,
        "current_top_family": cur_family,
        "reference_top_family": ref_family,
        "top_family_changed": cur_family != ref_family,
        "lost_target_top_action": ref_top == target and cur_top != target,
        "current_top_over_target_logp_margin": top_margin,
    }


def _coverage(count: int) -> dict[str, object]:
    return {
        "episode_count": count,
        "context_episode_count": count,
        "missing_context_episode_count": 0,
        "missing_context_policy_ids": [],
    }
