from __future__ import annotations

from pathlib import Path

from weiss_rl.experiments.paired_outcome_preference_row_guard import (
    PairedOutcomePreferenceRowGuardConfig,
    evaluate_row_guard_rows,
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
