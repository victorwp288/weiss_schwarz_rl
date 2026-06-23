from __future__ import annotations

import json
from pathlib import Path

import pytest
from weiss_rl.config import load_study_config
from weiss_rl.metagame import build_sensitivity_report

from .metagame_sensitivity_test_support import (
    STUDY_CONFIG_PATH,
    write_final_eval_fixture,
    write_final_eval_summary,
)


def test_build_sensitivity_report_rejects_mismatched_final_eval_policy_index(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["matchups"][0]["focal_policy_index"] = 2
    write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"focal_policy_index=2 does not match policy_ids position"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=study.metagame,
            sensitivity_config=study.sensitivity,
        )


def test_build_sensitivity_report_rejects_duplicate_final_eval_matchup(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["matchups"][-1] = dict(payload["matchups"][0])
    write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"duplicate canonical matchup"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=study.metagame,
            sensitivity_config=study.sensitivity,
        )


def test_build_sensitivity_report_rejects_mismatched_matchup_episodes_path(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    target = payload["matchups"][0]
    source = next(
        matchup
        for matchup in payload["matchups"][1:]
        if (
            matchup["focal_policy_id"],
            matchup["opponent_policy_id"],
        )
        != (
            target["focal_policy_id"],
            target["opponent_policy_id"],
        )
    )
    target["episodes_path"] = source["episodes_path"]
    write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"must equal canonical final_eval artifact path"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=study.metagame,
            sensitivity_config=study.sensitivity,
        )


def test_build_sensitivity_report_rejects_in_tree_same_pair_rogue_episodes_path(tmp_path: Path) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    target = payload["matchups"][0]
    rogue_path = final_eval_dir / "matchups" / "rogue_same_pair" / "episodes.jsonl"
    canonical_path = final_eval_dir / target["episodes_path"]
    rogue_path.parent.mkdir(parents=True, exist_ok=True)
    rogue_path.write_text(canonical_path.read_text(encoding="utf-8"), encoding="utf-8")
    target["episodes_path"] = rogue_path.relative_to(final_eval_dir).as_posix()
    write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=r"must equal canonical final_eval artifact path"):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=study.metagame,
            sensitivity_config=study.sensitivity,
        )


@pytest.mark.parametrize(
    ("episodes_path", "message"),
    [
        ("../outside.jsonl", r"resolves outside the final_eval root"),
        ("/tmp/outside.jsonl", r"must be relative to the final_eval root"),
    ],
)
def test_build_sensitivity_report_rejects_unsafe_matchup_episodes_path(
    tmp_path: Path, episodes_path: str, message: str
) -> None:
    final_eval_dir = write_final_eval_fixture(tmp_path)
    study = load_study_config(STUDY_CONFIG_PATH)

    summary_path = final_eval_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["matchups"][0]["episodes_path"] = episodes_path
    write_final_eval_summary(final_eval_dir, payload)

    with pytest.raises(ValueError, match=message):
        build_sensitivity_report(
            final_eval_dir=final_eval_dir,
            out_dir=tmp_path / "sensitivity",
            metagame_config=study.metagame,
            sensitivity_config=study.sensitivity,
        )
