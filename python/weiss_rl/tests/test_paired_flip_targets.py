from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.eval.export import load_eval_game_records
from weiss_rl.experiments.paired_flip_targets import (
    PairedFlipTargetsConfig,
    build_paired_flip_targets,
)


def _write_eval_tree(root: Path, label: str, opponent: str, outcomes: list[str]) -> Path:
    matchup_dir = root / label / "matchups" / opponent.replace(" ", "_")
    matchup_dir.mkdir(parents=True)
    episodes_path = matchup_dir / "episodes.jsonl"
    records = []
    for index, outcome in enumerate(outcomes):
        records.append(
            {
                "config_hash256": "a" * 64,
                "engine_status": 0,
                "episode_index": index,
                "episode_key": f"{index:064x}",
                "episode_key64": index,
                "episode_seed": 1000 + (index // 2),
                "focal_policy_id": label,
                "focal_seat": index % 2,
                "opponent_policy_id": opponent,
                "outcome": outcome,
                "pair_index": index // 2,
                "seat0_policy_id": label if index % 2 == 0 else opponent,
                "seat1_policy_id": opponent if index % 2 == 0 else label,
                "spec_hash256": "b" * 64,
                "swap_index": index % 2,
                "terminated": True,
                "truncated": False,
                "decision_count": 100 + index,
                "main_move_actions": 40 + index,
                "pass_actions": 10 + index,
                "pass_with_nonpass_available": 3 + index,
            }
        )
    episodes_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n")
    summary_path = matchup_dir / "matchup_summary.json"
    summary_path.write_text(json.dumps({"opponent_policy_id": opponent}, sort_keys=True), encoding="utf-8")
    targeted_summary = root / f"{label}_summary.json"
    targeted_summary.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "games": len(outcomes),
                        "mean": outcomes.count("W") / len(outcomes),
                        "opponent_policy_id": opponent,
                        "summary_path": summary_path.as_posix(),
                        "wins": outcomes.count("W"),
                    }
                ]
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return targeted_summary


def test_paired_flip_targets_writes_complete_candidate_episode_sets_and_tags(tmp_path: Path) -> None:
    baseline_opponent = "seed_inner_policy_000002"
    candidate_opponent = "seed_outer_seed_inner_policy_000002"
    baseline = _write_eval_tree(tmp_path, "baseline", baseline_opponent, ["W", "W", "L", "L"])
    candidate = _write_eval_tree(tmp_path, "candidate", candidate_opponent, ["L", "W", "W", "L"])
    pool_jsonl = tmp_path / "opponent_pool.jsonl"
    pool_jsonl.write_text(json.dumps({"hard_negative_ids": [baseline_opponent]}, sort_keys=True) + "\n")

    report = build_paired_flip_targets(
        PairedFlipTargetsConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            opponents=(candidate_opponent,),
            learned_opponents=(baseline_opponent,),
            opponent_pool_jsonls=(pool_jsonl,),
            episode_sets_dir=tmp_path / "episode_sets",
        )
    )

    assert report["target_count"] == 1
    target = report["targets"][0]
    assert target["target_id"].startswith("sha256:")
    assert target["opponent_policy_id"] == candidate_opponent
    assert target["baseline_opponent_policy_id"] == baseline_opponent
    assert target["candidate_opponent_policy_id"] == candidate_opponent
    assert target["tags"] == ["hard_negative", "learned"]
    assert target["pair_index"] == 0
    assert target["swap_index"] == 0
    assert target["baseline"]["outcome"] == "W"
    assert target["candidate"]["outcome"] == "L"
    assert report["seed_plan"]["hard_negative"][candidate_opponent] == [1000]

    episode_set = report["episode_sets"][0]
    assert episode_set["source"] == "candidate"
    records = load_eval_game_records(Path(episode_set["path"]))
    assert len(records) == 2
    assert {record.swap_index for record in records} == {0, 1}
    assert {record.pair_index for record in records} == {0}


def test_paired_flip_targets_respects_pair_filter_and_target_cap(tmp_path: Path) -> None:
    baseline = _write_eval_tree(tmp_path, "baseline", "B1 NoLeague baseline", ["W", "W", "W", "W", "W", "W"])
    candidate = _write_eval_tree(tmp_path, "candidate", "B1 NoLeague baseline", ["L", "L", "L", "L", "L", "L"])

    report = build_paired_flip_targets(
        PairedFlipTargetsConfig(
            baseline_summary_json=baseline,
            candidate_summary_json=candidate,
            opponents=("B1 NoLeague baseline",),
            pair_index_min=1,
            max_targets_per_opponent=2,
            episode_source="both",
            episode_sets_dir=tmp_path / "episode_sets",
        )
    )

    assert report["target_count"] == 2
    assert [target["pair_index"] for target in report["targets"]] == [1, 1]
    assert [target["swap_index"] for target in report["targets"]] == [0, 1]
    assert {episode_set["source"] for episode_set in report["episode_sets"]} == {"baseline", "candidate"}
    for episode_set in report["episode_sets"]:
        records = load_eval_game_records(Path(episode_set["path"]))
        assert len(records) == 2
        assert {record.swap_index for record in records} == {0, 1}
        assert {record.pair_index for record in records} == {1}
