from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from weiss_rl.eval.harness import EvalGameRecord
from weiss_rl.eval.heuristic_public import ActionCatalog


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "python" / "scripts" / "b2_disagreement_audit.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("b2_disagreement_audit_script", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_record(
    *,
    pair_index: int,
    swap_index: int,
    episode_seed: int,
    focal_policy_id: str = "learner",
    opponent_policy_id: str = "B2 HeuristicPublic",
) -> EvalGameRecord:
    if swap_index == 0:
        seat0_policy_id = focal_policy_id
        seat1_policy_id = opponent_policy_id
        focal_seat = 0
    else:
        seat0_policy_id = opponent_policy_id
        seat1_policy_id = focal_policy_id
        focal_seat = 1
    return EvalGameRecord(
        pair_index=pair_index,
        swap_index=swap_index,
        episode_index=pair_index * 2 + swap_index,
        episode_seed=episode_seed,
        episode_key=f"episode-{pair_index}-{swap_index}",
        episode_key64=pair_index * 10 + swap_index,
        config_hash256="a" * 64,
        spec_hash256="b" * 64,
        focal_policy_id=focal_policy_id,
        opponent_policy_id=opponent_policy_id,
        seat0_policy_id=seat0_policy_id,
        seat1_policy_id=seat1_policy_id,
        focal_seat=focal_seat,
        outcome="W" if focal_seat == 0 else "L",
        terminated=True,
        truncated=False,
        engine_status=0,
        decision_count=1,
        tick_count=1,
        no_progress_count=0,
        termination_reason="terminated",
    )


def test_load_matchup_source_extracts_unique_paired_seeds_in_first_seen_order(tmp_path: Path) -> None:
    module = _load_script_module()
    episodes_path = tmp_path / "episodes.jsonl"
    records = [
        _make_record(pair_index=2, swap_index=0, episode_seed=42),
        _make_record(pair_index=2, swap_index=1, episode_seed=42),
        _make_record(pair_index=0, swap_index=0, episode_seed=7),
        _make_record(pair_index=0, swap_index=1, episode_seed=7),
        _make_record(pair_index=1, swap_index=0, episode_seed=42),
        _make_record(pair_index=1, swap_index=1, episode_seed=42),
    ]
    episodes_path.write_text(
        "\n".join(json.dumps(record.to_dict(), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    source = module._load_matchup_source(episodes_path)

    assert source.focal_policy_id == "learner"
    assert source.opponent_policy_id == "B2 HeuristicPublic"
    assert source.paired_seeds == (42, 7)


def test_resolve_requested_policy_id_accepts_registry_alias_for_train_policy_id() -> None:
    module = _load_script_module()

    resolved = module._resolve_requested_policy_id(
        requested_policy_id="policy_000015",
        source_focal_policy_id="train_u300_p15",
    )

    assert resolved == "policy_000015"


def test_annotate_step_diff_uses_action_catalog_decoder() -> None:
    module = _load_script_module()
    decoder = ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_space_size": 7,
                "pass_action_id": 0,
                "constants": [["MAX_HAND", 50], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 3]],
                "attack_type_encoding": [["front", 0], ["side", 1]],
                "families": [
                    {"name": "pass", "base": 0, "count": 1},
                    {"name": "attack", "base": 1, "count": 6},
                ],
            }
        }
    )

    annotated = module._annotate_step_diff(
        {
            "recorded_action": 0,
            "policy_a_top_action": {"action": 0},
            "policy_b_top_action": {"action": 1},
        },
        decoder=decoder,
    )

    assert annotated["recorded_action_family"] == "pass"
    assert annotated["policy_a_top_action_family"] == "pass"
    assert annotated["policy_b_top_action_family"] == "attack"


def test_aggregate_audit_summary_ranks_repeated_family_pairs_and_weighted_means(tmp_path: Path) -> None:
    module = _load_script_module()
    source = module.MatchupSource(
        focal_policy_id="learner",
        opponent_policy_id="B2 HeuristicPublic",
        config_hash256="a" * 64,
        spec_hash256="b" * 64,
        paired_seeds=(42, 7),
    )
    bundle_summaries = [
        {
            "bundle_path": "/tmp/bundle-1.zip",
            "report_path": "/tmp/report-1.json",
            "pair_index": 0,
            "swap_index": 0,
            "episode_seed": 42,
            "replay_key64": "1111",
            "summary": {"max_total_variation": 0.6, "mean_total_variation": 0.2},
            "compared_steps": 10,
            "inspected_step_count": 4,
            "family_pair_counts": [
                {"policy_a_family": "attack", "policy_b_family": "pass", "count": 2},
                {"policy_a_family": "attack", "policy_b_family": "attack", "count": 1},
                {"policy_a_family": "clock_from_hand", "policy_b_family": "clock_from_hand", "count": 1},
            ],
            "policy_a_family_counts": [
                {"family": "attack", "count": 3},
                {"family": "clock_from_hand", "count": 1},
            ],
            "policy_b_family_counts": [
                {"family": "pass", "count": 2},
                {"family": "attack", "count": 1},
                {"family": "clock_from_hand", "count": 1},
            ],
            "recorded_family_counts": [
                {"family": "attack", "count": 2},
                {"family": "pass", "count": 1},
                {"family": "clock_from_hand", "count": 1},
            ],
            "action_label_pair_counts": [
                {"policy_a_action_label": "attack(slot=0, attack_type=direct)", "policy_b_action_label": "pass", "count": 2},
                {
                    "policy_a_action_label": "attack(slot=0, attack_type=direct)",
                    "policy_b_action_label": "attack(slot=0, attack_type=direct)",
                    "count": 1,
                },
                {
                    "policy_a_action_label": "clock_from_hand(hand_index=0)",
                    "policy_b_action_label": "clock_from_hand(hand_index=0)",
                    "count": 1,
                },
            ],
            "policy_a_action_label_counts": [
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 3},
                {"action_label": "clock_from_hand(hand_index=0)", "count": 1},
            ],
            "policy_b_action_label_counts": [
                {"action_label": "pass", "count": 2},
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 1},
                {"action_label": "clock_from_hand(hand_index=0)", "count": 1},
            ],
            "top_examples": [],
        },
        {
            "bundle_path": "/tmp/bundle-2.zip",
            "report_path": "/tmp/report-2.json",
            "pair_index": 1,
            "swap_index": 1,
            "episode_seed": 7,
            "replay_key64": "2222",
            "summary": {"max_total_variation": 0.9, "mean_total_variation": 0.4},
            "compared_steps": 20,
            "inspected_step_count": 5,
            "family_pair_counts": [
                {"policy_a_family": "attack", "policy_b_family": "pass", "count": 2},
                {"policy_a_family": "main_move", "policy_b_family": "pass", "count": 2},
                {"policy_a_family": "attack", "policy_b_family": "attack", "count": 1},
            ],
            "policy_a_family_counts": [
                {"family": "attack", "count": 3},
                {"family": "main_move", "count": 2},
            ],
            "policy_b_family_counts": [
                {"family": "pass", "count": 4},
                {"family": "attack", "count": 1},
            ],
            "recorded_family_counts": [
                {"family": "main_move", "count": 2},
                {"family": "attack", "count": 2},
                {"family": "pass", "count": 1},
            ],
            "action_label_pair_counts": [
                {"policy_a_action_label": "attack(slot=0, attack_type=direct)", "policy_b_action_label": "pass", "count": 2},
                {"policy_a_action_label": "main_move(from_slot=0, to_slot=2)", "policy_b_action_label": "pass", "count": 2},
                {
                    "policy_a_action_label": "attack(slot=0, attack_type=direct)",
                    "policy_b_action_label": "attack(slot=0, attack_type=direct)",
                    "count": 1,
                },
            ],
            "policy_a_action_label_counts": [
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 3},
                {"action_label": "main_move(from_slot=0, to_slot=2)", "count": 2},
            ],
            "policy_b_action_label_counts": [
                {"action_label": "pass", "count": 4},
                {"action_label": "attack(slot=0, attack_type=direct)", "count": 1},
            ],
            "top_examples": [],
        },
    ]

    summary = module._aggregate_audit_summary(
        source=source,
        policy_id="learner",
        episodes_jsonl=tmp_path / "source.jsonl",
        run_dir=tmp_path / "source_run",
        output_run_dir=tmp_path / "output_run",
        episodes_path=tmp_path / "audit" / "episodes.jsonl",
        game_count=4,
        bundle_summaries=bundle_summaries,
        inspection_errors=[],
    )

    assert summary["status"] == "ok"
    assert summary["games"] == 4
    assert summary["bundle_count"] == 2
    assert summary["top_family_pairs"][0] == {
        "policy_a_family": "attack",
        "policy_b_family": "pass",
        "count": 4,
    }
    assert summary["top_policy_a_families"][0] == {"family": "attack", "count": 6}
    assert summary["top_policy_b_families"][0] == {"family": "pass", "count": 6}
    assert summary["top_recorded_families"][0] == {"family": "attack", "count": 4}
    assert summary["top_action_label_pairs"][0] == {
        "policy_a_action_label": "attack(slot=0, attack_type=direct)",
        "policy_b_action_label": "pass",
        "count": 4,
    }
    assert summary["top_policy_a_action_labels"][0] == {
        "action_label": "attack(slot=0, attack_type=direct)",
        "count": 6,
    }
    assert summary["top_policy_b_action_labels"][0] == {"action_label": "pass", "count": 6}
    assert summary["max_total_variation"] == pytest.approx(0.9)
    assert summary["mean_total_variation"] == pytest.approx((0.2 * 10 + 0.4 * 20) / 30)


def test_materialize_audit_bundle_copy_uses_pair_and_swap_suffix(tmp_path: Path) -> None:
    module = _load_script_module()
    source_bundle = tmp_path / "replays" / "bundles" / "replay_deadbeef.zip"
    source_bundle.parent.mkdir(parents=True, exist_ok=True)
    source_bundle.write_text("bundle-bytes", encoding="utf-8")
    bundle_copies_dir = tmp_path / "audit" / "replay_bundles"
    bundle_copies_dir.mkdir(parents=True, exist_ok=True)

    copied = module._materialize_audit_bundle_copy(
        source_bundle_path=source_bundle,
        bundle_copies_dir=bundle_copies_dir,
        pair_index=7,
        swap_index=1,
    )

    assert copied == bundle_copies_dir / "replay_deadbeef_pair007_swap1.zip"
    assert copied.read_text(encoding="utf-8") == "bundle-bytes"
