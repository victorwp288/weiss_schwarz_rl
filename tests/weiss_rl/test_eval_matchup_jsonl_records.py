from __future__ import annotations

import json
from pathlib import Path

from weiss_rl.artifacts.reproducibility import key256_to_short64
from weiss_rl.eval.harness import GameResult, run_seat_swapped_matchup

from tests.weiss_rl.eval_harness_test_support import _CONFIG_HASH256, _RUN_ID256, _SPEC_HASH256, _FakeRunner


def test_run_seat_swapped_matchup_emits_per_game_records_jsonl(tmp_path: Path) -> None:
    runner = _FakeRunner(
        [
            GameResult(episode_seed=7, terminated=True, truncated=False, winner_seat=0, engine_status=0),
            GameResult(episode_seed=7, terminated=True, truncated=False, winner_seat=None, engine_status=5),
        ]
    )

    result = run_seat_swapped_matchup(
        focal_policy_id="champion",
        opponent_policy_id="baseline",
        paired_seeds=[7],
        runner=runner,
        episodes_path=tmp_path / "episodes.jsonl",
        run_id256=_RUN_ID256,
        config_hash256=_CONFIG_HASH256,
        spec_hash256=_SPEC_HASH256,
    )

    assert len(result.records) == 2
    assert result.episodes_path == tmp_path / "episodes.jsonl"
    assert [record.outcome for record in result.records] == ["W", "D"]
    assert [record.engine_status for record in result.records] == [0, 5]
    assert [record.truncated for record in result.records] == [False, False]
    assert [record.config_hash256 for record in result.records] == [_CONFIG_HASH256, _CONFIG_HASH256]
    assert [record.spec_hash256 for record in result.records] == [_SPEC_HASH256, _SPEC_HASH256]
    assert [record.episode_key64 for record in result.records] == [
        key256_to_short64(bytes.fromhex(record.episode_key)) for record in result.records
    ]

    payloads = [json.loads(line) for line in result.episodes_path.read_text(encoding="utf-8").splitlines()]
    assert [payload["swap_index"] for payload in payloads] == [0, 1]
    assert [payload["seat0_policy_id"] for payload in payloads] == ["champion", "baseline"]
    assert [payload["seat1_policy_id"] for payload in payloads] == ["baseline", "champion"]
    assert [payload["seat0_deck"] for payload in payloads] == [
        "preset:main_deck_5hy_yotsuba_v1",
        "preset:main_deck_5hy_yotsuba_v1",
    ]
    assert [payload["seat1_deck"] for payload in payloads] == [
        "preset:main_deck_5hy_yotsuba_v1",
        "preset:main_deck_5hy_yotsuba_v1",
    ]
    assert [payload["outcome"] for payload in payloads] == ["W", "D"]
    assert [payload["engine_status"] for payload in payloads] == [0, 5]
    assert [payload["config_hash256"] for payload in payloads] == [_CONFIG_HASH256, _CONFIG_HASH256]
    assert [payload["spec_hash256"] for payload in payloads] == [_SPEC_HASH256, _SPEC_HASH256]
    assert [payload["run_id256"] for payload in payloads] == [_RUN_ID256, _RUN_ID256]
    assert [payload["episode_key64"] for payload in payloads] == [record.episode_key64 for record in result.records]
    assert [call.swap_index for call in runner.calls] == [0, 1]
