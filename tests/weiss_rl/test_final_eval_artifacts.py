from __future__ import annotations

import csv
import json
from pathlib import Path

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.eval import final_eval as final_eval_module
from weiss_rl.eval.final.artifacts import (
    final_eval_matchup_manifest_rows,
    write_final_eval_artifacts,
)


def test_final_eval_artifact_writer_preserves_direct_output_shape(tmp_path: Path) -> None:
    output_dir = tmp_path / "final_eval"
    output_dir.mkdir(parents=True)
    payload = {
        "metadata": {"selection": {"mode": "explicit", "policy_count": 2}},
        "policy_ids": ["policy_a", "policy_b"],
        "posterior_samples": {"policy_ids": ["policy_a", "policy_b"], "values": [[[], []], [[], []]]},
        "matrices": {
            "mean": {"policy_ids": ["policy_a", "policy_b"], "values": [[0.5, 0.75], [0.25, 0.5]]},
        },
        "matchups": [],
    }
    matchup_results = [
        {
            "focal_policy_id": "policy_a",
            "opponent_policy_id": "policy_b",
            "matchup_dir": output_dir / "matchups" / "00_policy_a__vs__01_policy_b",
            "summary": {
                "paired_seeds": 3,
                "observed_paired_seeds": 3,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "budget",
            },
            "records": [],
            "replay_samples": [],
        }
    ]

    write_final_eval_artifacts(output_dir=output_dir, payload=payload, matchup_results=matchup_results)
    rows = final_eval_matchup_manifest_rows(output_dir=output_dir, matchup_results=matchup_results)

    assert final_eval_module._write_final_eval_artifacts is write_final_eval_artifacts
    assert json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")) == payload["metadata"]
    assert json.loads((output_dir / "policy_set.json").read_text(encoding="utf-8")) == {
        "policy_ids": ["policy_a", "policy_b"]
    }
    assert (
        json.loads((output_dir / "posterior_samples.json").read_text(encoding="utf-8")) == payload["posterior_samples"]
    )
    assert (output_dir / "matrices" / "mean.json").is_file()
    with (output_dir / "matrices" / "mean.csv").open("r", encoding="utf-8") as handle:
        assert list(csv.reader(handle)) == [
            ["focal_policy_id", "policy_a", "policy_b"],
            ["policy_a", "0.5", "0.75"],
            ["policy_b", "0.25", "0.5"],
        ]
    with (output_dir / "matchups.csv").open("r", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert rows[0]["matchup_dir"] == "matchups/00_policy_a__vs__01_policy_b"
    assert manifest[0]["matchup_dir"] == "matchups/00_policy_a__vs__01_policy_b"
    assert manifest[0]["has_payoff_samples"] == "True"


def test_final_eval_artifact_writer_preserves_canonical_layout_exports(tmp_path: Path) -> None:
    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.ensure_directories()
    payload = {
        "metadata": {"selection": {"mode": "explicit", "policy_count": 2}},
        "policy_ids": ["policy_a", "policy_b"],
        "posterior_samples": {
            "policy_ids": ["policy_a", "policy_b"],
            "values": [[[0.5], [0.75]], [[0.25], [0.5]]],
        },
        "matrices": {
            "mean": {"policy_ids": ["policy_a", "policy_b"], "values": [[0.5, 0.75], [0.25, 0.5]]},
            "paired_seed_count": {"policy_ids": ["policy_a", "policy_b"], "values": [[3, 4], [4, 3]]},
        },
        "matchups": [],
    }
    matchup_results = [
        {
            "focal_policy_id": "policy_a",
            "opponent_policy_id": "policy_b",
            "matchup_dir": layout.final_eval_matchups_dir / "00_policy_a__vs__01_policy_b",
            "summary": {
                "paired_seeds": 4,
                "observed_paired_seeds": 4,
                "excluded_paired_seeds": 0,
                "has_payoff_samples": True,
                "stop_reason": "budget",
                "summary": {"games": 8, "truncations": 2},
            },
            "records": [],
            "replay_samples": [],
        }
    ]

    write_final_eval_artifacts(
        output_dir=layout.final_eval_dir,
        payload=payload,
        matchup_results=matchup_results,
    )

    assert layout.final_eval_posterior_samples_npz().is_file()
    assert layout.final_eval_payoff_matrix_csv("p_mean").read_text(encoding="utf-8").splitlines() == [
        "focal_policy_id,policy_a,policy_b",
        "policy_a,0.5,0.75",
        "policy_b,0.25,0.5",
    ]
    assert layout.truncation_heatmap_csv().read_text(encoding="utf-8").splitlines() == [
        "focal_policy_id,policy_a,policy_b",
        "policy_a,0.0,0.25",
        "policy_b,0.25,0.0",
    ]
    replay_verification = json.loads(layout.replay_verification_json().read_text(encoding="utf-8"))
    replay_index = json.loads(layout.replay_index_json().read_text(encoding="utf-8"))
    hashes = json.loads(layout.final_eval_aggregate_hashes_json().read_text(encoding="utf-8"))
    assert replay_verification["status"] == "not_sampled"
    assert replay_verification["index_path"] == "replays/index.json"
    assert replay_index == {"kind": "replay_index_v1", "samples": []}
    assert "eval/final_eval/summary.json" in hashes["artifacts"]
    assert "eval/diagnostics/replay_verification.json" in hashes["artifacts"]
    assert "replays/index.json" in hashes["artifacts"]
