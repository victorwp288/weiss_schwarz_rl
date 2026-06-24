from __future__ import annotations

from pathlib import Path

import pytest
import torch
import weiss_rl.diagnostics.probes.checkpoint_family_bias_entrypoint as script


def test_parse_family_bias_offsets_merges_repeated_families() -> None:
    offsets = script._parse_family_bias_offsets(["attack=0.25", "encore_decline=-0.5", "attack=0.75"])

    assert offsets == {
        "attack": pytest.approx(1.0),
        "encore_decline": pytest.approx(-0.5),
    }


def test_family_bias_script_parser_accepts_repeated_biases(tmp_path: Path) -> None:
    args = script._build_parser().parse_args(
        [
            "--source-checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--source-run-dir",
            str(tmp_path / "source"),
            "--output-run-dir",
            str(tmp_path / "out"),
            "--policy-id",
            "biased_policy",
            "--family-bias",
            "attack=0.2",
            "--family-bias",
            "encore_decline=-0.3",
        ]
    )

    assert args.policy_id == "biased_policy"
    assert args.family_bias == ["attack=0.2", "encore_decline=-0.3"]


def test_family_bias_indices_rejects_unknown_families() -> None:
    with pytest.raises(ValueError, match="unknown action families"):
        script._family_bias_indices({"missing": 0.5}, family_index={"attack": 8})


def test_apply_family_bias_offsets_clones_and_changes_only_requested_indices() -> None:
    original = torch.tensor([1.0, 2.0, 3.0])
    state = {"policy_head.family_bias": original, "other": torch.tensor([5.0])}

    changed = script._apply_family_bias_offsets(
        state,
        family_offsets={"attack": 0.5, "level_up": -1.0},
        family_index={"attack": 1, "level_up": 2},
    )

    assert changed["policy_head.family_bias"].tolist() == pytest.approx([1.0, 2.5, 2.0])
    assert original.tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert changed["policy_head.family_bias"] is not original
    assert changed["other"] is state["other"]
