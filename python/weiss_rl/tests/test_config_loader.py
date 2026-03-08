from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.config import load_stack_config


def test_load_stack_config_merges_locked_components() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    stack = load_stack_config(repo_root / "configs/rl_stack_locked.yaml")

    assert stack.root == repo_root
    assert stack.schema_version == 1
    assert stack.components["system"] == repo_root / "configs/system_locked.yaml"
    assert stack.seed_sets["dev_eval"] == repo_root / "configs/seeds/dev_eval_seeds.txt"

    assert stack.config.system is not None
    assert stack.config.system.total_envs == 96
    assert stack.config.model is not None
    assert stack.config.model.dropout.ablation == 0.1
    assert stack.config.evaluation is not None
    assert stack.config.evaluation.final_policy_set_selection.tie_break == "lowest_policy_id"
    assert stack.config.reproducibility is not None
    assert stack.config.reproducibility.seed_derivation.base_seed64 == 20260212


def test_load_stack_config_allows_minimal_loop_index() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    stack = load_stack_config(repo_root / "configs/minimal_loop.yaml")

    assert stack.components == {}
    assert stack.seed_sets == {}
    assert stack.config.system is None


def test_load_stack_config_rejects_unknown_component(tmp_path: Path) -> None:
    stack_path = tmp_path / "configs" / "stack.yaml"
    stack_path.parent.mkdir(parents=True)
    stack_path.write_text(
        "rl_stack_locked:\n"
        "  components:\n"
        "    mystery: configs/mystery.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported component"):
        load_stack_config(stack_path)
