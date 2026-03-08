from __future__ import annotations

from pathlib import Path

import pytest

from weiss_rl.config import (
    canonical_config_json,
    compute_config_hash256,
    load_stack_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_load_stack_config_merges_locked_components() -> None:
    repo_root = _repo_root()
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


def test_load_stack_config_allows_minimal_smoke_index() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/stack_smoke.yaml")

    assert stack.schema_version == 1
    assert stack.components == {}
    assert stack.seed_sets == {}
    assert stack.config.system is None


def test_canonical_config_hash_is_stable() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    stack = load_stack_config(repo_root / "configs/rl_stack_locked.yaml")

    assert canonical_config_json(stack).startswith('{"components":{"compute_budget":')
    assert compute_config_hash256(stack) == compute_config_hash256(
        load_stack_config(repo_root / "configs/rl_stack_locked.yaml")
    )



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


def test_load_stack_config_rejects_mixed_top_level_config_docs(tmp_path: Path) -> None:
    stack_path = tmp_path / "configs" / "stack.yaml"
    stack_path.parent.mkdir(parents=True)
    stack_path.write_text(
        "rl_stack_locked:\n"
        "  components: {}\n"
        "  seed_sets: {}\n"
        "minimal_loop:\n"
        "  mode: fast\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra top-level keys: minimal_loop"):
        load_stack_config(stack_path)


def test_load_stack_config_rejects_unknown_component_fields(tmp_path: Path) -> None:
    repo_root = _repo_root()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True)

    system_text = (repo_root / "configs/system_locked.yaml").read_text(encoding="utf-8")
    (configs_dir / "system_locked.yaml").write_text(system_text + "  hidden_toggle: true\n", encoding="utf-8")
    (configs_dir / "stack.yaml").write_text(
        "rl_stack_locked:\n"
        "  components:\n"
        "    system: configs/system_locked.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="system has unsupported keys: hidden_toggle"):
        load_stack_config(configs_dir / "stack.yaml")
