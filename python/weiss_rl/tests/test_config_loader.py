from __future__ import annotations

import json
from pathlib import Path

import pytest

from weiss_rl.artifacts.reproducibility import canonical_json_bytes, parse_seed_file, sha256_hex
from weiss_rl.config import canonical_config_dict, canonical_config_json, compute_config_hash256, load_stack_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_load_stack_config_reads_typed_thesis_preset() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/presets/typed_thesis_locked.yaml")

    assert stack.root == repo_root
    assert stack.schema_version == 2
    assert stack.config.experiment is not None
    assert stack.config.experiment.role == "main"
    assert stack.config.model is not None
    assert stack.config.model.encoder_kind == "typed_v1"
    assert stack.config.training is not None
    assert stack.config.training.algorithm == "impala_vtrace_gru"
    assert stack.config.league is not None
    assert stack.config.league.enabled is True
    assert stack.seed_sets["report_eval"] == repo_root / "configs/seeds/report_eval_seeds.txt"


def test_canonical_config_helpers_are_stable_json() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/thesis/main_league.yaml")

    payload = canonical_config_dict(stack)
    canonical_json = canonical_config_json(stack)

    assert json.loads(json.dumps(payload)) == payload
    assert canonical_json == canonical_json_bytes(payload).decode("utf-8")
    assert compute_config_hash256(stack) == sha256_hex(canonical_json_bytes(payload))


@pytest.mark.parametrize(
    ("path", "role", "algorithm"),
    [
        ("configs/thesis/main_league.yaml", "main", "impala_vtrace_gru"),
        ("configs/thesis/main_league_auto_gpu.yaml", "main", "impala_vtrace_gru"),
        ("configs/thesis/b1_noleague.yaml", "baseline_noleague", "impala_vtrace_gru"),
        ("configs/thesis/ablations/no_gru.yaml", "baseline_norecurrence", "impala_vtrace_ff"),
        ("configs/thesis/ablations/ppo_lite.yaml", "baseline_ppo_lite", "ppo_lite_masked_v1"),
        ("configs/thesis/ablations/terminal_only_reward.yaml", "ablation_reward", "impala_vtrace_gru"),
    ],
)
def test_public_thesis_training_configs_load(path: str, role: str, algorithm: str) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / path)

    assert stack.config.experiment is not None
    assert stack.config.experiment.role == role
    assert stack.config.training is not None
    assert stack.config.training.algorithm == algorithm
    assert stack.config.environment is not None
    assert stack.config.environment.deck_pool == ("preset:main_deck_5hy_yotsuba_v1",)


def test_b1_noleague_disables_league_and_heuristic_sampling() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/thesis/b1_noleague.yaml")

    assert stack.config.league is not None
    assert stack.config.league.enabled is False
    assert stack.config.league.sampling.heuristic_public_mix_fraction == pytest.approx(0.0)
    assert stack.config.league.sampling.noleague_baseline_mix_fraction == pytest.approx(0.0)


@pytest.mark.parametrize(
    "path",
    [
        "configs/thesis/final_eval.yaml",
        "configs/thesis/final_eval_gpu.yaml",
        "configs/presets/structured_acceptance_standard_thesis_eval.yaml",
    ],
)
def test_final_eval_configs_include_fixed_anchor_panel(path: str) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / path)

    assert stack.config.evaluation is not None
    selection = stack.config.evaluation.final_policy_set_selection
    assert selection is not None
    assert selection.include_heuristic_public_anchors_b2_b3_b4 is True
    assert selection.fixed_anchor_set_v1.required == (
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "B3 HeuristicPublicAggro",
        "B4 HeuristicPublicControl",
    )


@pytest.mark.parametrize(
    "path",
    [
        "configs/presets/structured_acceptance_standard.yaml",
        "configs/presets/structured_acceptance_standard_auto_gpu.yaml",
        "configs/presets/structured_acceptance_standard_multideck.yaml",
        "configs/presets/structured_acceptance_standard_thesis_eval.yaml",
        "configs/thesis/multideck_exploratory.yaml",
    ],
)
def test_compatibility_presets_load(path: str) -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / path)

    assert stack.schema_version == 2
    assert stack.config.training is not None
    assert stack.config.model is not None


def test_multideck_config_is_explicitly_not_fixed_deck() -> None:
    repo_root = _repo_root()
    stack = load_stack_config(repo_root / "configs/thesis/multideck_exploratory.yaml")

    assert stack.config.environment is not None
    assert len(stack.config.environment.deck_pool) > 1
    assert "preset:main_deck_5hy_yotsuba_v1" in stack.config.environment.deck_pool


def test_hardneg_provenance_configs_load_without_old_guided_chain() -> None:
    repo_root = _repo_root()
    long_stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_long_probe.yaml"
    )
    rehearsal_stack = load_stack_config(
        repo_root / "configs/thesis/_shared/hardneg_core/main_league_champion_hardneg_rehearsal_probe.yaml"
    )

    assert long_stack.config.training is not None
    assert rehearsal_stack.config.training is not None
    assert rehearsal_stack.config.training.structured_aux.trajectory_bc_dataset_path is not None


def test_seed_files_parse_as_uint64_values() -> None:
    repo_root = _repo_root()

    for path in [
        "configs/seeds/dev_eval_seeds.txt",
        "configs/seeds/local_dev_eval_seeds.txt",
        "configs/seeds/local_promotion_eval_seeds.txt",
        "configs/seeds/promotion_eval_seeds.txt",
        "configs/seeds/report_eval_seeds.txt",
    ]:
        seeds = parse_seed_file(repo_root / path)
        assert seeds
        assert all(0 <= seed < 2**64 for seed in seeds)
