from __future__ import annotations

from pathlib import Path

from weiss_rl.sweeps import build_sweep_launch_plan, get_sweep_preset, list_sweep_presets


def test_list_sweep_presets_is_stable() -> None:
    assert list_sweep_presets() == ("noleague_impala_compact", "norecurrence_compact", "ppo_compact")


def test_get_sweep_preset_returns_candidates() -> None:
    preset = get_sweep_preset("noleague_impala_compact")

    assert preset.stack_config == "configs/baselines/noleague_impala.yaml"
    assert len(preset.candidates) == 4


def test_norecurrence_sweep_uses_noleague_anchor_surface() -> None:
    preset = get_sweep_preset("norecurrence_compact")

    assert preset.stack_config == "configs/baselines/norecurrence_noleague.yaml"


def test_build_sweep_launch_plan_builds_candidate_seed_grid(tmp_path: Path) -> None:
    plan, payload = build_sweep_launch_plan(
        preset_id="ppo_compact",
        repo_root=tmp_path,
        group_label="ppo_sweep",
        seeds=[1, 2],
        devices=("cuda:0", "cuda:1"),
        train_args=["--max-updates", "10"],
    )

    assert plan.max_parallel_jobs == 2
    assert len(plan.jobs) == 8
    assert plan.jobs[0].device == "cuda:0"
    assert plan.jobs[1].device == "cuda:1"
    assert "--config-override" in plan.jobs[0].extra_args
    assert payload["preset_id"] == "ppo_compact"
    assert len(payload["jobs"]) == 8
