from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.core.spec import spec_bundle_hash
from weiss_rl.experiments.toy_public_demo import public_demo_spec_hash256
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.model import PolicyValueModel

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mismatched_sha256(value: str) -> str:
    return ("0" if value[0] != "0" else "1") + value[1:]


def _typed_observation_spec() -> dict[str, Any]:
    return {
        "obs_encoding_version": 2,
        "dtype": "i32",
        "obs_len": 512,
        "self_first": True,
        "header_fields": [
            {"name": "active_player", "index": 0},
            {"name": "phase", "index": 1},
            {"name": "decision_kind", "index": 2},
        ],
        "player_blocks": [
            {
                "player_index": 0,
                "base": 16,
                "len": 42,
                "slices": [
                    {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                    {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                    {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                    {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                    {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                ],
            },
            {
                "player_index": 1,
                "base": 58,
                "len": 42,
                "slices": [
                    {"name": "level_count", "start": 0, "len": 1, "visibility": "public"},
                    {"name": "clock_count", "start": 1, "len": 1, "visibility": "public"},
                    {"name": "hand_count", "start": 2, "len": 1, "visibility": "private"},
                    {"name": "stage", "start": 3, "len": 35, "visibility": "public"},
                    {"name": "hand", "start": 38, "len": 4, "visibility": "private"},
                ],
            },
        ],
    }


def _write_stub_weiss_sim(
    tmp_path: Path,
    *,
    spec_hash: int = 123,
    pass_action_id: int = 8,
) -> dict[str, object]:
    bundle: dict[str, object] = {
        "policy_version": 3,
        "spec_hash": spec_hash,
        "observation": _typed_observation_spec(),
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 9,
            "pass_action_id": pass_action_id,
        },
    }
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            (
                "__version__ = '1.2.0'",
                "",
                "def build_info():",
                "    return 'stub-build'",
                "",
                "def db_info():",
                "    return 'stub-db'",
                "",
                "def export_spec_bundle():",
                f"    return {bundle!r}",
                "",
                "class _Cards:",
                "    def presets(self):",
                "        return ['main_deck_5hy_yotsuba_v1', 'aggro_deck_5hy_nino_v1', 'control_deck_jj_s66_v1']",
                "    def preset_min_rules_profile(self, name):",
                "        return 'approx'",
                "",
                "cards = _Cards()",
                "",
            )
        ),
        encoding="utf-8",
    )
    return bundle


def _write_runtime_weiss_sim(
    tmp_path: Path,
    *,
    spec_hash: int = 123,
    pass_action_id: int = 8,
    empty_eval_legal_row: bool = False,
) -> dict[str, object]:
    bundle = _write_stub_weiss_sim(
        tmp_path,
        spec_hash=spec_hash,
        pass_action_id=pass_action_id,
    )
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            (
                "from types import SimpleNamespace",
                "",
                "__version__ = '1.2.0'",
                f"_BUNDLE = {bundle!r}",
                f"PASS_ACTION_ID = {pass_action_id}",
                "OBS_LEN = 512",
                "ACTION_SPACE_SIZE = 9",
                f"SPEC_HASH = {spec_hash}",
                f"EMPTY_EVAL_LEGAL_ROW = {empty_eval_legal_row!r}",
                "",
                "def build_info():",
                "    return 'stub-build'",
                "",
                "def db_info():",
                "    return 'stub-db'",
                "",
                "def export_spec_bundle():",
                "    return _BUNDLE",
                "",
                "class _BaseOut:",
                "    def __init__(self, num_envs: int) -> None:",
                "        import numpy as np",
                "        self.obs = np.zeros((num_envs, 512), dtype=np.float32)",
                "        self.rewards = np.zeros((num_envs,), dtype=np.float32)",
                "        self.terminated = np.zeros((num_envs,), dtype=bool)",
                "        self.truncated = np.zeros((num_envs,), dtype=bool)",
                "        self.actor = np.zeros((num_envs,), dtype=np.int32)",
                "        self.decision_kind = np.zeros((num_envs,), dtype=np.int32)",
                "        self.decision_id = np.zeros((num_envs,), dtype=np.int64)",
                "        self.engine_status = np.zeros((num_envs,), dtype=np.uint8)",
                "        self.spec_hash = np.zeros((num_envs,), dtype=np.uint64)",
                "",
                "class BatchOutMinimal(_BaseOut):",
                "    def __init__(self, num_envs: int) -> None:",
                "        import numpy as np",
                "        super().__init__(num_envs)",
                "        self.masks = np.zeros((num_envs, 9), dtype=np.uint8)",
                "",
                "class BatchOutMinimalNoMask(_BaseOut):",
                "    pass",
                "",
                "class BatchOutMinimalI16LegalIds(_BaseOut):",
                "    def __init__(self, num_envs: int) -> None:",
                "        import numpy as np",
                "        super().__init__(num_envs)",
                "        self.obs = np.zeros((num_envs, 512), dtype=np.int16)",
                "        self.legal_ids = np.zeros((max(1, num_envs),), dtype=np.uint32)",
                "        self.legal_offsets = np.zeros((num_envs + 1,), dtype=np.int32)",
                "",
                "class _Pool:",
                "    def __init__(self, num_envs: int, seed: int) -> None:",
                "        import numpy as np",
                "        self.envs_len = int(num_envs)",
                "        self.action_space = 9",
                "        self.seed = int(seed)",
                "        self._episode_seed = np.full((num_envs,), self.seed, dtype=np.uint64)",
                "        self._episode_index = np.zeros((num_envs,), dtype=np.uint32)",
                "        self._env_index = np.arange(num_envs, dtype=np.uint32)",
                "",
                "    def close(self) -> None:",
                "        return None",
                "",
                "    def episode_seed_batch(self):",
                "        return self._episode_seed",
                "",
                "    def episode_index_batch(self):",
                "        return self._episode_index",
                "",
                "    def env_index_batch(self):",
                "        return self._env_index",
                "",
                "    def reset_indices_with_episode_seeds_into(self, indices, episode_seeds, out):",
                "        self._reset_with_episode_seeds(indices, episode_seeds, out, layout='mask')",
                "",
                "    def reset_indices_with_episode_seeds_into_i16_legal_ids(self, indices, episode_seeds, out):",
                "        self._reset_with_episode_seeds(indices, episode_seeds, out, layout='i16_legal_ids')",
                "",
                "    def _reset_with_episode_seeds(self, indices, episode_seeds, out, *, layout: str):",
                "        for env_index, episode_seed in zip(indices, episode_seeds):",
                "            self._episode_seed[int(env_index)] = int(episode_seed)",
                "        _fill_reset(out, layout=layout, seed=int(episode_seeds[0]))",
                "",
                "def _make_pool(**kwargs):",
                "    return SimpleNamespace(pool=_Pool(kwargs['num_envs'], kwargs.get('seed', 7)))",
                "",
                "def make_pool(**kwargs):",
                "    return _make_pool(**kwargs)",
                "",
                "def fast(**kwargs):",
                "    return _make_pool(**kwargs)",
                "",
                "def inspect(**kwargs):",
                "    return _make_pool(**kwargs)",
                "",
                "class EnvPoolBuffers:",
                "    pass",
                "",
                "class _Cards:",
                "    def presets(self):",
                "        return ['main_deck_5hy_yotsuba_v1', 'aggro_deck_5hy_nino_v1', 'control_deck_jj_s66_v1']",
                "    def preset_min_rules_profile(self, name):",
                "        return 'approx'",
                "",
                "cards = _Cards()",
                "",
                "def _fill_reset(out, *, layout: str, seed: int) -> None:",
                "    import numpy as np",
                "    out.rewards[:] = 0.0",
                "    out.terminated[:] = False",
                "    out.truncated[:] = False",
                "    out.actor[:] = 0",
                "    out.decision_kind[:] = 0",
                "    out.decision_id[:] = 1",
                "    out.engine_status[:] = 0",
                "    out.spec_hash[:] = np.uint64(seed)",
                "    if layout == 'mask':",
                "        out.obs[:] = 0.0",
                "        out.masks[:] = 0",
                "        out.masks[:, 0] = 1",
                "        return",
                "    out.obs[:] = 0",
                "    out.legal_ids[:] = 0",
                "    out.legal_offsets[:] = 0",
                "    if EMPTY_EVAL_LEGAL_ROW:",
                "        return",
                "    out.legal_ids[0] = 0",
                "    out.legal_offsets[1:] = 1",
                "",
                "def _fill_step(out, *, layout: str, seed: int) -> None:",
                "    import numpy as np",
                "    out.rewards[:] = 1.0",
                "    out.terminated[:] = True",
                "    out.truncated[:] = False",
                "    out.actor[:] = -1",
                "    out.decision_kind[:] = 0",
                "    out.decision_id[:] = 2",
                "    out.engine_status[:] = 0",
                "    out.spec_hash[:] = np.uint64(seed)",
                "    if layout == 'mask':",
                "        out.obs[:] = 1.0",
                "        out.masks[:] = 0",
                "        return",
                "    out.obs[:] = 1",
                "    out.legal_ids[:] = 0",
                "    out.legal_offsets[:] = 0",
                "",
                "class _Rl:",
                "    def reset_rl(self, pool, *, layout: str, out=None):",
                "        target = BatchOutMinimal(pool.envs_len) if out is None and layout == 'mask' else out",
                "        if target is None:",
                "            target = BatchOutMinimalI16LegalIds(pool.envs_len)",
                "        _fill_reset(target, layout=layout, seed=pool.seed)",
                "        return target",
                "",
                "    def step_rl(self, pool, actions, *, layout: str, out=None):",
                "        target = BatchOutMinimal(pool.envs_len) if out is None and layout == 'mask' else out",
                "        if target is None:",
                "            target = BatchOutMinimalI16LegalIds(pool.envs_len)",
                "        _fill_step(target, layout=layout, seed=pool.seed)",
                "        return target",
                "",
                "    def step_rl_sample_from_logits(self, pool, logits, *, layout: str, out=None, **kwargs):",
                "        return self.step_rl(pool, logits, layout=layout, out=out)",
                "",
                "    def step_rl_sample_from_logits_with_logp(self, pool, logits, *, layout: str, out=None, **kwargs):",
                "        result = self.step_rl(pool, logits, layout=layout, out=out)",
                "        return SimpleNamespace(batch=result, actions=[0] * pool.envs_len, logp=[0.0] * pool.envs_len)",
                "",
                "rl = _Rl()",
            )
        ),
        encoding="utf-8",
    )
    return bundle


def _patch_periodic_dev_eval_config(tmp_path: Path) -> None:
    preset_path = tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"
    preset_text = preset_path.read_text(encoding="utf-8")
    preset_text = preset_text.replace(
        "periodic_dev_eval_interval_updates: 50000",
        "periodic_dev_eval_interval_updates: 1",
    )
    preset_text = preset_text.replace(
        "periodic_dev_eval_paired_seeds: 64",
        "periodic_dev_eval_paired_seeds: 1",
    )
    preset_path.write_text(preset_text, encoding="utf-8")
    (tmp_path / "configs" / "seeds" / "dev_eval_seeds.txt").write_text("7\n", encoding="utf-8")


def _copy_repo_configs(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    return tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"


def _write_manifest_only_stack_config(tmp_path: Path) -> Path:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    stack_config = configs_dir / "manifest_only.yaml"
    stack_config.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "description: Minimal manifest-only scaffold preset.",
                "experiment:",
                "  role: main",
            )
        ),
        encoding="utf-8",
    )
    return stack_config


def _write_eval_only_stack_config(tmp_path: Path) -> Path:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    stack_config = configs_dir / "stack_eval_only.yaml"
    stack_config.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "description: Minimal scaffold preset with evaluation-only policy-set selection.",
                "experiment:",
                "  role: main",
                "evaluation:",
                "  eval_device: cpu",
                "  eval_sampling_algorithm: pinned_cdf_pcg_v1",
                "  eval_inference_mode: true",
                "  seat_swap: true",
                "  eval_assert_sorted_legal_ids: true",
                "  seed_files:",
                "    dev_eval: configs/seeds/dev_eval_seeds.txt",
                "    report_eval: configs/seeds/report_eval_seeds.txt",
                "    promotion_gate: configs/seeds/promotion_eval_seeds.txt",
                "  periodic_dev_eval_interval_updates: 0",
                "  periodic_dev_eval_paired_seeds: 16",
                "  final_policy_set_size: 10",
                "  final_matrix_stage1_paired_seeds: 2",
                "  final_matrix_stage2_adaptive_max_paired_seeds: 2",
                "  legal_fingerprint_checks:",
                "    enabled: true",
                "    version: legal_fingerprint_v1",
                "    mismatch_policy: hard_fail",
                "    require_strictly_increasing_legal_ids: true",
                "  stop_rules:",
                "    stop_delta_ci_half_width: 0.03",
                "    stop_confidence: 0.95",
                "  replay_capture_rate_eval: 0.0",
                "  regression_capture_count: 0",
                "  decision_kind_tagging:",
                "    required_for_training: false",
                "    enable_python_derived_debug_tag: false",
                "  final_policy_set_selection:",
                "    version: deterministic_v1",
                "    include_random_legal_baseline_b0: true",
                "    include_no_league_baseline_b1: true",
                "    include_heuristic_public_b2_if_exists: true",
                "    include_final_champion_snapshot: true",
                "    include_spaced_snapshots_near_percent_updates: [25, 50, 75]",
                "    remaining_slots_strategy: top_dev_performers_vs_anchor_set_v1",
                "    fixed_anchor_set_v1:",
                "      required: [B0 RandomLegal, B1 NoLeague baseline]",
                "      optional_if_available: [B2 HeuristicPublic]",
                "    seed_file: configs/seeds/dev_eval_seeds.txt",
                "    folding: S0",
                "    seat_swap: true",
                "    tie_break: lowest_policy_id",
                "",
            )
        ),
        encoding="utf-8",
    )
    return stack_config


def _write_b1_baseline_run_fixture(tmp_path: Path, *, stack_config: Path, update: int = 5) -> Path:
    stack = load_stack_config(stack_config)
    assert stack.config.model is not None
    config_hash256 = compute_config_hash256(stack)

    run_dir = tmp_path / "runs" / "b1_no_league_source"
    weights_path = run_dir / "training" / "snapshots" / "b1_noleague_baseline" / "weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model = PolicyValueModel(
        observation_dim=512,
        config=stack.config.model,
        action_dim=9,
        observation_spec=_typed_observation_spec(),
    )
    payload = {
        "policy_id": "b1_noleague_baseline",
        "update": update,
        "config_hash256": config_hash256,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, weights_path)
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    (run_dir / "config_hash256.txt").write_text(f"{config_hash256}\n", encoding="utf-8")

    registry = SnapshotRegistry()
    registry.add_snapshot(
        policy_id="b1_noleague_baseline",
        update=update,
        weights_sha256=weights_sha256,
        path="training/snapshots/b1_noleague_baseline/weights.pt",
    )
    registry.pin_snapshot("b1_noleague_baseline")
    registry.save(run_dir / "training" / "snapshots" / "registry.json")
    return run_dir


def _write_policy_set_inputs(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_registry_path = tmp_path / "policy_set_snapshot_registry.json"
    dev_eval_summaries_path = tmp_path / "policy_set_dev_eval_summaries.json"
    snapshot_registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": "policy_000100",
                        "update": 100,
                        "weights_sha256": "1" * 64,
                        "path": "training/snapshots/policy_000100/weights.pt",
                        "created_utc": "2026-01-01T00:00:00+00:00",
                    },
                    {
                        "policy_id": "policy_000200",
                        "update": 200,
                        "weights_sha256": "2" * 64,
                        "path": "training/snapshots/policy_000200/weights.pt",
                        "created_utc": "2026-01-01T00:00:01+00:00",
                    },
                    {
                        "policy_id": "policy_000300",
                        "update": 300,
                        "weights_sha256": "3" * 64,
                        "path": "training/snapshots/policy_000300/weights.pt",
                        "created_utc": "2026-01-01T00:00:02+00:00",
                    },
                    {
                        "policy_id": "policy_000400",
                        "update": 400,
                        "weights_sha256": "4" * 64,
                        "path": "training/snapshots/policy_000400/weights.pt",
                        "created_utc": "2026-01-01T00:00:03+00:00",
                    },
                ],
                "champion_snapshots": ["policy_000400"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    dev_eval_summaries_path.write_text(
        json.dumps(
            {
                "B2 HeuristicPublic": {"aggregate_score": 0.0, "anchor_scores": {}},
                "policy_000150": {
                    "aggregate_score": 0.99,
                    "anchor_scores": {
                        "B0 RandomLegal": 0.70,
                        "B1 NoLeague baseline": 0.70,
                        "B2 HeuristicPublic": 0.70,
                    },
                },
                "policy_000250": {
                    "aggregate_score": 0.95,
                    "anchor_scores": {
                        "B0 RandomLegal": 0.69,
                        "B1 NoLeague baseline": 0.69,
                        "B2 HeuristicPublic": 0.69,
                    },
                },
                "policy_000350": {
                    "aggregate_score": 0.90,
                    "anchor_scores": {
                        "B0 RandomLegal": 0.68,
                        "B1 NoLeague baseline": 0.68,
                        "B2 HeuristicPublic": 0.68,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return snapshot_registry_path, dev_eval_summaries_path


def _run_entrypoint(
    tmp_path: Path,
    *,
    script_name: str,
    stack_config: Path,
    spec_hash: str,
    run_label: str = "",
    run_id_alias: str = "",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT / "python")])
    env["WEISS_SIM_PYTHONPATH"] = str(tmp_path)
    env["WEISS_SIM_PYTHON"] = sys.executable

    command = [sys.executable, str(REPO_ROOT / "python" / "scripts" / script_name), "--stack-config", str(stack_config)]
    if spec_hash:
        command.extend(["--spec-hash", spec_hash])
    if run_label:
        command.extend(["--run-label", run_label])
    if run_id_alias:
        command.extend(["--run-id", run_id_alias])
    if extra_args:
        command.extend(extra_args)

    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)


def test_train_entrypoint_fails_fast_on_runtime_spec_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="999",
        run_label="mismatch_run",
    )

    assert result.returncode != 0
    assert "Spec mismatch" in result.stderr


def test_train_entrypoint_rejects_invalid_runtime_spec_bundle_before_claiming_verification(tmp_path: Path) -> None:
    invalid_bundle = {
        "policy_version": 3,
        "spec_hash": 123,
        "observation": {"obs_encoding_version": 2, "dtype": "i32", "obs_len": 512},
        "action": {"action_encoding_version": 1, "pass_action_id": 8},
    }
    (tmp_path / "weiss_sim.py").write_text(
        "\n".join(
            (
                "def build_info():",
                "    return 'stub-build'",
                "",
                "def db_info():",
                "    return 'stub-db'",
                "",
                "def export_spec_bundle():",
                f"    return {invalid_bundle!r}",
                "",
            )
        ),
        encoding="utf-8",
    )
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="123",
        run_label="invalid_spec_bundle",
    )

    assert result.returncode != 0
    assert "invalid spec_bundle payload" in result.stderr
    assert "Verified runtime spec bundle" not in result.stdout


def test_train_entrypoint_persists_runtime_spec_bundle(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="spec_bundle_run",
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "spec_bundle_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["simulator"]["compatibility_hash"] == "123"
    assert manifest["spec_bundle"] == bundle
    assert manifest["policy_set_selection"] == []
    assert manifest["policy_set_selection_details"] == {
        "mode": "not_configured",
        "status": "not_configured",
        "source_paths": {
            "snapshot_registry_json": None,
            "dev_eval_summaries_json": None,
        },
    }
    assert (manifest_path.parent / "spec_bundle.json").is_file()
    assert (manifest_path.parent / "spec_hash256.txt").read_text(encoding="utf-8").strip() == spec_bundle_hash(bundle)
    assert "computed_run_id64:" in result.stdout
    assert "computed_run_id256:" in result.stdout
    assert "run_label:              spec_bundle_run" in result.stdout
    assert "run_dir_name:           spec_bundle_run" in result.stdout
    assert "Manifest scaffold only: no learner training or rollout collection was executed." in result.stdout
    assert "missing config blocks: environment, training, model" in result.stdout


def test_train_entrypoint_locked_stack_fails_on_incomplete_runtime(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="locked_stack_requires_runtime",
    )

    assert result.returncode != 0
    assert "Canonical simulator-backed training requires a weiss_sim runtime with stepping support" in result.stderr
    assert "active weiss_sim runtime is missing stepping APIs" in result.stderr


def test_train_entrypoint_resolves_policy_set_selection_when_inputs_are_supplied(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="resolved_policy_set_run",
        extra_args=[
            "--snapshot-registry-json",
            str(snapshot_registry_path),
            "--dev-eval-summaries-json",
            str(dev_eval_summaries_path),
        ],
    )

    assert result.returncode == 0, result.stderr
    manifest_path = tmp_path / "runs" / "resolved_policy_set_run" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["policy_set_selection"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    assert manifest["policy_set_selection_details"] == {
        "mode": "deterministic_v1",
        "status": "resolved",
        "version": "deterministic_v1",
        "final_policy_set_size": 10,
        "source_paths": {
            "snapshot_registry_json": "policy_set_snapshot_registry.json",
            "dev_eval_summaries_json": "policy_set_dev_eval_summaries.json",
        },
        "missing_inputs": [],
        "selected_policy_count": 10,
    }


def test_eval_entrypoint_prefers_run_local_policy_selection_over_manifest_fallback(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    import weiss_rl.workflows.eval_entrypoint as eval_entrypoint
    from weiss_rl.workflows import (
        eval_canonical,
        eval_canonical_cli_messages,
        eval_canonical_dependencies,
        eval_canonical_entrypoint_adapter,
        eval_canonical_entrypoint_request,
        eval_canonical_figure_outputs,
        eval_canonical_final_eval,
        eval_canonical_metagame_outputs,
        eval_canonical_output_bundle,
        eval_canonical_outputs,
        eval_canonical_phases,
        eval_canonical_policy_runtime,
        eval_canonical_publisher,
        eval_canonical_readiness_outputs,
        eval_canonical_report_publication,
        eval_canonical_runtime,
        eval_canonical_seed_budget,
        eval_canonical_setup,
        eval_canonical_state,
        eval_canonical_supplemental_outputs,
        eval_canonical_tensorboard_publication,
        eval_dispatch,
        eval_dispatch_dependencies,
        eval_dispatch_request,
        eval_dispatch_route_adapters,
        eval_dispatch_routes,
        eval_entrypoint_compat,
        eval_entrypoint_export_groups,
        eval_entrypoint_exports,
        eval_entrypoint_external_exports,
        eval_entrypoint_main,
        eval_entrypoint_report_exports,
        eval_entrypoint_runtime,
        eval_entrypoint_workflow_exports,
        eval_modes,
        eval_parser,
        eval_public_demo_mode,
        eval_reports,
        eval_startup,
        eval_startup_dependencies,
        eval_startup_prepare,
        eval_startup_state,
        eval_startup_validation,
        eval_summary_mode,
    )

    assert eval_script is eval_entrypoint
    assert (
        eval_entrypoint.run_entrypoint_canonical_eval_pipeline
        is eval_entrypoint_compat.run_entrypoint_canonical_eval_pipeline
    )
    assert eval_entrypoint.EVAL_ENTRYPOINT_EXPORTS is eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS
    assert eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS is eval_entrypoint_export_groups.EVAL_ENTRYPOINT_EXPORTS
    assert eval_entrypoint_exports.EVAL_REPORT_HELPER_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_REPORT_HELPER_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_WORKFLOW_COMPAT_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_WORKFLOW_COMPAT_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_ADDITIONAL_COMPAT_EXPORTS is (
        eval_entrypoint_export_groups.EVAL_ADDITIONAL_COMPAT_EXPORTS
    )
    assert eval_entrypoint_exports.EVAL_ENTRYPOINT_EXPORTS == [
        *eval_entrypoint_export_groups.EVAL_REPORT_HELPER_EXPORTS,
        *eval_entrypoint_export_groups.EVAL_WORKFLOW_COMPAT_EXPORTS,
    ]
    assert set(eval_entrypoint_exports.__all__) == set(
        [
            *eval_entrypoint_export_groups.EVAL_ENTRYPOINT_EXPORTS,
            *eval_entrypoint_export_groups.EVAL_ADDITIONAL_COMPAT_EXPORTS,
            "EVAL_ADDITIONAL_COMPAT_EXPORTS",
            "EVAL_REPORT_HELPER_EXPORTS",
            "EVAL_WORKFLOW_COMPAT_EXPORTS",
        ]
    )
    assert eval_entrypoint.run_eval_entrypoint_main is eval_entrypoint_main.run_eval_entrypoint_main
    assert eval_entrypoint.run_eval_entrypoint is eval_entrypoint_runtime.run_eval_entrypoint
    assert (
        eval_entrypoint.run_eval_entrypoint_canonical_pipeline
        is eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline
    )
    assert eval_script.build_eval_parser is eval_parser.build_eval_parser
    assert eval_entrypoint_exports.build_eval_parser is eval_entrypoint_workflow_exports.build_eval_parser
    assert eval_entrypoint_exports.resolve_eval_policies is eval_entrypoint_external_exports.resolve_eval_policies
    assert eval_entrypoint_exports._resolve_policy_ids_for_run is (
        eval_entrypoint_report_exports._resolve_policy_ids_for_run
    )
    assert eval_entrypoint_exports.TensorBoardLogger is eval_entrypoint_external_exports.TensorBoardLogger
    assert eval_entrypoint_exports.CanonicalEvalDependencies is (
        eval_entrypoint_workflow_exports.CanonicalEvalDependencies
    )
    assert eval_script.run_canonical_eval_pipeline is eval_canonical.run_canonical_eval_pipeline
    assert (
        eval_script.run_canonical_eval_entrypoint_pipeline
        is eval_canonical_entrypoint_adapter.run_canonical_eval_entrypoint_pipeline
    )
    assert "CanonicalEvalEntrypointRequest" in eval_canonical_entrypoint_request.__all__
    assert "canonical_eval_entrypoint_request" in eval_canonical_entrypoint_request.__all__
    assert eval_script.CanonicalEvalDependencies is eval_canonical.CanonicalEvalDependencies
    assert eval_script.CanonicalEvalDependencies is eval_canonical_dependencies.CanonicalEvalDependencies
    assert eval_canonical.CanonicalEvalDependencies is eval_canonical_dependencies.CanonicalEvalDependencies
    assert eval_canonical.CanonicalEvalRunState is eval_canonical_state.CanonicalEvalRunState
    assert eval_canonical.CanonicalEvalRuntimeState is eval_canonical_state.CanonicalEvalRuntimeState
    assert eval_canonical_phases.CanonicalEvalRunState is eval_canonical_state.CanonicalEvalRunState
    assert eval_canonical_phases.CanonicalEvalRuntimeState is eval_canonical_state.CanonicalEvalRuntimeState
    assert eval_canonical.prepare_canonical_eval_run_state is eval_canonical_setup.prepare_canonical_eval_run_state
    assert eval_canonical_phases.prepare_canonical_eval_run_state is (
        eval_canonical_setup.prepare_canonical_eval_run_state
    )
    assert eval_canonical.resolve_canonical_eval_runtime_state is (
        eval_canonical_runtime.resolve_canonical_eval_runtime_state
    )
    assert eval_canonical_phases.resolve_canonical_eval_runtime_state is (
        eval_canonical_runtime.resolve_canonical_eval_runtime_state
    )
    assert eval_canonical.write_canonical_eval_outputs is eval_canonical_outputs.write_canonical_eval_outputs
    assert eval_canonical_phases.write_canonical_eval_outputs is (eval_canonical_outputs.write_canonical_eval_outputs)
    assert "render_canonical_eval_output_messages" in eval_canonical_cli_messages.__all__
    assert "build_canonical_figure_outputs" in eval_canonical_figure_outputs.__all__
    assert "run_canonical_final_eval_output" in eval_canonical_final_eval.__all__
    assert "build_canonical_metagame_output" in eval_canonical_metagame_outputs.__all__
    assert "build_canonical_readiness_output" in eval_canonical_readiness_outputs.__all__
    assert "build_canonical_supplemental_outputs" in eval_canonical_supplemental_outputs.__all__
    assert "build_canonical_eval_output_bundle" in eval_canonical_output_bundle.__all__
    assert "publish_canonical_eval_outputs" in eval_canonical_publisher.__all__
    assert "publish_canonical_eval_run_reports" in eval_canonical_report_publication.__all__
    assert "publish_canonical_eval_tensorboard_summaries" in eval_canonical_tensorboard_publication.__all__
    assert "resolve_canonical_eval_policy_runtime" in eval_canonical_policy_runtime.__all__
    assert "resolve_recommended_focal_policy_id" in eval_canonical_policy_runtime.__all__
    assert "resolve_canonical_eval_seed_budget" in eval_canonical_seed_budget.__all__
    assert eval_script.run_eval_dispatch is eval_dispatch_routes.run_eval_dispatch
    assert eval_dispatch.run_eval_dispatch is eval_dispatch_routes.run_eval_dispatch
    assert eval_script.EvalDispatchDependencies is eval_dispatch_dependencies.EvalDispatchDependencies
    assert eval_dispatch.EvalDispatchDependencies is eval_dispatch_dependencies.EvalDispatchDependencies
    assert eval_dispatch.EvalDispatchRequest is eval_dispatch_request.EvalDispatchRequest
    assert "EvalDispatchRequest" in eval_dispatch_request.__all__
    assert "run_canonical_eval_request" in eval_dispatch_route_adapters.__all__
    assert eval_script.build_eval_dispatch_dependencies is eval_dispatch_dependencies.build_eval_dispatch_dependencies
    assert eval_dispatch.build_eval_dispatch_dependencies is eval_dispatch_dependencies.build_eval_dispatch_dependencies
    assert eval_dispatch._print_startup_verification is eval_dispatch_route_adapters._print_startup_verification
    assert eval_dispatch.run_public_demo_eval_route is eval_dispatch_route_adapters.run_public_demo_eval_route
    assert eval_dispatch.run_canonical_eval_route is eval_dispatch_route_adapters.run_canonical_eval_route
    assert eval_dispatch.run_summary_only_eval_route is eval_dispatch_route_adapters.run_summary_only_eval_route
    assert eval_script.EvalStartup is eval_startup_state.EvalStartup
    assert eval_script.EvalStartupDependencies is eval_startup_dependencies.EvalStartupDependencies
    assert eval_startup.EvalStartupDependencies is eval_startup_dependencies.EvalStartupDependencies
    assert eval_script.build_eval_startup_dependencies is eval_startup_dependencies.build_eval_startup_dependencies
    assert eval_startup.build_eval_startup_dependencies is eval_startup_dependencies.build_eval_startup_dependencies
    assert eval_script.EvalValidatedArgs is eval_startup_state.EvalValidatedArgs
    assert eval_startup.EvalStartup is eval_startup_state.EvalStartup
    assert eval_startup.EvalValidatedArgs is eval_startup_state.EvalValidatedArgs
    assert eval_script.prepare_eval_startup is eval_startup_prepare.prepare_eval_startup
    assert eval_startup.prepare_eval_startup is eval_startup_prepare.prepare_eval_startup
    assert eval_script.validate_eval_args is eval_startup_validation.validate_eval_args
    assert eval_startup.validate_eval_args is eval_startup_validation.validate_eval_args
    assert eval_script.run_public_demo_eval_mode is eval_public_demo_mode.run_public_demo_eval_mode
    assert eval_modes.run_public_demo_eval_mode is eval_public_demo_mode.run_public_demo_eval_mode
    assert eval_script.run_summary_only_eval_mode is eval_summary_mode.run_summary_only_eval_mode
    assert eval_modes.run_summary_only_eval_mode is eval_summary_mode.run_summary_only_eval_mode
    assert eval_script._resolve_policy_ids_for_run is eval_reports._resolve_policy_ids_for_run
    assert eval_script._load_run_summary_or_default is eval_reports._load_run_summary_or_default

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_stale_only"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_report_facade_reexports_split_module_owners() -> None:
    from weiss_rl.workflows import (
        eval_policy_final_set_resolution,
        eval_policy_manifest_selection,
        eval_policy_selection,
        eval_policy_selection_results,
        eval_report_io,
        eval_report_scaffolding,
        eval_report_update_payloads,
        eval_report_updates,
        eval_reports,
    )

    assert (
        eval_reports._authoritative_manifest_policy_selection
        is eval_policy_manifest_selection._authoritative_manifest_policy_selection
    )
    assert eval_reports._effective_manifest_git_commit is eval_policy_manifest_selection._effective_manifest_git_commit
    assert (
        eval_reports._persist_policy_selection_in_manifest
        is eval_policy_manifest_selection._persist_policy_selection_in_manifest
    )
    assert eval_reports._policy_selection_mode is eval_policy_manifest_selection._policy_selection_mode
    assert (
        eval_reports._resolve_selection_inputs_from_manifest
        is eval_policy_manifest_selection._resolve_selection_inputs_from_manifest
    )
    assert (
        eval_reports._run_summary_marks_canonical_eval_completed
        is eval_policy_manifest_selection._run_summary_marks_canonical_eval_completed
    )
    assert eval_reports._default_dev_eval_summaries_path is (
        eval_policy_final_set_resolution._default_dev_eval_summaries_path
    )
    assert eval_reports._explicit_policy_selection is eval_policy_selection_results._explicit_policy_selection
    assert (
        eval_reports._manifest_policy_selection_fallback
        is eval_policy_selection_results._manifest_policy_selection_fallback
    )
    assert eval_reports.RunLevelReportUpdateInputs is eval_report_update_payloads.RunLevelReportUpdateInputs
    assert eval_reports.build_run_summary_update_fields is eval_report_update_payloads.build_run_summary_update_fields
    assert (
        eval_reports.build_determinism_report_update_fields
        is eval_report_update_payloads.build_determinism_report_update_fields
    )
    assert eval_reports._load_json_object is eval_report_io._load_json_object
    assert eval_reports._expected_sha256 is eval_report_io._expected_sha256
    assert eval_reports._load_run_summary_or_default is eval_report_scaffolding._load_run_summary_or_default
    assert eval_reports._ensure_run_level_report_scaffolding is (
        eval_report_scaffolding._ensure_run_level_report_scaffolding
    )
    assert eval_reports._resolve_policy_ids_for_run is eval_policy_selection._resolve_policy_ids_for_run
    assert eval_policy_selection._explicit_policy_selection is eval_policy_selection_results._explicit_policy_selection
    assert (
        eval_policy_selection._manifest_policy_selection_fallback
        is eval_policy_selection_results._manifest_policy_selection_fallback
    )
    assert eval_reports._persist_policy_selection_in_manifest is (
        eval_policy_selection._persist_policy_selection_in_manifest
    )
    assert eval_reports._update_run_level_reports is eval_report_updates._update_run_level_reports


def test_eval_policy_selection_results_build_explicit_cli_details() -> None:
    from weiss_rl.workflows.eval_support.eval_policy_selection_results import _explicit_policy_selection

    assert _explicit_policy_selection([" B0 RandomLegal ", "", "policy_000100"]) == (
        ["B0 RandomLegal", "policy_000100"],
        {"mode": "explicit_cli", "policy_count": 2},
    )
    assert _explicit_policy_selection(["", "   "]) is None


def test_eval_policy_selection_results_build_manifest_fallback_details() -> None:
    from weiss_rl.workflows.eval_support.eval_policy_selection_results import _manifest_policy_selection_fallback

    assert _manifest_policy_selection_fallback({"policy_set_selection": [" B0 RandomLegal ", "", 123]}) == (
        ["B0 RandomLegal", "123"],
        {"mode": "manifest_policy_set_selection_fallback", "policy_count": 2},
    )
    assert _manifest_policy_selection_fallback({"policy_set_selection": "not-a-list"}) is None
    assert _manifest_policy_selection_fallback({}) is None


def test_eval_policy_final_set_resolution_uses_available_source_paths(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _resolve_available_policy_source_paths

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    default_registry = layout.training_snapshots_dir / "registry.json"
    default_registry.write_text("{}\n", encoding="utf-8")
    periodic_dev_eval = layout.training_logs_dir / "periodic_dev_eval_summaries.json"
    periodic_dev_eval.write_text("{}\n", encoding="utf-8")
    manifest_registry = tmp_path / "manifest" / "registry.json"
    manifest_registry.parent.mkdir(parents=True, exist_ok=True)
    manifest_registry.write_text("{}\n", encoding="utf-8")
    explicit_dev_eval = tmp_path / "explicit" / "dev_eval.json"
    explicit_dev_eval.parent.mkdir(parents=True, exist_ok=True)
    explicit_dev_eval.write_text("{}\n", encoding="utf-8")

    resolved_registry, resolved_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=explicit_dev_eval,
        manifest_snapshot_registry=manifest_registry,
        manifest_dev_eval=None,
    )

    assert resolved_registry == manifest_registry
    assert resolved_dev_eval == explicit_dev_eval

    fallback_registry, fallback_dev_eval = _resolve_available_policy_source_paths(
        layout=layout,
        snapshot_registry_path=tmp_path / "missing" / "registry.json",
        dev_eval_summaries_path=tmp_path / "missing" / "dev_eval.json",
        manifest_snapshot_registry=None,
        manifest_dev_eval=None,
    )

    assert fallback_registry is None
    assert fallback_dev_eval == periodic_dev_eval


def test_eval_policy_final_set_resolution_builds_deterministic_selection_details(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _resolve_deterministic_final_policy_set

    observed: dict[str, object] = {}
    registry_path = tmp_path / "registry.json"
    dev_eval_path = tmp_path / "dev_eval.json"

    def fake_resolve_final_policy_set(**kwargs: object) -> list[str]:
        observed["resolve"] = kwargs
        return ["B0 RandomLegal", "policy_000100"]

    resolved = _resolve_deterministic_final_policy_set(
        evaluation=SimpleNamespace(final_policy_set_selection={"folding": "seat_swap_mean"}, final_policy_set_size=2),
        resolved_snapshot_registry=registry_path,
        resolved_dev_eval=dev_eval_path,
        resolve_final_policy_set_fn=fake_resolve_final_policy_set,
    )

    assert resolved == (
        ["B0 RandomLegal", "policy_000100"],
        {
            "mode": "deterministic_v1",
            "policy_count": 2,
            "snapshot_registry_path": registry_path.as_posix(),
            "dev_eval_summaries_path": dev_eval_path.as_posix(),
            "final_policy_set_size": 2,
        },
    )
    assert observed["resolve"] == {
        "snapshot_registry_path": registry_path,
        "dev_eval_summaries_path": dev_eval_path,
        "config": {"folding": "seat_swap_mean"},
        "final_policy_set_size": 2,
    }
    assert (
        _resolve_deterministic_final_policy_set(
            evaluation=SimpleNamespace(final_policy_set_selection={}, final_policy_set_size=2),
            resolved_snapshot_registry=None,
            resolved_dev_eval=dev_eval_path,
        )
        is None
    )


def test_eval_policy_final_set_resolution_reports_missing_inputs(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_final_set_resolution import _raise_missing_final_policy_inputs

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")

    with pytest.raises(FileNotFoundError, match="requires a snapshot registry") as registry_exc:
        _raise_missing_final_policy_inputs(
            layout=layout,
            resolved_snapshot_registry=None,
            resolved_dev_eval=tmp_path / "dev_eval.json",
            snapshot_registry_path=tmp_path / "explicit" / "registry.json",
            manifest_snapshot_registry=tmp_path / "manifest" / "registry.json",
            dev_eval_summaries_path=None,
            manifest_dev_eval=None,
        )
    assert str(tmp_path / "explicit" / "registry.json") in str(registry_exc.value)

    with pytest.raises(FileNotFoundError, match="requires dev-eval summaries") as dev_eval_exc:
        _raise_missing_final_policy_inputs(
            layout=layout,
            resolved_snapshot_registry=tmp_path / "registry.json",
            resolved_dev_eval=None,
            snapshot_registry_path=None,
            manifest_snapshot_registry=None,
            dev_eval_summaries_path=tmp_path / "explicit" / "dev_eval.json",
            manifest_dev_eval=tmp_path / "manifest" / "dev_eval.json",
        )
    message = str(dev_eval_exc.value)
    assert (tmp_path / "explicit" / "dev_eval.json").as_posix() in message
    assert (tmp_path / "manifest" / "dev_eval.json").as_posix() in message
    assert "training/logs/dev_eval_summaries.json" in message
    assert "training/logs/periodic_dev_eval_summaries.json" in message


def test_eval_policy_manifest_selection_resolves_source_paths_from_manifest(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import _resolve_selection_inputs_from_manifest

    absolute_dev_eval = tmp_path / "external" / "dev_eval.json"
    snapshot_registry, dev_eval = _resolve_selection_inputs_from_manifest(
        stack_root=tmp_path / "stack",
        manifest={
            "policy_set_selection_details": {
                "source_paths": {
                    "snapshot_registry_json": "runs/main/training/snapshots/registry.json",
                    "dev_eval_summaries_json": absolute_dev_eval.as_posix(),
                }
            }
        },
    )

    assert snapshot_registry == tmp_path / "stack" / "runs" / "main" / "training" / "snapshots" / "registry.json"
    assert dev_eval == absolute_dev_eval


def test_eval_policy_manifest_selection_requires_completed_artifacts(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_policy_manifest_selection import _authoritative_manifest_policy_selection

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_set_selection": [" B0 RandomLegal ", "policy_000100"],
        "policy_set_selection_details": {"mode": "deterministic_v1", "status": "resolved"},
    }

    assert (
        _authoritative_manifest_policy_selection(
            manifest=manifest,
            layout=layout,
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
        )
        is None
    )

    layout.run_summary_path.write_text(
        json.dumps({"canonical_eval_completed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resolved = _authoritative_manifest_policy_selection(
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert resolved == (
        ["B0 RandomLegal", "policy_000100"],
        {"mode": "deterministic_v1", "status": "resolved", "policy_count": 2},
    )
    assert (
        _authoritative_manifest_policy_selection(
            manifest=manifest,
            layout=layout,
            snapshot_registry_path=tmp_path / "registry.json",
            dev_eval_summaries_path=None,
        )
        is None
    )


def test_eval_report_update_payloads_preserve_summary_and_determinism_fields(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_report_update_payloads import (
        RunLevelReportUpdateInputs,
        build_determinism_report_update_fields,
        build_run_summary_update_fields,
    )

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    selection_details = {"mode": "deterministic_v1", "status": "resolved"}
    inputs = RunLevelReportUpdateInputs(
        layout=layout,
        run_dir=layout.run_dir,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details=selection_details,
        final_eval_payload={"matchups": [{"a": 1}, {"a": 2}]},
        metagame_payload={"kind": "metagame"},
        figure_paths=(layout.figures_paper_dir / "seat_bias.pdf", tmp_path / "external.pdf"),
        readiness_payload={"passed": True},
    )

    assert build_run_summary_update_fields(inputs) == {
        "final_eval_dir": "eval/final_eval",
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "policy_set_selection_mode": "deterministic_v1",
        "metagame_dir": "eval/metagame",
        "figure_outputs": ["figures/paper/seat_bias.pdf", (tmp_path / "external.pdf").as_posix()],
        "paper_readiness_summary_path": "paper_readiness_summary.json",
        "paper_grade": True,
        "canonical_eval_completed": True,
    }

    determinism_fields = build_determinism_report_update_fields(
        inputs,
        replay_verification={
            "status": "verified",
            "sampled_episode_count": 5,
            "verified_episode_count": 4,
            "failed_episode_count": 1,
        },
        artifact_hashes={"artifacts": {"summary.json": "ab" * 32}},
    )

    assert determinism_fields == {
        "run_dir": layout.run_dir.as_posix(),
        "policy_selection_mode": "deterministic_v1",
        "replay_verification": {
            "path": "eval/diagnostics/replay_verification.json",
            "status": "verified",
            "sampled_episode_count": 5,
            "verified_episode_count": 4,
            "failed_episode_count": 1,
        },
        "canonical_artifact_hashes": {"summary.json": "ab" * 32},
        "final_eval": {
            "path": "eval/final_eval/summary.json",
            "policy_ids": ["B0 RandomLegal", "policy_000100"],
            "selection": selection_details,
            "matchup_count": 2,
        },
    }


def test_eval_report_update_writes_summary_and_determinism_artifacts(tmp_path: Path) -> None:
    from weiss_rl.workflows.eval_support.eval_report_updates import _update_run_level_reports

    layout = ArtifactLayout.from_run_dir(tmp_path / "run")
    layout.ensure_directories()
    layout.run_summary_path.write_text(
        json.dumps({"kind": "run_summary_v1", "preexisting": "summary"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    layout.determinism_report_path.write_text(
        json.dumps({"kind": "determinism_report_v1", "preexisting": "determinism"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    layout.replay_verification_json().write_text(
        json.dumps(
            {
                "status": "verified",
                "sampled_episode_count": 3,
                "verified_episode_count": 3,
                "failed_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    layout.final_eval_aggregate_hashes_json().write_text(
        json.dumps({"artifacts": {"summary.json": "cd" * 32}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _update_run_level_reports(
        layout=layout,
        run_dir=layout.run_dir,
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"mode": "deterministic_v1", "status": "resolved"},
        final_eval_payload={"matchups": [{"winner": "a"}]},
        metagame_payload=None,
        figure_paths=(),
        readiness_payload={"passed": False},
    )

    run_summary = json.loads(layout.run_summary_path.read_text(encoding="utf-8"))
    assert run_summary["preexisting"] == "summary"
    assert run_summary["final_eval_dir"] == "eval/final_eval"
    assert run_summary["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert run_summary["policy_set_selection_mode"] == "deterministic_v1"
    assert run_summary["metagame_dir"] is None
    assert run_summary["figure_outputs"] == []
    assert run_summary["paper_grade"] is False
    assert run_summary["canonical_eval_completed"] is True

    determinism = json.loads(layout.determinism_report_path.read_text(encoding="utf-8"))
    assert determinism["preexisting"] == "determinism"
    assert determinism["policy_selection_mode"] == "deterministic_v1"
    assert determinism["replay_verification"] == {
        "path": "eval/diagnostics/replay_verification.json",
        "status": "verified",
        "sampled_episode_count": 3,
        "verified_episode_count": 3,
        "failed_episode_count": 0,
    }
    assert determinism["canonical_artifact_hashes"] == {"summary.json": "cd" * 32}
    assert determinism["final_eval"]["matchup_count"] == 1


def test_eval_entrypoint_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    import scripts.eval as eval_script

    class FakeTensorBoardLogger:
        pass

    def fake_run_final_eval(**_kwargs: object) -> dict[str, object]:
        return {}

    monkeypatch.setattr(eval_script, "TensorBoardLogger", FakeTensorBoardLogger)
    monkeypatch.setattr(eval_script, "run_final_eval", fake_run_final_eval)

    dependencies = eval_script._canonical_eval_dependencies()

    assert dependencies.tensorboard_logger_cls is FakeTensorBoardLogger
    assert dependencies.run_final_eval_fn is fake_run_final_eval


def test_eval_dispatch_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    import scripts.eval as eval_script

    def fake_public_demo_eval_mode(**_kwargs: object) -> None:
        return None

    def fake_summary_json(_path: Path, _payload: object) -> None:
        return None

    monkeypatch.setattr(eval_script, "run_public_demo_eval_mode", fake_public_demo_eval_mode)
    monkeypatch.setattr(eval_script, "write_matchup_summary_json", fake_summary_json)

    dependencies = eval_script._eval_dispatch_dependencies()

    assert dependencies.run_public_demo_eval_mode_fn is fake_public_demo_eval_mode
    assert dependencies.write_matchup_summary_json_fn is fake_summary_json


def test_eval_startup_dependency_builder_preserves_monkeypatch_surface(monkeypatch) -> None:
    import scripts.eval as eval_script

    def fake_load_stack_config(_path: Path) -> object:
        return object()

    def fake_banner(
        _reported_spec_hash: str,
        _config_hash256: str,
        *,
        run_label: str,
        spec_mismatch_policy: str,
    ) -> None:
        assert run_label
        assert spec_mismatch_policy

    monkeypatch.setattr(eval_script, "load_stack_config", fake_load_stack_config)
    monkeypatch.setattr(eval_script, "print_startup_banner", fake_banner)

    dependencies = eval_script._eval_startup_dependencies()

    assert dependencies.load_stack_config_fn is fake_load_stack_config
    assert dependencies.print_startup_banner_fn is fake_banner


def test_eval_entrypoint_main_runner_threads_startup_and_dispatch() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_entrypoint_support.main import run_eval_entrypoint_main

    calls: list[str] = []
    parsed_args = SimpleNamespace(kind="args")
    validated_state = SimpleNamespace(run_label="eval_label")
    startup_state = SimpleNamespace(kind="startup")
    startup_dependencies = SimpleNamespace(kind="startup_deps")
    dispatch_dependencies = SimpleNamespace(kind="dispatch_deps")

    class FakeParser:
        def parse_args(self) -> object:
            calls.append("parse_args")
            return parsed_args

    parser_obj = FakeParser()

    def fake_build_parser() -> FakeParser:
        calls.append("build_parser")
        return parser_obj

    def fake_validate(*, parser: object, args: object) -> object:
        calls.append("validate")
        assert parser is parser_obj
        assert args is parsed_args
        return validated_state

    def fake_startup_dependencies() -> object:
        calls.append("startup_dependencies")
        return startup_dependencies

    def fake_prepare_startup(*, args: object, run_label: str, dependencies: object) -> object:
        calls.append("prepare_startup")
        assert args is parsed_args
        assert run_label == "eval_label"
        assert dependencies is startup_dependencies
        return startup_state

    def fake_dispatch_dependencies() -> object:
        calls.append("dispatch_dependencies")
        return dispatch_dependencies

    def fake_dispatch(
        *,
        parser: object,
        args: object,
        validated: object,
        startup: object,
        dependencies: object,
    ) -> None:
        calls.append("dispatch")
        assert parser is parser_obj
        assert args is parsed_args
        assert validated is validated_state
        assert startup is startup_state
        assert dependencies is dispatch_dependencies

    run_eval_entrypoint_main(
        build_eval_parser_fn=fake_build_parser,
        validate_eval_args_fn=fake_validate,
        prepare_eval_startup_fn=fake_prepare_startup,
        run_eval_dispatch_fn=fake_dispatch,
        eval_startup_dependencies_fn=fake_startup_dependencies,
        eval_dispatch_dependencies_fn=fake_dispatch_dependencies,
    )

    assert calls == [
        "build_parser",
        "parse_args",
        "validate",
        "startup_dependencies",
        "prepare_startup",
        "dispatch_dependencies",
        "dispatch",
    ]


def test_eval_entrypoint_runtime_threads_facade_globals(monkeypatch) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows import eval_entrypoint_runtime

    calls: list[str] = []
    parser_obj = SimpleNamespace(kind="parser")
    startup_dependencies = SimpleNamespace(kind="startup_deps")
    dispatch_dependencies = SimpleNamespace(kind="dispatch_deps")

    def fake_main(**kwargs: object) -> None:
        calls.append("main")
        assert kwargs["build_eval_parser_fn"]() is parser_obj
        assert kwargs["eval_startup_dependencies_fn"]() is startup_dependencies
        assert kwargs["eval_dispatch_dependencies_fn"]() is dispatch_dependencies

    def fake_startup_dependencies(entrypoint_globals: object) -> object:
        calls.append("startup_dependencies")
        assert entrypoint_globals is globals_map
        return startup_dependencies

    def fake_dispatch_dependencies(entrypoint_globals: object) -> object:
        calls.append("dispatch_dependencies")
        assert entrypoint_globals is globals_map
        return dispatch_dependencies

    monkeypatch.setattr(eval_entrypoint_runtime, "run_eval_entrypoint_main", fake_main)
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_startup_dependencies",
        fake_startup_dependencies,
    )
    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_dispatch_dependencies",
        fake_dispatch_dependencies,
    )
    globals_map = {
        "build_eval_parser": lambda: parser_obj,
        "validate_eval_args": object(),
        "prepare_eval_startup": object(),
        "run_eval_dispatch": object(),
    }

    eval_entrypoint_runtime.run_eval_entrypoint(entrypoint_globals=globals_map)

    assert calls == ["main", "startup_dependencies", "dispatch_dependencies"]


def test_eval_entrypoint_runtime_canonical_wrapper_uses_facade_globals(monkeypatch, tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    from weiss_rl.workflows import eval_entrypoint_runtime

    observed: dict[str, object] = {}
    dependencies = object()
    pipeline = object()
    entrypoint_adapter = object()

    def fake_dependencies(entrypoint_globals: object) -> object:
        observed["dependency_globals"] = entrypoint_globals
        return dependencies

    def fake_wrapper(**kwargs: object) -> int:
        observed["wrapper"] = kwargs
        assert kwargs["canonical_dependencies_fn"]() is dependencies
        return 41

    monkeypatch.setattr(
        eval_entrypoint_runtime,
        "build_eval_entrypoint_canonical_dependencies",
        fake_dependencies,
    )
    monkeypatch.setattr(eval_entrypoint_runtime, "run_entrypoint_canonical_eval_pipeline", fake_wrapper)
    globals_map = {
        "run_canonical_eval_pipeline": pipeline,
        "run_canonical_eval_entrypoint_pipeline": entrypoint_adapter,
    }
    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")

    result = eval_entrypoint_runtime.run_eval_entrypoint_canonical_pipeline(
        entrypoint_globals=globals_map,
        parser=parser,
        stack=stack,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=["B0 RandomLegal"],
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=8,
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=True,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
    )

    assert result == 41
    assert observed["dependency_globals"] is globals_map
    wrapper_call = observed["wrapper"]
    assert wrapper_call["parser"] is parser
    assert wrapper_call["stack"] is stack
    assert wrapper_call["run_canonical_eval_pipeline_fn"] is pipeline
    assert wrapper_call["run_canonical_eval_entrypoint_pipeline_fn"] is entrypoint_adapter


def test_eval_entrypoint_compat_canonical_wrapper_forwards_callables(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_entrypoint_support.compat import run_entrypoint_canonical_eval_pipeline

    observed: dict[str, object] = {}
    dependencies = object()

    def fake_dependencies() -> object:
        observed["dependencies_called"] = True
        return dependencies

    def fake_pipeline(**kwargs: object) -> int:
        observed["pipeline"] = kwargs
        return 29

    def fake_entrypoint_adapter(**kwargs: object) -> int:
        observed["adapter"] = kwargs
        return kwargs["run_canonical_eval_pipeline_fn"](
            parser=kwargs["parser"],
            stack=kwargs["stack"],
            run_dir=kwargs["run_dir"],
            final_eval_dir=kwargs["final_eval_dir"],
            policy_ids=kwargs["policy_ids"],
            snapshot_registry_path=kwargs["snapshot_registry_path"],
            dev_eval_summaries_path=kwargs["dev_eval_summaries_path"],
            b1_baseline_run_dir=kwargs["b1_baseline_run_dir"],
            bootstrap_samples=kwargs["bootstrap_samples"],
            paired_seed_limit=kwargs["paired_seed_limit"],
            stage1_paired_seeds=kwargs["stage1_paired_seeds"],
            max_paired_seeds=kwargs["max_paired_seeds"],
            skip_metagame=kwargs["skip_metagame"],
            study_config_path=kwargs["study_config_path"],
            skip_figures=kwargs["skip_figures"],
            skip_readiness=kwargs["skip_readiness"],
            git_commit_override=kwargs["git_commit_override"],
            dependencies=kwargs["canonical_dependencies_fn"](),
        )

    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")

    result = run_entrypoint_canonical_eval_pipeline(
        parser=parser,
        stack=stack,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=["B0 RandomLegal"],
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=8,
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=True,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        canonical_dependencies_fn=fake_dependencies,
        run_canonical_eval_pipeline_fn=fake_pipeline,
        run_canonical_eval_entrypoint_pipeline_fn=fake_entrypoint_adapter,
    )

    assert result == 29
    assert observed["dependencies_called"] is True
    adapter_call = observed["adapter"]
    assert adapter_call["canonical_dependencies_fn"] is fake_dependencies
    assert adapter_call["run_canonical_eval_pipeline_fn"] is fake_pipeline
    pipeline_call = observed["pipeline"]
    assert pipeline_call["parser"] is parser
    assert pipeline_call["stack"] is stack
    assert pipeline_call["run_dir"] == tmp_path / "run"
    assert pipeline_call["final_eval_dir"] == tmp_path / "final"
    assert pipeline_call["policy_ids"] == ["B0 RandomLegal"]
    assert pipeline_call["snapshot_registry_path"] == tmp_path / "registry.json"
    assert pipeline_call["dev_eval_summaries_path"] == tmp_path / "dev.json"
    assert pipeline_call["b1_baseline_run_dir"] == tmp_path / "b1"
    assert pipeline_call["bootstrap_samples"] == 8
    assert pipeline_call["paired_seed_limit"] == 1
    assert pipeline_call["stage1_paired_seeds"] == 2
    assert pipeline_call["max_paired_seeds"] == 3
    assert pipeline_call["skip_metagame"] is True
    assert pipeline_call["study_config_path"] == tmp_path / "study.yaml"
    assert pipeline_call["skip_figures"] is True
    assert pipeline_call["skip_readiness"] is True
    assert pipeline_call["git_commit_override"] == "abc123"
    assert pipeline_call["dependencies"] is dependencies


def test_canonical_eval_entrypoint_request_preserves_flat_kwargs(tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.entrypoint_request import (
        canonical_eval_entrypoint_request,
        run_canonical_entrypoint_request_adapter,
        run_canonical_entrypoint_request_pipeline,
    )

    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(name="stack")
    dependencies = object()
    request = canonical_eval_entrypoint_request(
        parser=parser,
        stack=stack,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=("B0 RandomLegal", "policy_000100"),
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples="8",
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=1,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=0,
        skip_readiness=True,
        git_commit_override=123,
    )

    entrypoint_kwargs = request.entrypoint_kwargs()
    pipeline_kwargs = request.pipeline_kwargs(dependencies=dependencies)

    assert request.policy_ids == ["B0 RandomLegal", "policy_000100"]
    assert request.bootstrap_samples == 8
    assert request.skip_metagame is True
    assert request.skip_figures is False
    assert request.git_commit_override == "123"
    assert entrypoint_kwargs["parser"] is parser
    assert entrypoint_kwargs["stack"] is stack
    assert entrypoint_kwargs["run_dir"] == tmp_path / "run"
    assert entrypoint_kwargs["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert "dependencies" not in entrypoint_kwargs
    assert pipeline_kwargs["dependencies"] is dependencies

    observed: dict[str, object] = {}

    def fake_pipeline(**kwargs: object) -> int:
        observed["pipeline"] = kwargs
        return 19

    def fake_adapter(**kwargs: object) -> int:
        observed["adapter"] = kwargs
        return 23

    assert (
        run_canonical_entrypoint_request_pipeline(
            request=request,
            dependencies=dependencies,
            run_canonical_eval_pipeline_fn=fake_pipeline,
        )
        == 19
    )
    assert observed["pipeline"] == pipeline_kwargs
    assert (
        run_canonical_entrypoint_request_adapter(
            request=request,
            canonical_dependencies_fn=lambda: dependencies,
            run_canonical_eval_pipeline_fn=fake_pipeline,
            run_canonical_eval_entrypoint_pipeline_fn=fake_adapter,
        )
        == 23
    )
    adapter_call = observed["adapter"]
    assert adapter_call["parser"] is parser
    assert adapter_call["canonical_dependencies_fn"]() is dependencies
    assert adapter_call["run_canonical_eval_pipeline_fn"] is fake_pipeline


def test_canonical_eval_entrypoint_adapter_injects_dependencies(tmp_path: Path) -> None:
    import argparse
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.entrypoint_adapter import run_canonical_eval_entrypoint_pipeline

    observed: dict[str, object] = {}
    dependencies = object()

    def fake_pipeline(**kwargs: object) -> int:
        observed.update(kwargs)
        return 17

    result = run_canonical_eval_entrypoint_pipeline(
        parser=argparse.ArgumentParser(),
        stack=SimpleNamespace(name="stack"),
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final",
        policy_ids=["B0 RandomLegal"],
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=8,
        paired_seed_limit=1,
        stage1_paired_seeds=2,
        max_paired_seeds=3,
        skip_metagame=True,
        study_config_path=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        canonical_dependencies_fn=lambda: dependencies,
        run_canonical_eval_pipeline_fn=fake_pipeline,
    )

    assert result == 17
    assert observed["run_dir"] == tmp_path / "run"
    assert observed["final_eval_dir"] == tmp_path / "final"
    assert observed["policy_ids"] == ["B0 RandomLegal"]
    assert observed["snapshot_registry_path"] == tmp_path / "registry.json"
    assert observed["dev_eval_summaries_path"] == tmp_path / "dev.json"
    assert observed["b1_baseline_run_dir"] == tmp_path / "b1"
    assert observed["bootstrap_samples"] == 8
    assert observed["paired_seed_limit"] == 1
    assert observed["stage1_paired_seeds"] == 2
    assert observed["max_paired_seeds"] == 3
    assert observed["skip_metagame"] is True
    assert observed["study_config_path"] == tmp_path / "study.yaml"
    assert observed["skip_figures"] is True
    assert observed["skip_readiness"] is True
    assert observed["git_commit_override"] == "abc123"
    assert observed["dependencies"] is dependencies


def test_canonical_eval_runtime_phase_persists_selection_before_loading_policies(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.phases import (
        CanonicalEvalRunState,
        resolve_canonical_eval_runtime_state,
    )

    calls: list[str] = []
    observed: dict[str, object] = {}
    layout = SimpleNamespace()
    evaluation = SimpleNamespace(
        eval_assert_sorted_legal_ids=True,
        replay_capture_rate_eval=0.25,
        regression_capture_count=2,
        final_matrix_stage1_paired_seeds=2,
        final_matrix_stage2_adaptive_max_paired_seeds=3,
    )
    stack = SimpleNamespace(
        root=tmp_path,
        seed_sets={"report_eval": tmp_path / "report_eval_seeds.txt"},
        config=SimpleNamespace(evaluation=evaluation),
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=evaluation,
        study_config=None,
    )

    class FakeContract:
        spec_bundle = {
            "observation": {"obs_len": 512},
            "action": {"action_space_size": 9, "pass_action_id": 8},
        }

    def fake_resolve_policy_ids_for_run_fn(**_kwargs: object) -> tuple[list[str], dict[str, object], None, None]:
        calls.append("resolve_policy_ids")
        return ["B0 RandomLegal", "policy_000100"], {"status": "resolved"}, None, None

    def fake_persist_policy_selection_in_manifest_fn(**kwargs: object) -> None:
        calls.append("persist_selection")
        observed["persisted"] = kwargs

    def fake_load_verified_simulator_contract_fn(*_args: object, **_kwargs: object) -> FakeContract:
        calls.append("load_contract")
        return FakeContract()

    def fake_resolve_eval_policies_fn(**kwargs: object) -> list[str]:
        calls.append("resolve_eval_policies")
        observed["policy_resolution"] = kwargs
        return ["policy-object"]

    def fake_simulator_eval_runner_cls(**kwargs: object) -> object:
        calls.append("build_runner")
        observed["runner"] = kwargs
        return object()

    def fake_parse_seed_file_fn(path: Path) -> list[int]:
        calls.append("parse_seeds")
        observed["seed_file"] = path
        return [101, 202, 303]

    dependencies = SimpleNamespace(
        resolve_policy_ids_for_run_fn=fake_resolve_policy_ids_for_run_fn,
        persist_policy_selection_in_manifest_fn=fake_persist_policy_selection_in_manifest_fn,
        load_verified_simulator_contract_fn=fake_load_verified_simulator_contract_fn,
        resolve_eval_policies_fn=fake_resolve_eval_policies_fn,
        simulator_eval_runner_cls=fake_simulator_eval_runner_cls,
        parse_seed_file_fn=fake_parse_seed_file_fn,
    )

    runtime_state = resolve_canonical_eval_runtime_state(
        stack=stack,
        run_dir=tmp_path / "run",
        policy_ids=[],
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        b1_baseline_run_dir=tmp_path / "b1",
        paired_seed_limit=2,
        stage1_paired_seeds=None,
        max_paired_seeds=None,
        run_state=run_state,
        dependencies=dependencies,
    )

    assert calls == [
        "resolve_policy_ids",
        "persist_selection",
        "load_contract",
        "resolve_eval_policies",
        "build_runner",
        "parse_seeds",
    ]
    assert runtime_state.policy_ids == ["B0 RandomLegal", "policy_000100"]
    assert runtime_state.paired_seeds == [101, 202]
    assert runtime_state.paired_seed_limit == 2
    assert runtime_state.stage1_paired_seeds == 2
    assert runtime_state.max_paired_seeds == 2
    assert observed["policy_resolution"]["observation_dim"] == 512
    assert observed["policy_resolution"]["action_dim"] == 9
    assert observed["policy_resolution"]["b1_baseline_run_dir"] == tmp_path / "b1"
    assert observed["runner"]["pass_action_id"] == 8
    assert observed["runner"]["require_sorted_legal_ids"] is True


def test_canonical_eval_seed_budget_preserves_limit_defaults_and_errors(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.seed_budget import resolve_canonical_eval_seed_budget

    seed_path = tmp_path / "report_eval_seeds.txt"
    stack = SimpleNamespace(seed_sets={"report_eval": seed_path})
    evaluation = SimpleNamespace(
        final_matrix_stage1_paired_seeds=4,
        final_matrix_stage2_adaptive_max_paired_seeds=8,
    )
    dependencies = SimpleNamespace(parse_seed_file_fn=lambda path: [11, 22, 33] if path == seed_path else [])

    seed_budget = resolve_canonical_eval_seed_budget(
        stack=stack,
        evaluation=evaluation,
        paired_seed_limit=2,
        stage1_paired_seeds=None,
        max_paired_seeds=None,
        dependencies=dependencies,
    )

    assert seed_budget.seed_file_path == seed_path
    assert seed_budget.paired_seeds == [11, 22]
    assert seed_budget.paired_seed_limit == 2
    assert seed_budget.stage1_paired_seeds == 2
    assert seed_budget.max_paired_seeds == 2

    with pytest.raises(ValueError, match=r"stage1 paired seeds \(3\) cannot exceed max paired seeds \(2\)"):
        resolve_canonical_eval_seed_budget(
            stack=stack,
            evaluation=evaluation,
            paired_seed_limit=None,
            stage1_paired_seeds=3,
            max_paired_seeds=2,
            dependencies=dependencies,
        )

    empty_dependencies = SimpleNamespace(parse_seed_file_fn=lambda _path: [])
    with pytest.raises(ValueError, match="report_eval seed file produced no usable seeds"):
        resolve_canonical_eval_seed_budget(
            stack=stack,
            evaluation=evaluation,
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
            dependencies=empty_dependencies,
        )


def test_canonical_eval_output_phase_preserves_final_eval_metadata(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.phases import (
        CanonicalEvalRunState,
        CanonicalEvalRuntimeState,
        write_canonical_eval_outputs,
    )

    class FakeLayout:
        final_eval_dir = tmp_path / "run" / "eval" / "final_eval"
        metagame_dir = tmp_path / "run" / "eval" / "metagame"
        figures_paper_dir = tmp_path / "run" / "figures" / "paper"
        paper_readiness_summary_path = tmp_path / "run" / "paper_readiness_summary.json"

        def final_eval_summary_json(self) -> Path:
            return self.final_eval_dir / "summary.json"

        def replay_verification_json(self) -> Path:
            return self.final_eval_dir / "replay_verification.json"

    tensorboard_logger = SimpleNamespace(enabled=False)
    evaluation = SimpleNamespace(
        stop_rules={"minimum": 1},
        final_policy_set_selection=SimpleNamespace(folding="seat_swap_mean"),
        final_policy_set_size=2,
    )
    run_state = CanonicalEvalRunState(
        layout=FakeLayout(),
        tensorboard_logger=tensorboard_logger,
        manifest={"run_id256": "ab" * 32, "config_hash256": "cd" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=evaluation,
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=tmp_path / "registry.json",
        dev_eval_summaries_path=tmp_path / "dev_eval.json",
        runner=object(),
        paired_seeds=[101, 202],
        paired_seed_limit=2,
        stage1_paired_seeds=1,
        max_paired_seeds=2,
        seed_file_path=tmp_path / "report_eval_seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )
    observed: dict[str, object] = {}

    def fake_run_final_eval_fn(**kwargs: object) -> dict[str, object]:
        observed["final_eval"] = kwargs
        return {"kind": "summary"}

    dependencies = SimpleNamespace(
        tensorboard_unavailable_reason_fn=lambda: "no writer",
        run_final_eval_fn=fake_run_final_eval_fn,
        ensure_run_level_report_scaffolding_fn=lambda layout: observed.setdefault("scaffold", layout),
        update_run_level_reports_fn=lambda **kwargs: observed.setdefault("reports", kwargs),
    )

    result = write_canonical_eval_outputs(
        run_dir=tmp_path / "run",
        bootstrap_samples=8,
        skip_metagame=True,
        skip_figures=True,
        skip_readiness=True,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )

    assert result == 0
    final_eval_call = observed["final_eval"]
    assert final_eval_call["paired_seeds"] == [101, 202]
    assert final_eval_call["stage1_paired_seeds"] == 1
    assert final_eval_call["max_paired_seeds"] == 2
    assert final_eval_call["sample_count"] == 8
    assert final_eval_call["metadata"]["pipeline"] == {
        "kind": "canonical_eval_pipeline_v1",
        "selection": {"status": "resolved"},
        "seed_file": (tmp_path / "report_eval_seeds.txt").as_posix(),
        "paired_seed_limit": 2,
    }
    assert final_eval_call["metadata"]["recommended_focal_policy_id"] == "policy_000100"
    assert observed["reports"]["final_eval_payload"] == {"kind": "summary"}
    assert observed["reports"]["metagame_payload"] is None
    assert observed["reports"]["figure_paths"] == ()
    assert observed["reports"]["readiness_payload"] is None
    assert "Resolved policy set: ['B0 RandomLegal', 'policy_000100']" in capsys.readouterr().out


def test_canonical_supplemental_outputs_builds_thesis_artifacts_in_order(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import build_canonical_supplemental_outputs

    calls: list[str] = []
    observed: dict[str, object] = {}
    layout = SimpleNamespace(
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "config_hash256": "cd" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=SimpleNamespace(metagame={"m": 1}, sensitivity={"s": 2}),
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    def fake_metagame(**kwargs: object) -> dict[str, object]:
        calls.append("metagame")
        observed["metagame"] = kwargs
        return {"metagame": "payload"}

    def fake_figures(run_dir: Path) -> tuple[Path, ...]:
        calls.append("figures")
        observed["figures"] = run_dir
        return (run_dir / "figures" / "paper" / "seat_bias.pdf",)

    def fake_scaffold(scaffold_layout: object) -> None:
        calls.append("scaffold")
        observed["scaffold"] = scaffold_layout

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        calls.append("readiness")
        observed["readiness"] = kwargs
        return {"passed": True}

    def fake_write_readiness(path: Path, payload: dict[str, object]) -> None:
        calls.append("write_readiness")
        observed["write_readiness"] = (path, payload)

    dependencies = SimpleNamespace(
        build_sensitivity_report_fn=fake_metagame,
        render_paper_figures_fn=fake_figures,
        ensure_run_level_report_scaffolding_fn=fake_scaffold,
        build_paper_readiness_summary_fn=fake_readiness,
        write_paper_readiness_json_fn=fake_write_readiness,
    )

    outputs = build_canonical_supplemental_outputs(
        run_dir=tmp_path / "run",
        skip_metagame=False,
        skip_figures=False,
        skip_readiness=False,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=dependencies,
    )

    assert calls == ["metagame", "figures", "scaffold", "readiness", "write_readiness"]
    assert outputs.metagame_payload == {"metagame": "payload"}
    assert outputs.figure_paths == (tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",)
    assert outputs.readiness_payload == {"passed": True}
    assert observed["metagame"] == {
        "final_eval_dir": layout.final_eval_dir,
        "out_dir": layout.metagame_dir,
        "metagame_config": {"m": 1},
        "sensitivity_config": {"s": 2},
    }
    assert observed["figures"] == tmp_path / "run"
    assert observed["scaffold"] is layout
    assert observed["readiness"] == {"run_dir": tmp_path / "run", "focal_policy_id": "policy_000100"}
    assert observed["write_readiness"] == (layout.paper_readiness_summary_path, {"passed": True})


def test_canonical_output_bundle_builds_final_eval_before_supplemental_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import weiss_rl.workflows.canonical_eval.output_bundle as output_bundle_module
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    calls: list[str] = []
    run_state = CanonicalEvalRunState(
        layout=SimpleNamespace(),
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32, "config_hash256": "cd" * 32, "spec_hash256": "ef" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    def fake_final_eval(**kwargs: object) -> dict[str, object]:
        calls.append("final")
        assert kwargs["bootstrap_samples"] == 8
        assert kwargs["run_state"] is run_state
        assert kwargs["runtime_state"] is runtime_state
        return {"final": "payload"}

    def fake_supplemental(**kwargs: object) -> CanonicalEvalSupplementalOutputs:
        calls.append("supplemental")
        assert kwargs["run_dir"] == tmp_path / "run"
        assert kwargs["skip_metagame"] is True
        assert kwargs["skip_figures"] is False
        assert kwargs["skip_readiness"] is True
        assert kwargs["run_state"] is run_state
        assert kwargs["runtime_state"] is runtime_state
        return CanonicalEvalSupplementalOutputs(
            metagame_payload=None,
            figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
            readiness_payload=None,
        )

    monkeypatch.setattr(output_bundle_module, "run_canonical_final_eval_output", fake_final_eval)
    monkeypatch.setattr(output_bundle_module, "build_canonical_supplemental_outputs", fake_supplemental)

    bundle = output_bundle_module.build_canonical_eval_output_bundle(
        run_dir=tmp_path / "run",
        bootstrap_samples=8,
        skip_metagame=True,
        skip_figures=False,
        skip_readiness=True,
        run_state=run_state,
        runtime_state=runtime_state,
        dependencies=SimpleNamespace(),
    )

    assert calls == ["final", "supplemental"]
    assert bundle.final_eval_payload == {"final": "payload"}
    assert bundle.supplemental.figure_paths == (tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",)


def test_canonical_metagame_output_forwards_study_configs(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.metagame_outputs import build_canonical_metagame_output

    observed: dict[str, object] = {}
    layout = SimpleNamespace(
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
    )
    study_config = SimpleNamespace(metagame={"m": 1}, sensitivity={"s": 2})

    def fake_metagame(**kwargs: object) -> dict[str, object]:
        observed["metagame"] = kwargs
        return {"metagame": "payload"}

    payload = build_canonical_metagame_output(
        layout=layout,
        study_config=study_config,
        dependencies=SimpleNamespace(build_sensitivity_report_fn=fake_metagame),
    )

    assert payload == {"metagame": "payload"}
    assert observed["metagame"] == {
        "final_eval_dir": layout.final_eval_dir,
        "out_dir": layout.metagame_dir,
        "metagame_config": {"m": 1},
        "sensitivity_config": {"s": 2},
    }


def test_canonical_figure_outputs_forward_run_dir(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.figure_outputs import build_canonical_figure_outputs

    observed: dict[str, object] = {}

    def fake_figures(run_dir: Path) -> tuple[Path, ...]:
        observed["run_dir"] = run_dir
        return (run_dir / "figures" / "paper" / "seat_bias.pdf",)

    outputs = build_canonical_figure_outputs(
        run_dir=tmp_path / "run",
        dependencies=SimpleNamespace(render_paper_figures_fn=fake_figures),
    )

    assert observed["run_dir"] == tmp_path / "run"
    assert outputs == (tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",)


def test_canonical_readiness_output_writes_focal_policy_summary(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.readiness_outputs import build_canonical_readiness_output
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRuntimeState

    observed: dict[str, object] = {}
    layout = SimpleNamespace(paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json")
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    def fake_readiness(**kwargs: object) -> dict[str, object]:
        observed["readiness"] = kwargs
        return {"passed": True, "focal_policy_id": kwargs["focal_policy_id"]}

    def fake_write(path: Path, payload: dict[str, object]) -> None:
        observed["write"] = (path, payload)

    payload = build_canonical_readiness_output(
        run_dir=tmp_path / "run",
        layout=layout,
        runtime_state=runtime_state,
        dependencies=SimpleNamespace(
            build_paper_readiness_summary_fn=fake_readiness,
            write_paper_readiness_json_fn=fake_write,
        ),
    )

    assert payload == {"passed": True, "focal_policy_id": "policy_000100"}
    assert observed["readiness"] == {"run_dir": tmp_path / "run", "focal_policy_id": "policy_000100"}
    assert observed["write"] == (layout.paper_readiness_summary_path, payload)


def test_canonical_output_message_renderer_handles_optional_outputs(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.cli_messages import render_canonical_eval_output_messages
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    layout = SimpleNamespace(
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        figures_paper_dir=tmp_path / "run" / "figures" / "paper",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
        final_eval_summary_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "summary.json",
        replay_verification_json=lambda: tmp_path / "run" / "eval" / "diagnostics" / "replay_verification.json",
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )

    minimal_messages = render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=CanonicalEvalSupplementalOutputs(
            metagame_payload=None,
            figure_paths=(),
            readiness_payload=None,
        ),
    )

    assert minimal_messages == (
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
        "Resolved policy set: ['B0 RandomLegal', 'policy_000100']",
    )

    full_messages = render_canonical_eval_output_messages(
        layout=layout,
        runtime_state=runtime_state,
        supplemental=CanonicalEvalSupplementalOutputs(
            metagame_payload={"kind": "summary"},
            figure_paths=(
                tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",
                tmp_path / "run" / "figures" / "paper" / "main_eval.png",
            ),
            readiness_payload={"passed": False},
        ),
    )

    assert full_messages == (
        f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}",
        f"Canonical replay verification JSON: {layout.replay_verification_json()}",
        f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}",
        f"Rendered 2 paper figure files to {layout.figures_paper_dir}",
        f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}",
        "Paper readiness: failed",
        "Resolved policy set: ['B0 RandomLegal', 'policy_000100']",
    )


def test_canonical_tensorboard_publication_handles_enabled_and_disabled(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs
    from weiss_rl.workflows.canonical_eval.tensorboard_publication import (
        begin_canonical_eval_tensorboard_logging,
        publish_canonical_eval_tensorboard_summaries,
    )

    class FakeTensorBoardLogger:
        def __init__(self, *, enabled: bool) -> None:
            self.enabled = enabled
            self.calls: list[tuple[str, object]] = []

        def log_text(self, tag: str, payload: object) -> None:
            self.calls.append(("text", (tag, payload)))

        def log_final_eval_summary(self, payload: object, *, step: int) -> None:
            self.calls.append(("final", (payload, step)))

        def log_metagame_summary(self, payload: object, *, metagame_dir: Path, step: int) -> None:
            self.calls.append(("metagame", (payload, metagame_dir, step)))

        def log_paper_readiness(self, payload: object, *, step: int) -> None:
            self.calls.append(("readiness", (payload, step)))

    layout = SimpleNamespace(metagame_dir=tmp_path / "run" / "eval" / "metagame")
    enabled_logger = FakeTensorBoardLogger(enabled=True)
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=enabled_logger,
        manifest={"run_id256": "ab" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    supplemental = CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(),
        readiness_payload={"passed": True},
    )

    begin_canonical_eval_tensorboard_logging(
        run_state=run_state,
        dependencies=SimpleNamespace(tensorboard_unavailable_reason_fn=lambda: None),
    )
    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=enabled_logger,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
    )

    assert enabled_logger.calls == [
        ("text", ("eval/run/manifest", {"run_id256": "ab" * 32})),
        ("final", ({"summary": "payload"}, 0)),
        ("metagame", ({"meta": "payload"}, layout.metagame_dir, 0)),
        ("readiness", ({"passed": True}, 0)),
    ]

    disabled_logger = FakeTensorBoardLogger(enabled=False)
    begin_canonical_eval_tensorboard_logging(
        run_state=CanonicalEvalRunState(
            layout=layout,
            tensorboard_logger=disabled_logger,
            manifest={},
            run_id256="",
            evaluation=SimpleNamespace(),
            study_config=None,
        ),
        dependencies=SimpleNamespace(tensorboard_unavailable_reason_fn=lambda: None),
    )
    publish_canonical_eval_tensorboard_summaries(
        layout=layout,
        tensorboard_logger=disabled_logger,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
    )

    assert disabled_logger.calls == []
    assert "TensorBoard logging is disabled for eval: SummaryWriter unavailable" in capsys.readouterr().err


def test_canonical_report_publication_updates_run_level_reports(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.report_publication import publish_canonical_eval_run_reports
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    layout = SimpleNamespace()
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=SimpleNamespace(),
        manifest={"run_id256": "ab" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )
    supplemental = CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
        readiness_payload={"passed": True},
    )
    observed: dict[str, object] = {}

    publish_canonical_eval_run_reports(
        run_dir=tmp_path / "run",
        run_state=run_state,
        runtime_state=runtime_state,
        final_eval_payload={"summary": "payload"},
        supplemental=supplemental,
        dependencies=SimpleNamespace(
            update_run_level_reports_fn=lambda **kwargs: observed.setdefault("reports", kwargs)
        ),
    )

    assert observed["reports"] == {
        "layout": layout,
        "run_dir": tmp_path / "run",
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "selection_details": {"status": "resolved"},
        "final_eval_payload": {"summary": "payload"},
        "metagame_payload": {"meta": "payload"},
        "figure_paths": supplemental.figure_paths,
        "readiness_payload": {"passed": True},
    }


def test_canonical_output_publisher_updates_reports_tensorboard_and_cli(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.canonical_eval.publisher import (
        begin_canonical_eval_output_logging,
        publish_canonical_eval_outputs,
    )
    from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
    from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs

    class FakeTensorBoardLogger:
        enabled = True

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def log_text(self, tag: str, payload: object) -> None:
            self.calls.append(("text", (tag, payload)))

        def log_final_eval_summary(self, payload: object, *, step: int) -> None:
            self.calls.append(("final", (payload, step)))

        def log_metagame_summary(self, payload: object, *, metagame_dir: Path, step: int) -> None:
            self.calls.append(("metagame", (payload, metagame_dir, step)))

        def log_paper_readiness(self, payload: object, *, step: int) -> None:
            self.calls.append(("readiness", (payload, step)))

    tensorboard_logger = FakeTensorBoardLogger()
    layout = SimpleNamespace(
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        figures_paper_dir=tmp_path / "run" / "figures" / "paper",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
        final_eval_summary_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "summary.json",
        replay_verification_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "replay_verification.json",
    )
    run_state = CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=tensorboard_logger,
        manifest={"run_id256": "ab" * 32},
        run_id256="ab" * 32,
        evaluation=SimpleNamespace(),
        study_config=None,
    )
    runtime_state = CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )
    supplemental = CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
        readiness_payload={"passed": True},
    )
    observed: dict[str, object] = {}
    dependencies = SimpleNamespace(
        tensorboard_unavailable_reason_fn=lambda: None,
        update_run_level_reports_fn=lambda **kwargs: observed.setdefault("reports", kwargs),
    )
    final_eval_payload = {"summary": "payload"}

    begin_canonical_eval_output_logging(run_state=run_state, dependencies=dependencies)
    publish_canonical_eval_outputs(
        run_dir=tmp_path / "run",
        run_state=run_state,
        runtime_state=runtime_state,
        final_eval_payload=final_eval_payload,
        supplemental=supplemental,
        dependencies=dependencies,
    )

    assert observed["reports"]["final_eval_payload"] is final_eval_payload
    assert observed["reports"]["metagame_payload"] == {"meta": "payload"}
    assert observed["reports"]["figure_paths"] == supplemental.figure_paths
    assert observed["reports"]["readiness_payload"] == {"passed": True}
    assert tensorboard_logger.calls == [
        ("text", ("eval/run/manifest", {"run_id256": "ab" * 32})),
        ("final", (final_eval_payload, 0)),
        ("metagame", ({"meta": "payload"}, layout.metagame_dir, 0)),
        ("readiness", ({"passed": True}, 0)),
    ]
    output = capsys.readouterr().out
    assert f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}" in output
    assert f"Canonical replay verification JSON: {layout.replay_verification_json()}" in output
    assert f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}" in output
    assert f"Rendered 1 paper figure files to {layout.figures_paper_dir}" in output
    assert f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}" in output
    assert "Paper readiness: passed" in output
    assert "Resolved policy set: ['B0 RandomLegal', 'policy_000100']" in output


def test_eval_startup_validation_preserves_mode_errors() -> None:
    from weiss_rl.workflows.eval_support.eval_parser import build_eval_parser
    from weiss_rl.workflows.eval_support.eval_startup import validate_eval_args

    parser = build_eval_parser()
    args = parser.parse_args(
        [
            "--stack-config",
            "configs/presets/structured_acceptance_standard_thesis_eval.yaml",
            "--run-dir",
            "runs/demo",
            "--episodes-jsonl",
            "runs/demo/eval/final_eval/episodes.jsonl",
        ]
    )

    with pytest.raises(SystemExit):
        validate_eval_args(parser=parser, args=args)


def test_eval_startup_preparation_uses_public_demo_contract_and_banner() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_startup import EvalStartupDependencies, prepare_eval_startup

    observed: dict[str, object] = {}
    stack = SimpleNamespace(root=Path("repo"))
    args = SimpleNamespace(
        stack_config=Path("configs/demo.yaml"),
        config_hash="",
        public_demo=True,
        spec_hash="",
    )

    def fake_banner(
        reported_spec_hash: str,
        config_hash256: str,
        *,
        run_label: str,
        spec_mismatch_policy: str,
    ) -> None:
        observed["banner"] = (reported_spec_hash, config_hash256, run_label, spec_mismatch_policy)

    startup = prepare_eval_startup(
        args=args,
        run_label="demo_eval",
        dependencies=EvalStartupDependencies(
            load_stack_config_fn=lambda path: stack,
            compute_config_hash256_fn=lambda loaded_stack: "c" * 64,
            expected_sha256_fn=lambda value, *, flag_name: "",
            require_matching_hash_fn=lambda **kwargs: observed.setdefault("hash", kwargs),
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            assert_spec_bundle_contract_fn=lambda expected, bundle: observed.setdefault("spec", (expected, bundle)),
            public_demo_spec_hash256_fn=lambda: "d" * 64,
            load_verified_simulator_contract_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("public demo must not load simulator contract")
            ),
            print_startup_banner_fn=fake_banner,
        ),
    )

    assert startup.stack is stack
    assert startup.config_hash256 == "c" * 64
    assert startup.reported_spec_hash == "d" * 64
    assert startup.contract is None
    assert observed["hash"] == {"flag_name": "--config-hash", "expected": "", "actual": "c" * 64}
    assert observed["spec"] == ("", {"spec_hash": "public_demo"})
    assert observed["banner"] == ("d" * 64, "c" * 64, "demo_eval", "hard_fail")


def test_eval_startup_preparation_uses_verified_simulator_contract() -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_startup import EvalStartupDependencies, prepare_eval_startup

    observed: dict[str, object] = {}
    stack = SimpleNamespace(root=Path("repo"))
    contract = SimpleNamespace(spec_hash256="e" * 64)
    args = SimpleNamespace(
        stack_config=Path("configs/thesis.yaml"),
        config_hash="c" * 64,
        public_demo=False,
        spec_hash="e" * 64,
    )

    def fake_load_verified_simulator_contract(*args: object, **kwargs: object) -> object:
        observed["contract_call"] = (args, kwargs)
        return contract

    startup = prepare_eval_startup(
        args=args,
        run_label="canonical_eval",
        dependencies=EvalStartupDependencies(
            load_stack_config_fn=lambda path: stack,
            compute_config_hash256_fn=lambda loaded_stack: "c" * 64,
            expected_sha256_fn=lambda value, *, flag_name: value,
            require_matching_hash_fn=lambda **kwargs: observed.setdefault("hash", kwargs),
            public_demo_spec_bundle_fn=lambda: (_ for _ in ()).throw(
                AssertionError("canonical startup must not load public-demo spec")
            ),
            assert_spec_bundle_contract_fn=lambda *_args: (_ for _ in ()).throw(
                AssertionError("canonical startup must not assert public-demo contract")
            ),
            public_demo_spec_hash256_fn=lambda: (_ for _ in ()).throw(
                AssertionError("canonical startup must not report public-demo hash")
            ),
            load_verified_simulator_contract_fn=fake_load_verified_simulator_contract,
            print_startup_banner_fn=lambda *args, **kwargs: observed.setdefault("banner", (args, kwargs)),
        ),
    )

    assert startup.stack is stack
    assert startup.config_hash256 == "c" * 64
    assert startup.reported_spec_hash == "e" * 64
    assert startup.contract is contract
    assert observed["hash"] == {"flag_name": "--config-hash", "expected": "c" * 64, "actual": "c" * 64}
    assert observed["contract_call"] == ((Path("repo"),), {"expected_spec_hash": "e" * 64})
    assert observed["banner"] == (
        ("e" * 64, "c" * 64),
        {"run_label": "canonical_eval", "spec_mismatch_policy": "hard_fail"},
    )


def test_eval_dispatch_routes_public_demo_with_resolved_paths(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    observed: dict[str, object] = {}
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=True,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "custom_eval",
        public_demo_paired_seeds=7,
        public_demo_bootstrap_samples=11,
        run_label="demo",
    )

    run_eval_dispatch(
        parser=argparse.ArgumentParser(),
        args=args,
        validated=EvalValidatedArgs(
            run_label="demo",
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
        ),
        startup=EvalStartup(
            stack=stack,
            config_hash256="c" * 64,
            reported_spec_hash="d" * 64,
            contract=None,
        ),
        dependencies=EvalDispatchDependencies(
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            public_demo_stop_rules_fn=lambda: "stop_rules",
            run_public_demo_final_eval_fn=lambda **_kwargs: {"policy_ids": []},
            run_public_demo_eval_mode_fn=lambda **kwargs: observed.setdefault("public_demo", kwargs),
            run_canonical_eval_pipeline_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("canonical mode should not run")
            ),
            run_summary_only_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("summary mode should not run")
            ),
            load_eval_game_records_fn=None,
            build_matchup_export_fn=None,
            build_seat_advantage_diagnostics_fn=None,
            write_matchup_diagnostics_json_fn=None,
            write_matchup_summary_csv_fn=None,
            write_matchup_summary_json_fn=None,
        ),
    )

    call = observed["public_demo"]
    assert call["stack"] is stack
    assert call["run_dir"] == (tmp_path / "run").resolve()
    assert call["final_eval_dir"] == (tmp_path / "custom_eval").resolve()
    assert call["paired_seed_limit"] == 7
    assert call["bootstrap_samples"] == 11
    assert call["config_hash256"] == "c" * 64
    assert call["spec_hash256"] == "d" * 64
    assert "Verified public-demo spec bundle" in capsys.readouterr().out


def test_eval_dispatch_request_preserves_route_payloads(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_dispatch_request import eval_dispatch_request
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    parser = argparse.ArgumentParser()
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "final_eval",
        policy_id=("B0 RandomLegal", "policy_000100"),
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples="13",
        skip_metagame=1,
        study_config=tmp_path / "study.yaml",
        skip_figures=0,
        skip_readiness=True,
        git_commit_override=123,
        episodes_jsonl=tmp_path / "episodes.jsonl",
        summary_json=tmp_path / "summary.json",
        summary_csv=tmp_path / "summary.csv",
        diagnostics_json=tmp_path / "diagnostics.json",
        bootstrap_seed="23",
        public_demo_paired_seeds="7",
        public_demo_bootstrap_samples="11",
    )
    validated = EvalValidatedArgs(
        run_label="canonical",
        paired_seed_limit=5,
        stage1_paired_seeds=3,
        max_paired_seeds=9,
    )
    startup = EvalStartup(
        stack=stack,
        config_hash256="c" * 64,
        reported_spec_hash="e" * 64,
        contract=SimpleNamespace(simulator={"compatibility_hash": "compat123"}, spec_hash256="e" * 64),
    )
    dependencies = EvalDispatchDependencies(
        public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
        public_demo_stop_rules_fn="stop-rules",
        run_public_demo_final_eval_fn="public-final",
        run_public_demo_eval_mode_fn="public-mode",
        run_canonical_eval_pipeline_fn="canonical",
        run_summary_only_eval_mode_fn="summary",
        load_eval_game_records_fn="load-records",
        build_matchup_export_fn="build-export",
        build_seat_advantage_diagnostics_fn="seat-diagnostics",
        write_matchup_diagnostics_json_fn="write-diagnostics",
        write_matchup_summary_csv_fn="write-csv",
        write_matchup_summary_json_fn="write-json",
    )

    request = eval_dispatch_request(
        parser=parser,
        args=args,
        validated=validated,
        startup=startup,
        dependencies=dependencies,
    )

    assert request.is_public_demo is False
    assert request.has_run_dir is True
    assert request.has_episodes_jsonl is True
    assert request.public_demo_kwargs() == {
        "stack": stack,
        "run_dir": (tmp_path / "run").resolve(),
        "final_eval_dir": (tmp_path / "final_eval").resolve(),
        "paired_seed_limit": 7,
        "bootstrap_samples": 11,
        "config_hash256": "c" * 64,
        "spec_hash256": "e" * 64,
        "public_demo_stop_rules_fn": "stop-rules",
        "run_public_demo_final_eval_fn": "public-final",
    }
    assert request.canonical_kwargs() == {
        "parser": parser,
        "stack": stack,
        "run_dir": (tmp_path / "run").resolve(),
        "final_eval_dir": (tmp_path / "final_eval").resolve(),
        "policy_ids": ["B0 RandomLegal", "policy_000100"],
        "snapshot_registry_path": (tmp_path / "registry.json").resolve(),
        "dev_eval_summaries_path": (tmp_path / "dev_eval.json").resolve(),
        "b1_baseline_run_dir": (tmp_path / "b1").resolve(),
        "bootstrap_samples": 13,
        "paired_seed_limit": 5,
        "stage1_paired_seeds": 3,
        "max_paired_seeds": 9,
        "skip_metagame": True,
        "study_config_path": (tmp_path / "study.yaml").resolve(),
        "skip_figures": False,
        "skip_readiness": True,
        "git_commit_override": "123",
    }
    assert request.summary_only_kwargs() == {
        "stack": stack,
        "episodes_jsonl": tmp_path / "episodes.jsonl",
        "summary_json": tmp_path / "summary.json",
        "summary_csv": tmp_path / "summary.csv",
        "diagnostics_json": tmp_path / "diagnostics.json",
        "bootstrap_samples": 13,
        "bootstrap_seed": 23,
        "load_eval_game_records_fn": "load-records",
        "build_matchup_export_fn": "build-export",
        "build_seat_advantage_diagnostics_fn": "seat-diagnostics",
        "write_matchup_diagnostics_json_fn": "write-diagnostics",
        "write_matchup_summary_csv_fn": "write-csv",
        "write_matchup_summary_json_fn": "write-json",
    }


def test_eval_dispatch_routes_canonical_with_normalized_args(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    observed: dict[str, object] = {}
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=tmp_path / "run",
        final_eval_dir=tmp_path / "run" / "eval" / "final_eval",
        policy_id=["B0 RandomLegal", "policy_000100"],
        snapshot_registry_json=tmp_path / "registry.json",
        dev_eval_summaries_json=tmp_path / "dev_eval.json",
        b1_baseline_run_dir=tmp_path / "b1",
        bootstrap_samples=13,
        skip_metagame=True,
        study_config=tmp_path / "study.yaml",
        skip_figures=True,
        skip_readiness=True,
        git_commit_override="abc123",
        episodes_jsonl=None,
    )
    contract = SimpleNamespace(
        simulator={"compatibility_hash": "compat123"},
        spec_hash256="e" * 64,
    )

    def fake_canonical(**kwargs: object) -> int:
        observed["canonical"] = kwargs
        return 23

    with pytest.raises(SystemExit) as exc_info:
        run_eval_dispatch(
            parser=argparse.ArgumentParser(),
            args=args,
            validated=EvalValidatedArgs(
                run_label="canonical",
                paired_seed_limit=5,
                stage1_paired_seeds=3,
                max_paired_seeds=9,
            ),
            startup=EvalStartup(
                stack=stack,
                config_hash256="c" * 64,
                reported_spec_hash="e" * 64,
                contract=contract,
            ),
            dependencies=EvalDispatchDependencies(
                public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
                public_demo_stop_rules_fn=None,
                run_public_demo_final_eval_fn=None,
                run_public_demo_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("public demo should not run")
                ),
                run_canonical_eval_pipeline_fn=fake_canonical,
                run_summary_only_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("summary mode should not run")
                ),
                load_eval_game_records_fn=None,
                build_matchup_export_fn=None,
                build_seat_advantage_diagnostics_fn=None,
                write_matchup_diagnostics_json_fn=None,
                write_matchup_summary_csv_fn=None,
                write_matchup_summary_json_fn=None,
            ),
        )

    assert exc_info.value.code == 23
    call = observed["canonical"]
    assert call["stack"] is stack
    assert call["run_dir"] == (tmp_path / "run").resolve()
    assert call["final_eval_dir"] == (tmp_path / "run" / "eval" / "final_eval").resolve()
    assert call["policy_ids"] == ["B0 RandomLegal", "policy_000100"]
    assert call["snapshot_registry_path"] == (tmp_path / "registry.json").resolve()
    assert call["dev_eval_summaries_path"] == (tmp_path / "dev_eval.json").resolve()
    assert call["b1_baseline_run_dir"] == (tmp_path / "b1").resolve()
    assert call["bootstrap_samples"] == 13
    assert call["paired_seed_limit"] == 5
    assert call["stage1_paired_seeds"] == 3
    assert call["max_paired_seeds"] == 9
    assert call["skip_metagame"] is True
    assert call["study_config_path"] == (tmp_path / "study.yaml").resolve()
    assert call["skip_figures"] is True
    assert call["skip_readiness"] is True
    assert call["git_commit_override"] == "abc123"


def test_eval_dispatch_routes_summary_only_with_dependency_bundle(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    observed: dict[str, object] = {}
    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "seeds.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=None,
        episodes_jsonl=tmp_path / "episodes.jsonl",
        summary_json=tmp_path / "summary.json",
        summary_csv=tmp_path / "summary.csv",
        diagnostics_json=tmp_path / "diagnostics.json",
        bootstrap_samples=17,
        bootstrap_seed=23,
    )

    def fake_summary(**kwargs: object) -> None:
        observed["summary"] = kwargs

    run_eval_dispatch(
        parser=argparse.ArgumentParser(),
        args=args,
        validated=EvalValidatedArgs(
            run_label="summary",
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
        ),
        startup=EvalStartup(
            stack=stack,
            config_hash256="c" * 64,
            reported_spec_hash="e" * 64,
            contract=SimpleNamespace(
                simulator={"compatibility_hash": "compat123"},
                spec_hash256="e" * 64,
            ),
        ),
        dependencies=EvalDispatchDependencies(
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            public_demo_stop_rules_fn=None,
            run_public_demo_final_eval_fn=None,
            run_public_demo_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("public demo should not run")
            ),
            run_canonical_eval_pipeline_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("canonical mode should not run")
            ),
            run_summary_only_eval_mode_fn=fake_summary,
            load_eval_game_records_fn="load-records",
            build_matchup_export_fn="build-export",
            build_seat_advantage_diagnostics_fn="seat-diagnostics",
            write_matchup_diagnostics_json_fn="write-diagnostics",
            write_matchup_summary_csv_fn="write-csv",
            write_matchup_summary_json_fn="write-json",
        ),
    )

    call = observed["summary"]
    assert call["stack"] is stack
    assert call["episodes_jsonl"] == tmp_path / "episodes.jsonl"
    assert call["summary_json"] == tmp_path / "summary.json"
    assert call["summary_csv"] == tmp_path / "summary.csv"
    assert call["diagnostics_json"] == tmp_path / "diagnostics.json"
    assert call["bootstrap_samples"] == 17
    assert call["bootstrap_seed"] == 23
    assert call["load_eval_game_records_fn"] == "load-records"
    assert call["build_matchup_export_fn"] == "build-export"
    assert call["build_seat_advantage_diagnostics_fn"] == "seat-diagnostics"
    assert call["write_matchup_diagnostics_json_fn"] == "write-diagnostics"
    assert call["write_matchup_summary_csv_fn"] == "write-csv"
    assert call["write_matchup_summary_json_fn"] == "write-json"
    assert "Verified runtime spec bundle" in capsys.readouterr().out


def test_eval_dispatch_contract_check_only_skips_route_adapters(tmp_path: Path, capsys) -> None:
    from types import SimpleNamespace

    from weiss_rl.workflows.eval_support.eval_dispatch import run_eval_dispatch
    from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
    from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs

    stack = SimpleNamespace(seed_sets={"report_eval": tmp_path / "report_eval.txt", "dev_eval": tmp_path / "dev.txt"})
    args = SimpleNamespace(
        public_demo=False,
        run_dir=None,
        episodes_jsonl=None,
    )

    run_eval_dispatch(
        parser=argparse.ArgumentParser(),
        args=args,
        validated=EvalValidatedArgs(
            run_label="contract_check",
            paired_seed_limit=None,
            stage1_paired_seeds=None,
            max_paired_seeds=None,
        ),
        startup=EvalStartup(
            stack=stack,
            config_hash256="c" * 64,
            reported_spec_hash="e" * 64,
            contract=SimpleNamespace(
                simulator={"compatibility_hash": "compat123"},
                spec_hash256="e" * 64,
            ),
        ),
        dependencies=EvalDispatchDependencies(
            public_demo_spec_bundle_fn=lambda: {"spec_hash": "public_demo"},
            public_demo_stop_rules_fn=None,
            run_public_demo_final_eval_fn=None,
            run_public_demo_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("public demo should not run")
            ),
            run_canonical_eval_pipeline_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("canonical mode should not run")
            ),
            run_summary_only_eval_mode_fn=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("summary mode should not run")
            ),
        ),
    )

    output = capsys.readouterr().out
    assert "Verified runtime spec bundle" in output
    assert "Evaluation contract check complete; no episodes were summarized." in output
    assert "Seed sets: ['dev_eval', 'report_eval']" in output


def test_eval_entrypoint_honors_completed_manifest_policy_selection(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_locked"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.run_summary_path.write_text(
        json.dumps({"kind": "run_summary_v1", "canonical_eval_completed": True}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_locked"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == ["B0 RandomLegal", "policy_locked"]
    assert details["mode"] == "deterministic_v1"
    assert details["status"] == "resolved"
    assert details["policy_count"] == 2
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_entrypoint_ignores_incomplete_manifest_selection_from_canonical_eval_pipeline(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_incomplete"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_stale_only"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
            "resolved_by": "canonical_eval_pipeline_v1",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_entrypoint_ignores_completed_explicit_cli_manifest_selection(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_explicit_cli"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.final_eval_summary_json().parent.mkdir(parents=True, exist_ok=True)
    layout.final_eval_summary_json().write_text("{}\n", encoding="utf-8")
    manifest = {
        "policy_set_selection": ["policy_custom_only"],
        "policy_set_selection_details": {
            "mode": "explicit_cli",
            "status": "resolved",
        },
    }

    policy_ids, details, resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=[],
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert policy_ids == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    assert details["mode"] == "deterministic_v1"
    assert resolved_snapshot_registry == layout.training_snapshots_dir / "registry.json"
    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_manifest_persistence_records_explicit_cli_policy_selection(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    run_dir = tmp_path / "runs" / "eval_manifest_persistence"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "policy_set_selection": ["policy_original"],
        "policy_set_selection_details": {
            "mode": "deterministic_v1",
            "status": "resolved",
        },
    }
    layout.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    eval_script._persist_policy_selection_in_manifest(
        layout=layout,
        manifest=dict(manifest),
        policy_ids=["policy_explicit"],
        selection_details={"mode": "explicit_cli", "policy_count": 1},
    )

    persisted = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert persisted["policy_set_selection"] == ["policy_explicit"]
    assert persisted["policy_set_selection_details"] == {
        "mode": "explicit_cli",
        "policy_count": 1,
        "resolved_by": "canonical_eval_pipeline_v1",
        "status": "resolved",
    }


def test_eval_report_helpers_create_defaults_for_interpolated_runs(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    run_dir = tmp_path / "runs" / "interpolated_eval"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_id256": "ab" * 32,
                "run_id64": "ab" * 8,
                "evaluation_pinning": {"eval_device": "cpu"},
                "seed_derivation": {"base_seed": 7},
                "seed_files": {"report_eval": {"path": "seeds/report.txt", "sha256": "cd" * 32}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_summary = eval_script._load_run_summary_or_default(layout)
    determinism = eval_script._load_determinism_report_or_default(layout)
    environment = eval_script._load_environment_or_default(layout)

    assert run_summary["runtime_mode"] == "interpolated_checkpoint"
    assert run_summary["run_id256"] == "ab" * 32
    assert determinism["device_policy"]["learner"] == "interpolated_checkpoint"
    assert determinism["device_policy"]["evaluation"] == "cpu"
    assert determinism["seed_derivation"] == {"base_seed": 7}
    assert environment["kind"] == "environment_manifest_v1"
    assert environment["run_id256"] == "ab" * 32


def test_eval_pipeline_persists_policy_selection_before_run_final_eval(tmp_path: Path, monkeypatch) -> None:
    import scripts.eval as eval_script

    expected_policy_ids = [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "B2 HeuristicPublic",
        "policy_000400",
        "policy_000100",
        "policy_000200",
        "policy_000300",
        "policy_000150",
        "policy_000250",
        "policy_000350",
    ]
    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_pipeline_persist_before_final_eval"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_id256": "ab" * 32,
                "config_hash256": "cd" * 32,
                "spec_hash256": "ef" * 32,
                "policy_set_selection": [],
                "policy_set_selection_details": {
                    "status": "unresolved",
                    "reason": "selection_pending",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeTensorBoardLogger:
        enabled = False

        def __init__(self, _log_dir: Path) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeContract:
        spec_bundle = {
            "observation": {"obs_len": 512},
            "action": {"action_space_size": 9, "pass_action_id": 8},
        }

    observed: dict[str, dict[str, object]] = {}

    def _fake_run_final_eval(**_kwargs: object) -> dict[str, object]:
        observed["manifest"] = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        raise RuntimeError("stop after manifest check")

    monkeypatch.setattr(eval_script, "TensorBoardLogger", _FakeTensorBoardLogger)
    monkeypatch.setattr(
        eval_script,
        "load_verified_simulator_contract",
        lambda *_args, **_kwargs: _FakeContract(),
    )
    monkeypatch.setattr(eval_script, "resolve_eval_policies", lambda **_kwargs: [])
    monkeypatch.setattr(eval_script, "SimulatorEvalRunner", lambda **_kwargs: object())
    monkeypatch.setattr(eval_script, "run_final_eval", _fake_run_final_eval)

    try:
        eval_script._run_canonical_eval_pipeline(
            parser=eval_script.argparse.ArgumentParser(),
            stack=stack,
            run_dir=run_dir,
            final_eval_dir=None,
            policy_ids=[],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            b1_baseline_run_dir=None,
            bootstrap_samples=8,
            paired_seed_limit=1,
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            skip_metagame=True,
            study_config_path=None,
            skip_figures=True,
            skip_readiness=True,
            git_commit_override="",
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after manifest check"
    else:
        raise AssertionError("expected fake run_final_eval to stop the pipeline")

    persisted = observed["manifest"]
    assert persisted["policy_set_selection"] == expected_policy_ids
    assert persisted["policy_set_selection_details"] == {
        "mode": "deterministic_v1",
        "policy_count": len(expected_policy_ids),
        "resolved_by": "canonical_eval_pipeline_v1",
        "snapshot_registry_path": (layout.training_snapshots_dir / "registry.json").as_posix(),
        "dev_eval_summaries_path": (layout.training_logs_dir / "periodic_dev_eval_summaries.json").as_posix(),
        "final_policy_set_size": 10,
        "status": "resolved",
    }


def test_eval_git_commit_override_does_not_mutate_manifest_payload() -> None:
    import scripts.eval as eval_script

    manifest = {"run_id256": "ab" * 32}

    effective = eval_script._effective_manifest_git_commit(
        manifest=manifest,
        git_commit_override="deadbeef" * 5,
    )

    assert effective == "deadbeef" * 5
    assert "git_commit" not in manifest


def test_train_entrypoint_uses_default_run_dir_when_no_label_override(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="123",
    )

    assert result.returncode == 0, result.stderr
    manifest_path_line = next(line for line in result.stdout.splitlines() if line.startswith("Wrote manifest: "))
    manifest_path = Path(manifest_path_line.removeprefix("Wrote manifest: ").strip())
    assert manifest_path.name == "manifest.json"
    assert manifest_path.parent.name.startswith("run_")
    assert "run_label:              (default)" in result.stdout
    assert f"run_dir_name:           {manifest_path.parent.name}" in result.stdout


def test_train_entrypoint_accepts_deprecated_run_id_alias(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash="123",
        run_id_alias="compat_alias_run",
    )

    assert result.returncode == 0, result.stderr
    assert "deprecated; use --run-label instead" in result.stderr
    assert (tmp_path / "runs" / "compat_alias_run" / "manifest.json").is_file()


def test_eval_entrypoint_honors_explicit_spec_hash_without_reproducibility_config(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _write_manifest_only_stack_config(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash=_mismatched_sha256(spec_bundle_hash(bundle)),
    )

    assert result.returncode != 0
    assert "Spec bundle hash mismatch" in result.stderr


def test_eval_entrypoint_accepts_spec_bundle_sha256(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash=spec_bundle_hash(bundle),
    )

    assert result.returncode == 0, result.stderr
    assert "Verified runtime spec bundle" in result.stdout
    assert "run_label:              (default)" in result.stdout
    assert "computed_run_id64:" not in result.stdout


def test_eval_entrypoint_reports_run_label_without_claiming_run_identity(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        run_label="eval_report_label",
    )

    assert result.returncode == 0, result.stderr
    assert "run_label:              eval_report_label" in result.stdout
    assert "Verified runtime spec bundle" in result.stdout
    assert "computed_run_id64:" not in result.stdout
    assert "computed_run_id256:" not in result.stdout


def test_eval_entrypoint_fails_fast_on_config_hash_mismatch(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    config_hash256 = compute_config_hash256(load_stack_config(stack_config))

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--config-hash", _mismatched_sha256(config_hash256)],
    )

    assert result.returncode != 0
    assert "--config-hash mismatch" in result.stderr


def test_eval_entrypoint_requires_skip_readiness_when_skipping_required_outputs(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        extra_args=["--run-dir", str(tmp_path / "runs" / "candidate"), "--skip-metagame"],
    )

    assert result.returncode != 0
    assert "--skip-metagame or --skip-figures requires --skip-readiness" in result.stderr


def test_eval_entrypoint_exports_summary_json_and_csv(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    diagnostics_json = tmp_path / "diagnostics.json"
    episodes_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "pair_index": 0,
                        "swap_index": 0,
                        "episode_index": 0,
                        "episode_seed": 7,
                        "episode_key": "01" * 32,
                        "episode_key64": 1,
                        "config_hash256": "ab" * 32,
                        "spec_hash256": "cd" * 32,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "champion",
                        "seat1_policy_id": "baseline",
                        "focal_seat": 0,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "pair_index": 0,
                        "swap_index": 1,
                        "episode_index": 1,
                        "episode_seed": 7,
                        "episode_key": "02" * 32,
                        "episode_key64": 2,
                        "config_hash256": "ab" * 32,
                        "spec_hash256": "cd" * 32,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "baseline",
                        "seat1_policy_id": "champion",
                        "focal_seat": 1,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                    },
                    sort_keys=True,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash="",
        extra_args=[
            "--episodes-jsonl",
            str(episodes_path),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
            "--diagnostics-json",
            str(diagnostics_json),
            "--bootstrap-samples",
            "16",
            "--bootstrap-seed",
            "7",
        ],
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_json.read_text(encoding="utf-8"))
    assert payload["stop_reason"] == "decisive"
    assert payload["summary"]["wins"] == 2
    assert diagnostics["seat_results"]["seat0_wins"] == 1
    assert diagnostics["seat_results"]["seat1_wins"] == 1
    assert summary_csv.read_text(encoding="utf-8").splitlines()[0].startswith("focal_policy_id,")


def test_paper_readiness_entrypoint_writes_summary_json(tmp_path: Path) -> None:
    final_eval_dir = tmp_path / "final_eval"
    summary_path = final_eval_dir / "summary.json"
    readiness_json = final_eval_dir / "paper_readiness_summary.json"
    diagnostics_paths = [
        final_eval_dir / "matchups" / "00_b0_randomlegal__vs__00_b0_randomlegal" / "diagnostics.json",
        final_eval_dir / "matchups" / "00_b0_randomlegal__vs__01_policy_000300" / "diagnostics.json",
        final_eval_dir / "matchups" / "01_policy_000300__vs__00_b0_randomlegal" / "diagnostics.json",
        final_eval_dir / "matchups" / "01_policy_000300__vs__01_policy_000300" / "diagnostics.json",
    ]
    for diagnostics_path in diagnostics_paths:
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(
                {
                    "seat_results": {
                        "seat0_wins": 1,
                        "seat1_wins": 1,
                        "draws": 0,
                        "truncations": 0,
                        "engine_errors": 0,
                        "decisive_games": 2,
                        "total_games": 2,
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "policy_ids": ["B0 RandomLegal", "policy_000300"],
                "metadata": {"selection": {"mode": "deterministic_v1"}},
                "matrices": {
                    "games": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[2, 2], [2, 2]]},
                    "truncations": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0, 0], [0, 0]]},
                    "mean": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0.5, 0.0], [0.9, 0.5]]},
                    "ci_low": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0.5, 0.0], [0.88, 0.5]]},
                    "ci_high": {"policy_ids": ["B0 RandomLegal", "policy_000300"], "values": [[0.5, 0.0], [0.95, 0.5]]},
                    "has_payoff_samples": {
                        "policy_ids": ["B0 RandomLegal", "policy_000300"],
                        "values": [[True, True], [True, True]],
                    },
                    "paired_seed_count": {
                        "policy_ids": ["B0 RandomLegal", "policy_000300"],
                        "values": [[1, 1], [2, 1]],
                    },
                    "stop_reason": {
                        "policy_ids": ["B0 RandomLegal", "policy_000300"],
                        "values": [["precision", "precision"], ["precision", "precision"]],
                    },
                },
                "posterior_samples": {
                    "policy_ids": ["B0 RandomLegal", "policy_000300"],
                    "sample_count": 4,
                    "values": [[[], []], [[0.88, 0.9, 0.92, 0.95], []]],
                },
                "matchups": [
                    {
                        "focal_policy_id": "B0 RandomLegal",
                        "opponent_policy_id": "B0 RandomLegal",
                        "diagnostics_path": "matchups/00_b0_randomlegal__vs__00_b0_randomlegal/diagnostics.json",
                    },
                    {
                        "focal_policy_id": "B0 RandomLegal",
                        "opponent_policy_id": "policy_000300",
                        "diagnostics_path": "matchups/00_b0_randomlegal__vs__01_policy_000300/diagnostics.json",
                    },
                    {
                        "focal_policy_id": "policy_000300",
                        "opponent_policy_id": "B0 RandomLegal",
                        "diagnostics_path": "matchups/01_policy_000300__vs__00_b0_randomlegal/diagnostics.json",
                    },
                    {
                        "focal_policy_id": "policy_000300",
                        "opponent_policy_id": "policy_000300",
                        "diagnostics_path": "matchups/01_policy_000300__vs__01_policy_000300/diagnostics.json",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "paper_readiness_check.py"),
            "--final-eval-dir",
            str(final_eval_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["checks"]["baseline_win_rate_vs_b0"]["focal_policy_id"] == "policy_000300"


def test_paper_readiness_entrypoint_requires_explicit_focal_policy_for_ambiguous_multi_policy_artifacts(
    tmp_path: Path,
) -> None:
    final_eval_dir = tmp_path / "final_eval"
    summary_path = final_eval_dir / "summary.json"
    readiness_json = final_eval_dir / "paper_readiness_summary.json"
    policies = ["B0 RandomLegal", "policy_000300", "policy_000400"]
    matchups: list[dict[str, object]] = []

    for focal_index, focal_policy_id in enumerate(policies):
        for opponent_index, opponent_policy_id in enumerate(policies):
            diagnostics_path = (
                final_eval_dir
                / "matchups"
                / f"{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
                / f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
                / "diagnostics.json"
            )
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text(
                json.dumps(
                    {
                        "seat_results": {
                            "seat0_wins": 1,
                            "seat1_wins": 1,
                            "draws": 0,
                            "truncations": 0,
                            "engine_errors": 0,
                            "decisive_games": 2,
                            "total_games": 2,
                        }
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            matchups.append(
                {
                    "focal_policy_id": focal_policy_id,
                    "opponent_policy_id": opponent_policy_id,
                    "focal_policy_index": focal_index,
                    "opponent_policy_index": opponent_index,
                    "diagnostics_path": (
                        f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__/"
                        f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}/diagnostics.json"
                    ),
                }
            )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "policy_ids": policies,
                "metadata": {"selection": {"mode": "deterministic_v1"}},
                "matrices": {
                    "games": {"policy_ids": policies, "values": [[2, 2, 2], [2, 2, 2], [2, 2, 2]]},
                    "truncations": {"policy_ids": policies, "values": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
                    "mean": {
                        "policy_ids": policies,
                        "values": [[0.5, 0.0, 0.0], [0.9, 0.5, 0.49], [0.94, 0.51, 0.5]],
                    },
                    "ci_low": {
                        "policy_ids": policies,
                        "values": [[0.5, 0.0, 0.0], [0.88, 0.5, 0.45], [0.9, 0.5, 0.5]],
                    },
                    "ci_high": {
                        "policy_ids": policies,
                        "values": [[0.5, 0.0, 0.0], [0.95, 0.5, 0.53], [0.97, 0.54, 0.5]],
                    },
                    "has_payoff_samples": {
                        "policy_ids": policies,
                        "values": [[True, True, True], [True, True, True], [True, True, True]],
                    },
                    "paired_seed_count": {
                        "policy_ids": policies,
                        "values": [[1, 1, 1], [2, 1, 1], [2, 1, 1]],
                    },
                    "stop_reason": {
                        "policy_ids": policies,
                        "values": [
                            ["precision", "precision", "precision"],
                            ["precision", "precision", "precision"],
                            ["precision", "precision", "precision"],
                        ],
                    },
                },
                "posterior_samples": {
                    "policy_ids": policies,
                    "sample_count": 4,
                    "values": [
                        [[], [], []],
                        [[0.88, 0.9, 0.92, 0.95], [], [0.45, 0.48, 0.5, 0.53]],
                        [[0.9, 0.93, 0.95, 0.97], [0.5, 0.51, 0.52, 0.54], []],
                    ],
                },
                "matchups": matchups,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "paper_readiness_check.py"),
            "--final-eval-dir",
            str(final_eval_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "pass --focal-policy-id" in result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["checks"]["baseline_win_rate_vs_b0"]["reason"] == "ambiguous_non_baseline_focal_policy"
    assert payload["checks"]["baseline_win_rate_vs_b0"]["eligible_non_baseline_policy_ids"] == [
        "policy_000300",
        "policy_000400",
    ]


def _write_paper_readiness_run_dir_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_ready"
    policies = ["B0 RandomLegal", "policy_000300"]
    manifest = {
        "run_id256": "ab" * 32,
        "run_id64": "0123456789abcdef",
        "start_nonce": 7,
        "git_commit": "deadbeef" * 5,
        "git_dirty": False,
        "spec_hash256": "cd" * 32,
        "config_hash256": "ef" * 32,
        "simulator": {"version": "0.7.0", "compatibility_hash": "feedfacecafebeef"},
        "spec_bundle": {"version": 1, "cards": []},
        "config_canonical": {"stack": {"name": "synthetic"}},
        "seed_files": {"final_eval": {"path": "configs/seeds/report_eval_seeds.txt", "sha256": "12" * 32}},
        "hardware": {"platform": "test", "cpu": "synthetic"},
        "evaluation_pinning": {"eval_sampling_algorithm": "pinned_cdf_pcg_v1"},
        "policy_set_selection": list(policies),
        "policy_set_selection_details": {"mode": "deterministic_v1"},
    }
    for path, payload in (
        (run_dir / "manifest.json", manifest),
        (run_dir / "spec_bundle.json", manifest["spec_bundle"]),
        (run_dir / "config_canonical.json", manifest["config_canonical"]),
        (
            run_dir / "environment.json",
            {
                "kind": "environment_manifest_v1",
                "artifact_schema_version": "run_artifacts_v2",
                "run_id256": manifest["run_id256"],
                "run_id64": manifest["run_id64"],
            },
        ),
        (
            run_dir / "run_summary.json",
            {
                "kind": "run_summary_v1",
                "artifact_schema_version": "run_artifacts_v2",
                "runtime_mode": "train_ordered",
                "policy_set_selection_mode": "deterministic_v1",
            },
        ),
        (
            run_dir / "determinism_report.json",
            {
                "kind": "determinism_report_v1",
                "artifact_schema_version": "run_artifacts_v2",
                "policy_selection_mode": "deterministic_v1",
                "replay_verification": {
                    "path": "eval/diagnostics/replay_verification.json",
                    "status": "pending",
                },
            },
        ),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "spec_hash256.txt").write_text(str(manifest["spec_hash256"]) + "\n", encoding="utf-8")
    (run_dir / "config_hash256.txt").write_text(str(manifest["config_hash256"]) + "\n", encoding="utf-8")
    (run_dir / "training" / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "training" / "logs" / "training_metrics.jsonl").write_text(
        json.dumps({"loss": 0.8, "policy_version": 1, "update_count": 1}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final_eval_dir = run_dir / "eval" / "final_eval"
    canonical_matchups = [
        (0, 0, policies[0], policies[0]),
        (0, 1, policies[0], policies[1]),
        (1, 1, policies[1], policies[1]),
    ]
    matchups: list[dict[str, object]] = []
    matchup_rows = [
        "focal_policy_id,opponent_policy_id,matchup_dir,paired_seed_count,observed_paired_seed_count,excluded_paired_seed_count,has_payoff_samples,stop_reason"
    ]
    for focal_index, opponent_index, focal_policy_id, opponent_policy_id in canonical_matchups:
        prefix = (
            f"matchups/{focal_index:02d}_{focal_policy_id.lower().replace(' ', '_')}__vs__"
            f"{opponent_index:02d}_{opponent_policy_id.lower().replace(' ', '_')}"
        )
        matchup_dir = final_eval_dir / prefix
        matchup_dir.mkdir(parents=True, exist_ok=True)
        (matchup_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_seed": 11}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (matchup_dir / "matchup_summary.json").write_text(
            json.dumps(
                {
                    "has_payoff_samples": True,
                    "observed_paired_seeds": 2,
                    "paired_seeds": 2,
                    "stop_reason": "precision",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (matchup_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "seat_results": {
                        "seat0_wins": 1,
                        "seat1_wins": 1,
                        "draws": 0,
                        "truncations": 0,
                        "engine_errors": 0,
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (matchup_dir / "posterior_samples.json").write_text(
            json.dumps(
                {"sample_count": 4, "samples": [0.88, 0.91, 0.93, 0.95]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        matchups.append(
            {
                "focal_policy_id": focal_policy_id,
                "opponent_policy_id": opponent_policy_id,
                "focal_policy_index": focal_index,
                "opponent_policy_index": opponent_index,
                "matchup_dir": prefix,
                "episodes_path": prefix + "/episodes.jsonl",
                "summary_path": prefix + "/matchup_summary.json",
                "diagnostics_path": prefix + "/diagnostics.json",
                "posterior_samples_path": prefix + "/posterior_samples.json",
            }
        )
        matchup_rows.append(",".join((focal_policy_id, opponent_policy_id, prefix, "2", "2", "0", "True", "precision")))

    (final_eval_dir / "policy_set.json").write_text(
        json.dumps({"policy_ids": policies}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (final_eval_dir / "posterior_samples.json").write_text(
        json.dumps(
            {
                "policy_ids": policies,
                "sample_count": 4,
                "values": [[[], []], [[0.88, 0.91, 0.93, 0.95], []]],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (final_eval_dir / "matchups.csv").write_text("\n".join(matchup_rows) + "\n", encoding="utf-8")
    (final_eval_dir / "matrices").mkdir(parents=True, exist_ok=True)
    (final_eval_dir / "matrices" / "mean.csv").write_text(
        "focal_policy_id,B0 RandomLegal,policy_000300\nB0 RandomLegal,0.5,0.0\npolicy_000300,0.9,0.5\n",
        encoding="utf-8",
    )
    (final_eval_dir / "summary.json").write_text(
        json.dumps(
            {
                "policy_ids": policies,
                "metadata": {"selection": {"mode": "deterministic_v1"}},
                "matrices": {
                    "games": {"policy_ids": policies, "values": [[2, 2], [2, 2]]},
                    "truncations": {"policy_ids": policies, "values": [[0, 0], [0, 0]]},
                    "mean": {"policy_ids": policies, "values": [[0.5, 0.0], [0.9, 0.5]]},
                    "ci_low": {"policy_ids": policies, "values": [[0.5, 0.0], [0.88, 0.5]]},
                    "ci_high": {"policy_ids": policies, "values": [[0.5, 0.0], [0.95, 0.5]]},
                    "has_payoff_samples": {"policy_ids": policies, "values": [[True, True], [True, True]]},
                    "paired_seed_count": {"policy_ids": policies, "values": [[1, 1], [2, 1]]},
                    "stop_reason": {
                        "policy_ids": policies,
                        "values": [["precision", "precision"], ["precision", "precision"]],
                    },
                },
                "posterior_samples": {
                    "policy_ids": policies,
                    "sample_count": 4,
                    "values": [[[], []], [[0.88, 0.91, 0.93, 0.95], []]],
                },
                "matchups": matchups,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval" / "diagnostics").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "diagnostics" / "seat_bias.json").write_text(
        json.dumps(
            {
                "global": {"seat0_win_rate": 0.5, "ci_low": 0.4, "ci_high": 0.6, "decisive_games": 6},
                "matchups": [
                    {
                        "policy_a": "B0 RandomLegal",
                        "policy_b": "policy_000300",
                        "seat0_win_rate": 0.5,
                        "seat1_win_rate": 0.5,
                        "decisive_games": 2,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "eval" / "diagnostics" / "truncation_heatmap_data.csv").write_text(
        ",B0 RandomLegal,policy_000300\nB0 RandomLegal,0.0,0.0\npolicy_000300,0.0,0.0\n",
        encoding="utf-8",
    )
    (run_dir / "eval" / "diagnostics" / "replay_verification.json").write_text(
        json.dumps({"status": "ok"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (final_eval_dir / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "kind": "final_eval_artifact_hashes_v1",
                "artifacts": {"eval/final_eval/summary.json": "ab" * 32},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metagame_dir = run_dir / "eval" / "metagame"
    metagame_dir.mkdir(parents=True, exist_ok=True)
    (metagame_dir / "summary.json").write_text(
        json.dumps(
            {"policy_ids": policies, "cases": {case_id: {} for case_id in ("S0", "S1", "S2")}},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for case_id in ("S0", "S1", "S2"):
        case_dir = metagame_dir / case_id
        (case_dir / "payoff").mkdir(parents=True, exist_ok=True)
        (case_dir / "nash").mkdir(parents=True, exist_ok=True)
        (case_dir / "alpharank").mkdir(parents=True, exist_ok=True)
        (case_dir / "summary.json").write_text(
            json.dumps({"case_id": case_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (case_dir / "payoff" / "matchups.csv").write_text(
            "focal_policy_id,opponent_policy_id,p_mean\nB0 RandomLegal,policy_000300,0.1\n",
            encoding="utf-8",
        )
        (case_dir / "nash" / "mixture_mean.csv").write_text(
            "policy_id,mean_mixture\nB0 RandomLegal,0.5\npolicy_000300,0.5\n",
            encoding="utf-8",
        )
        (case_dir / "alpharank" / "stationary_mean.csv").write_text(
            "policy_id,mean_stationary_mass\nB0 RandomLegal,0.5\npolicy_000300,0.5\n",
            encoding="utf-8",
        )
    (run_dir / "figures" / "paper").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures" / "paper" / "fig_matchup_heatmap.pdf").write_text("pdf\n", encoding="utf-8")
    (run_dir / "figures" / "paper" / "fig_matchup_heatmap.png").write_text("png\n", encoding="utf-8")
    return run_dir


def test_paper_readiness_entrypoint_accepts_run_dir_and_writes_run_summary(tmp_path: Path) -> None:
    run_dir = _write_paper_readiness_run_dir_fixture(tmp_path)
    readiness_json = run_dir / "paper_readiness_summary.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "paper_readiness_check.py"),
            "--run-dir",
            str(run_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(readiness_json.read_text(encoding="utf-8"))
    assert payload["scope"] == "run_dir"
    assert payload["passed"] is True
    assert payload["run_directory_audit"]["passed"] is True
    assert payload["manifest_contract"]["passed"] is True
    assert payload["final_eval_artifact_contract"]["passed"] is True


def test_train_entrypoint_runs_periodic_dev_eval_and_handles_empty_ids_pass_fallback(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(
        tmp_path,
        spec_hash=123,
        pass_action_id=3,
        empty_eval_legal_row=True,
    )
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="periodic_dev_eval_run",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--checkpoint-interval-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "Periodic dev eval: update=1 opponent=b0_randomlegal" in result.stdout

    eval_root = tmp_path / "runs" / "periodic_dev_eval_run" / "eval" / "dev_eval" / "update_1"
    seed_usage = json.loads((eval_root / "b0_randomlegal" / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    diagnostics_payload = json.loads((eval_root / "b0_randomlegal" / "diagnostics.json").read_text(encoding="utf-8"))
    episodes_lines = (eval_root / "b0_randomlegal" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()

    assert seed_usage["seed_file"]["path"] == "configs/seeds/dev_eval_seeds.txt"
    assert seed_usage["paired_seed_count"] == 1
    assert seed_usage["paired_seeds"] == [7]
    assert seed_usage["focal_policy"]["update_count"] == 1
    assert seed_usage["focal_policy"]["policy_version"] == 1
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert len(episodes_lines) == 2
    assert summary_payload["summary"]["games"] == 2
    assert summary_payload["evaluation_context"] == {
        "artifact_scope": "periodic_dev_eval",
        "update_count": 1,
        "policy_version": 1,
        "checkpoint_path": "training/checkpoints/checkpoint_1.pt",
        "seed_usage_path": "eval/dev_eval/update_1/b0_randomlegal/seed_usage.json",
        "anchor_display_name": "B0 RandomLegal",
    }
    assert diagnostics_payload["seat_results"]["seat0_wins"] == 2
    assert diagnostics_payload["seat_results"]["seat1_wins"] == 0
    assert (eval_root / "b0_randomlegal" / "matchup_summary.csv").is_file()


def test_train_entrypoint_periodic_dev_eval_writes_exact_current_checkpoint(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="periodic_dev_eval_checkpoint_traceability",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--checkpoint-interval-updates",
            "2",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr

    run_root = tmp_path / "runs" / "periodic_dev_eval_checkpoint_traceability"
    eval_root = run_root / "eval" / "dev_eval" / "update_1"
    checkpoint_path = run_root / "training" / "checkpoints" / "checkpoint_1.pt"
    seed_usage = json.loads((eval_root / "b0_randomlegal" / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint_path.is_file()
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert summary_payload["evaluation_context"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert checkpoint_payload["update_count"] == 1


def test_train_entrypoint_uses_configured_checkpoint_interval_by_default(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)

    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label="checkpoint_default_from_config",
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / "checkpoint_default_from_config"
    registry = json.loads((run_root / "training" / "snapshots" / "registry.json").read_text(encoding="utf-8"))
    assert [snapshot["policy_id"] for snapshot in registry["snapshots"]] == ["b1_noleague_baseline"]
    assert not (run_root / "eval" / "promotion_gate" / "update_1").exists()


def test_train_entrypoint_public_demo_accepts_profile_timers_flag(tmp_path: Path) -> None:
    stack_config = _copy_repo_configs(tmp_path)
    run_label = "toy_public_demo_profile_timers"
    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        run_label=run_label,
        extra_args=["--public-demo", "--profile-timers"],
    )

    assert result.returncode == 0, result.stderr
    assert "Staged public-demo toy catalog and policy bundle" in result.stdout
    run_summary = json.loads((tmp_path / "runs" / run_label / "run_summary.json").read_text(encoding="utf-8"))
    training_controls = run_summary["training_controls"]
    assert training_controls["profile_timers"] is True
    assert training_controls["torch_profiler"] is False


def test_train_entrypoint_profile_timers_does_not_emit_torch_profiler_trace(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)
    run_label = "profile_timers_no_trace"
    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label=run_label,
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
            "--profile-timers",
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / run_label
    assert not (run_root / "profiling" / "torch_profiler" / "trace.json").exists()
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    determinism = json.loads((run_root / "determinism_report.json").read_text(encoding="utf-8"))
    assert run_summary["training_controls"]["profile_timers"] is True
    assert run_summary["training_controls"]["torch_profiler"] is False
    assert determinism["training_controls"]["profile_timers"] is True
    assert determinism["training_controls"]["torch_profiler"] is False


def test_train_entrypoint_emits_torch_profiler_trace(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    b1_baseline_run_dir = _write_b1_baseline_run_fixture(tmp_path, stack_config=stack_config)
    run_label = "torch_profiler_trace"
    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=str(bundle["spec_hash"]),
        run_label=run_label,
        extra_args=[
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--unroll-length",
            "1",
            "--max-updates",
            "1",
            "--b1-baseline-run-dir",
            str(b1_baseline_run_dir),
            "--torch-profiler",
        ],
    )

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "runs" / run_label
    assert (run_root / "profiling" / "torch_profiler" / "trace.json").exists()
    run_summary = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    determinism = json.loads((run_root / "determinism_report.json").read_text(encoding="utf-8"))
    assert run_summary["training_controls"]["profile_timers"] is False
    assert run_summary["training_controls"]["torch_profiler"] is True
    assert determinism["training_controls"]["profile_timers"] is False
    assert determinism["training_controls"]["torch_profiler"] is True


def _run_public_demo_train(
    tmp_path: Path,
    *,
    run_label: str = "toy_public_demo",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    stack_config = _copy_repo_configs(tmp_path)
    result = _run_entrypoint(
        tmp_path,
        script_name="train.py",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        run_label=run_label,
        extra_args=["--public-demo"],
    )
    return result, tmp_path / "runs" / run_label


def test_train_entrypoint_public_demo_stages_public_safe_catalog_without_weiss_sim(tmp_path: Path) -> None:
    result, run_dir = _run_public_demo_train(tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((run_dir / "public_demo" / "catalog.json").read_text(encoding="utf-8"))
    policies = json.loads((run_dir / "public_demo" / "policy_manifest.json").read_text(encoding="utf-8"))
    scalars_lines = (run_dir / "training" / "logs" / "scalars.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["simulator"]["runtime"] == "public_demo"
    assert manifest["simulator"]["public_safe"] is True
    assert manifest["spec_bundle"]["action"]["action_space_size"] == 9
    assert catalog["public_safe"] is True
    assert len(catalog["card_pool"]) == 12
    assert len(catalog["decks"]) == 3
    assert policies["policy_ids"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "toy_policy_000100",
        "toy_policy_000200",
    ]
    assert len(scalars_lines) == 1
    assert "Loaded synthetic public-demo spec bundle" in result.stdout
    assert "Verified runtime spec bundle" not in result.stdout
    assert "Staged public-demo toy catalog and policy bundle" in result.stdout
    assert "demo-only" in result.stdout


def test_eval_entrypoint_public_demo_generates_demo_only_final_eval_artifacts(tmp_path: Path) -> None:
    train_result, run_dir = _run_public_demo_train(tmp_path, run_label="toy_public_demo_eval")
    assert train_result.returncode == 0, train_result.stderr
    stack_config = tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"

    result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        extra_args=[
            "--public-demo",
            "--run-dir",
            str(run_dir),
            "--public-demo-paired-seeds",
            "4",
            "--public-demo-bootstrap-samples",
            "8",
        ],
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((run_dir / "eval" / "final_eval" / "summary.json").read_text(encoding="utf-8"))
    metadata = summary["metadata"]

    assert summary["policy_ids"] == [
        "B0 RandomLegal",
        "B1 NoLeague baseline",
        "toy_policy_000100",
        "toy_policy_000200",
    ]
    assert metadata["demo_only"] is True
    assert metadata["public_safe"] is True
    assert metadata["catalog_path"] == "public_demo/catalog.json"
    assert metadata["policy_manifest_path"] == "public_demo/policy_manifest.json"
    assert metadata["paired_seed_budget"] == 4
    assert metadata["recommended_focal_policy_id"] == "toy_policy_000200"
    assert len(summary["matchups"]) == 10
    assert "Public-demo final_eval summary JSON" in result.stdout


def test_make_figures_entrypoint_public_demo_writes_clearly_labeled_bundle(tmp_path: Path) -> None:
    train_result, run_dir = _run_public_demo_train(tmp_path, run_label="toy_public_demo_figures")
    assert train_result.returncode == 0, train_result.stderr
    stack_config = tmp_path / "configs" / "presets" / "typed_thesis_locked.yaml"
    eval_result = _run_entrypoint(
        tmp_path,
        script_name="eval.py",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        extra_args=[
            "--public-demo",
            "--run-dir",
            str(run_dir),
            "--public-demo-paired-seeds",
            "4",
            "--public-demo-bootstrap-samples",
            "8",
        ],
    )
    assert eval_result.returncode == 0, eval_result.stderr

    figures_dir = run_dir / "figures"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "python")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "python" / "scripts" / "make_figures.py"),
            "--public-demo",
            "--final-eval-dir",
            str(run_dir / "eval" / "final_eval"),
            "--out-dir",
            str(figures_dir),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    placeholder_path = figures_dir / "toy_demo_placeholder.txt"
    manifest_path = figures_dir / "toy_demo_manifest.json"
    assert placeholder_path.is_file()
    assert manifest_path.is_file()
    placeholder_text = placeholder_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "toy_public_demo_placeholder_figure" in placeholder_text
    assert manifest["demo_only"] is True
    assert manifest["public_safe"] is True
    assert "Wrote public-demo placeholder figure bundle" in result.stdout


def test_figure_mode_helpers_preserve_messages_and_default_paths(tmp_path: Path) -> None:
    from weiss_rl.workflows.figures.figure_modes import (
        run_paper_figure_mode,
        run_placeholder_figure_mode,
        run_public_demo_figure_mode,
    )

    observed: dict[str, object] = {}
    final_eval_dir = tmp_path / "runs" / "demo" / "eval" / "final_eval"

    def fake_public_demo_figures(*, final_eval_dir: Path, out_dir: Path) -> dict[str, Path]:
        observed["public_demo"] = (final_eval_dir, out_dir)
        return {"manifest": out_dir / "toy_demo_manifest.json"}

    def fake_placeholder(out: Path) -> None:
        observed["placeholder"] = out

    def fake_paper(run_dir: Path, *, formats: tuple[str, ...], fig_id: str | None) -> tuple[Path, ...]:
        observed["paper"] = (run_dir, formats, fig_id)
        return (run_dir / "figures" / "paper" / "seat_bias.pdf",)

    public_message = run_public_demo_figure_mode(
        final_eval_dir=final_eval_dir,
        out_dir=None,
        render_public_demo_figures_fn=fake_public_demo_figures,
    )
    placeholder_message = run_placeholder_figure_mode(
        out=tmp_path / "placeholder.txt",
        render_placeholder_figure_fn=fake_placeholder,
    )
    paper_message = run_paper_figure_mode(
        run_dir=tmp_path / "runs" / "main",
        formats=(),
        fig_id="seat_bias",
        render_paper_figures_fn=fake_paper,
    )

    assert observed["public_demo"] == (final_eval_dir, tmp_path / "runs" / "demo" / "figures")
    assert public_message.endswith("runs/demo/figures/toy_demo_manifest.json") or public_message.endswith(
        "runs\\demo\\figures\\toy_demo_manifest.json"
    )
    assert observed["placeholder"] == tmp_path / "placeholder.txt"
    assert placeholder_message == f"Wrote placeholder figure: {tmp_path / 'placeholder.txt'}"
    assert observed["paper"] == (tmp_path / "runs" / "main", ("pdf", "png"), "seat_bias")
    assert (
        paper_message == f"Wrote 1 files for fig-id 'seat_bias' to {tmp_path / 'runs' / 'main' / 'figures' / 'paper'}"
    )


def test_make_figures_script_shim_reexports_figure_mode_helpers() -> None:
    script_path = REPO_ROOT / "python" / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("test_make_figures_shim", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from weiss_rl.workflows import figure_modes, figures_entrypoint

    assert module.main is figures_entrypoint.main
    assert module.run_public_demo_figure_mode is figure_modes.run_public_demo_figure_mode
    assert module.run_placeholder_figure_mode is figure_modes.run_placeholder_figure_mode
    assert module.run_paper_figure_mode is figure_modes.run_paper_figure_mode
