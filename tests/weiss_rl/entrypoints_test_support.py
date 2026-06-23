from __future__ import annotations

import argparse
import hashlib
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

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    module_name: str,
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

    command = [sys.executable, "-m", module_name, "--stack-config", str(stack_config)]
    if spec_hash:
        command.extend(["--spec-hash", spec_hash])
    if run_label:
        command.extend(["--run-label", run_label])
    if run_id_alias:
        command.extend(["--run-id", run_id_alias])
    if extra_args:
        command.extend(extra_args)

    return subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)


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


def _run_public_demo_train(
    tmp_path: Path,
    *,
    run_label: str = "toy_public_demo",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    stack_config = _copy_repo_configs(tmp_path)
    result = _run_entrypoint(
        tmp_path,
        module_name="weiss_rl.training.train_entrypoint",
        stack_config=stack_config,
        spec_hash=public_demo_spec_hash256(),
        run_label=run_label,
        extra_args=["--public-demo"],
    )
    return result, tmp_path / "runs" / run_label


__all__ = (
    "Any",
    "argparse",
    "ArtifactLayout",
    "compute_config_hash256",
    "hashlib",
    "json",
    "load_stack_config",
    "os",
    "Path",
    "PolicyValueModel",
    "public_demo_spec_hash256",
    "pytest",
    "REPO_ROOT",
    "shutil",
    "SnapshotRegistry",
    "spec_bundle_hash",
    "subprocess",
    "sys",
    "torch",
    "_copy_repo_configs",
    "_mismatched_sha256",
    "_patch_periodic_dev_eval_config",
    "_run_entrypoint",
    "_run_public_demo_train",
    "_typed_observation_spec",
    "_write_b1_baseline_run_fixture",
    "_write_eval_only_stack_config",
    "_write_manifest_only_stack_config",
    "_write_paper_readiness_run_dir_fixture",
    "_write_policy_set_inputs",
    "_write_runtime_weiss_sim",
    "_write_stub_weiss_sim",
)
