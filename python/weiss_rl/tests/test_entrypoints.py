from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.league.registry import SnapshotRegistry
from weiss_rl.model import PolicyValueModel
from weiss_rl.spec import spec_bundle_hash
from weiss_rl.toy_public_demo import public_demo_spec_hash256

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mismatched_sha256(value: str) -> str:
    return ("0" if value[0] != "0" else "1") + value[1:]


def _typed_observation_spec() -> dict[str, object]:
    return {
        "obs_encoding_version": 2,
        "dtype": "i32",
        "obs_len": 512,
        "self_first": True,
        "header_fields": [
            {"name": "active_player", "index": 0},
            {"name": "phase", "index": 1},
            {"name": "decision_kind", "index": 2},
            {"name": "choice_page_start", "index": 3},
            {"name": "choice_total", "index": 4},
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
            "constants": [
                ["MAX_HAND", 5],
                ["MAX_STAGE", 5],
                ["ATTACK_SLOT_COUNT", 3],
            ],
            "families": [
                {"name": "pass", "base": pass_action_id, "count": 1},
            ],
            "attack_type_encoding": [
                ["direct", 0],
            ],
        },
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
                f"    return {bundle!r}",
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
                f"_BUNDLE = {bundle!r}",
                f"PASS_ACTION_ID = {pass_action_id}",
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
                "def fast(**kwargs):",
                "    return _make_pool(**kwargs)",
                "",
                "def inspect(**kwargs):",
                "    return _make_pool(**kwargs)",
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
                "    out.legal_ids[0] = PASS_ACTION_ID",
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
                "rl = _Rl()",
            )
        ),
        encoding="utf-8",
    )
    return bundle


def _patch_periodic_dev_eval_config(tmp_path: Path) -> None:
    preset_path = tmp_path / "configs" / "thesis_locked.yaml"
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
    return tmp_path / "configs" / "thesis_locked.yaml"


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
        observation_spec=_typed_observation_spec(),  # type: ignore[arg-type]
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
    _copy_repo_configs(tmp_path)
    stack_config = tmp_path / "configs" / "stack_smoke.yaml"

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


def test_eval_entrypoint_requires_explicit_opt_in_to_reuse_completed_manifest_policy_selection(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_locked_no_reuse"
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
        allow_completed_manifest_policy_selection=False,
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


def test_eval_entrypoint_rejects_summary_only_manifest_reuse(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_partial"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    layout.final_eval_summary_json().parent.mkdir(parents=True, exist_ok=True)
    layout.final_eval_summary_json().write_text("{}\n", encoding="utf-8")
    manifest = {
        "policy_set_selection": ["B0 RandomLegal", "policy_partial_only"],
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

    assert "policy_partial_only" not in policy_ids
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


def test_eval_entrypoint_prefers_current_periodic_dev_eval_summaries(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_current_dev_eval"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")
    (layout.training_logs_dir / "dev_eval_summaries.json").write_text("[]\n", encoding="utf-8")

    _policy_ids, _details, _resolved_snapshot_registry, resolved_dev_eval = eval_script._resolve_policy_ids_for_run(
        policy_ids=["B0 RandomLegal"],
        stack=stack,
        manifest={},
        layout=layout,
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
    )

    assert resolved_dev_eval == layout.training_logs_dir / "periodic_dev_eval_summaries.json"


def test_eval_entrypoint_rejects_missing_explicit_dev_eval_summaries(tmp_path: Path) -> None:
    import scripts.eval as eval_script

    _copy_repo_configs(tmp_path)
    stack_config = _write_eval_only_stack_config(tmp_path)
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "runs" / "eval_policy_selection_missing_explicit"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.training_snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.training_logs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_registry_path, dev_eval_summaries_path = _write_policy_set_inputs(tmp_path)
    shutil.copy2(snapshot_registry_path, layout.training_snapshots_dir / "registry.json")
    shutil.copy2(dev_eval_summaries_path, layout.training_logs_dir / "periodic_dev_eval_summaries.json")

    with pytest.raises(FileNotFoundError, match="Explicit dev eval summaries path"):
        eval_script._resolve_policy_ids_for_run(
            policy_ids=[],
            stack=stack,
            manifest={},
            layout=layout,
            snapshot_registry_path=None,
            dev_eval_summaries_path=tmp_path / "missing_dev_eval_summaries.json",
        )


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
        observed["metadata"] = cast(dict[str, object], _kwargs["metadata"])
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
            stack_config_path=stack_config.resolve(),
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
            parallel_workers=1,
            parallel_worker_devices=(),
            reuse_manifest_policy_selection=False,
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
    assert observed["metadata"]["selection"] == {
        "mode": "deterministic_v1",
        "policy_count": len(expected_policy_ids),
        "snapshot_registry_path": (layout.training_snapshots_dir / "registry.json").as_posix(),
        "dev_eval_summaries_path": (layout.training_logs_dir / "periodic_dev_eval_summaries.json").as_posix(),
        "final_policy_set_size": 10,
    }


def test_eval_pipeline_uses_parallel_helper_when_requested(tmp_path: Path, monkeypatch) -> None:
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
    run_dir = tmp_path / "runs" / "eval_pipeline_parallel"
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

    observed: dict[str, object] = {}

    def _fake_parallel_final_eval(**kwargs: object) -> dict[str, object]:
        observed["policy_ids"] = kwargs["policy_ids"]
        observed["parallel_workers"] = kwargs["parallel_workers"]
        observed["parallel_worker_devices"] = kwargs["parallel_worker_devices"]
        observed["manifest"] = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        raise RuntimeError("stop after parallel helper check")

    monkeypatch.setattr(eval_script, "TensorBoardLogger", _FakeTensorBoardLogger)
    monkeypatch.setattr(
        eval_script,
        "load_verified_simulator_contract",
        lambda *_args, **_kwargs: _FakeContract(),
    )
    monkeypatch.setattr(eval_script, "_run_parallel_final_eval", _fake_parallel_final_eval)

    try:
        eval_script._run_canonical_eval_pipeline(
            parser=eval_script.argparse.ArgumentParser(),
            stack_config_path=stack_config.resolve(),
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
            parallel_workers=2,
            parallel_worker_devices=("cuda:1", "cuda:2"),
            reuse_manifest_policy_selection=False,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after parallel helper check"
    else:
        raise AssertionError("expected fake parallel helper to stop the pipeline")

    assert observed["policy_ids"] == expected_policy_ids
    assert observed["parallel_workers"] == 2
    assert observed["parallel_worker_devices"] == ("cuda:1", "cuda:2")
    persisted = cast(dict[str, object], observed["manifest"])
    assert persisted["policy_set_selection"] == expected_policy_ids


def test_parallel_eval_worker_devices_treat_auto_as_cuda_auto(monkeypatch) -> None:
    import scripts.eval as eval_script

    monkeypatch.setattr(eval_script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(eval_script.torch.cuda, "device_count", lambda: 2)

    assert eval_script._resolved_parallel_worker_devices(
        parallel_workers=5,
        explicit_worker_devices=(),
        eval_device="auto",
    ) == ("cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0")


def test_parallel_eval_worker_devices_reject_invalid_explicit_cuda(monkeypatch) -> None:
    import scripts.eval as eval_script

    monkeypatch.setattr(eval_script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(eval_script.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="only 1 CUDA device"):
        eval_script._resolved_parallel_worker_devices(
            parallel_workers=1,
            explicit_worker_devices=("cuda:2",),
            eval_device="cpu",
        )


def test_parallel_final_eval_falls_back_to_serial_for_single_matchup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import scripts.eval as eval_script

    stack_config = REPO_ROOT / "configs" / "local.yaml"
    stack = load_stack_config(stack_config)
    run_dir = tmp_path / "run"
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.ensure_directories()
    observed: dict[str, object] = {}

    def _fake_worker(**kwargs: object) -> list[dict[str, object]]:
        observed["worker_policy_ids"] = kwargs["policy_ids"]
        observed["worker_matchup_specs"] = kwargs["matchup_specs"]
        observed["worker_eval_device"] = kwargs["eval_device"]
        return [{"focal_index": 0, "opponent_index": 1, "summary": {"games": 1}}]

    def _fake_finalize_final_eval(**kwargs: object) -> dict[str, object]:
        observed["selection_payload"] = kwargs["selection_payload"]
        observed["metadata"] = kwargs["metadata"]
        observed["matchup_results"] = kwargs["matchup_results"]
        return {"status": "ok"}

    monkeypatch.setattr(
        eval_script,
        "build_final_eval_matchups",
        lambda *, policy_ids: [
            {
                "focal_index": 0,
                "opponent_index": 1,
                "focal_policy_id": str(policy_ids[0]),
                "opponent_policy_id": str(policy_ids[1]),
            }
        ],
    )
    monkeypatch.setattr(eval_script, "_run_final_eval_matchup_worker", _fake_worker)
    monkeypatch.setattr(eval_script, "finalize_final_eval", _fake_finalize_final_eval)

    payload = eval_script._run_parallel_final_eval(
        stack_config_path=stack_config.resolve(),
        stack=stack,
        run_dir=run_dir,
        layout=layout,
        policy_ids=("policy_a", "policy_b"),
        paired_seeds=(1,),
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        stop_rules=stack.config.evaluation.stop_rules,
        run_id256="ab" * 32,
        config_hash256="cd" * 32,
        spec_hash256="ef" * 32,
        scheme=cast(object, stack.config.evaluation.final_policy_set_selection.folding),
        sample_count=8,
        snapshot_registry_path=None,
        b1_baseline_run_dir=None,
        metadata={},
        seed_file_path=None,
        parallel_workers=4,
        parallel_worker_devices=("cuda:1", "cuda:2"),
    )

    assert payload == {"status": "ok"}
    assert observed["worker_policy_ids"] == ["policy_a", "policy_b"]
    assert len(cast(list[dict[str, object]], observed["worker_matchup_specs"])) == 1
    assert observed["worker_eval_device"] == str(stack.config.evaluation.eval_device)
    assert observed["selection_payload"] == {"mode": "explicit", "policy_count": 2}
    metadata = cast(dict[str, object], observed["metadata"])
    parallel_eval = cast(dict[str, object], cast(dict[str, object], metadata["pipeline"])["parallel_eval"])
    assert parallel_eval["enabled"] is False
    assert parallel_eval["fallback_reason"] == "single_matchup"
    assert observed["matchup_results"] == [{"focal_index": 0, "opponent_index": 1, "summary": {"games": 1}}]


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
    _copy_repo_configs(tmp_path)
    stack_config = tmp_path / "configs" / "stack_smoke.yaml"

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
    _copy_repo_configs(tmp_path)
    stack_config = tmp_path / "configs" / "stack_smoke.yaml"

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
    _copy_repo_configs(tmp_path)
    stack_config = tmp_path / "configs" / "stack_smoke.yaml"

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


def test_eval_entrypoint_exports_summary_json_and_csv(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    config_hash256 = compute_config_hash256(load_stack_config(stack_config))
    spec_hash256 = spec_bundle_hash(bundle)
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
                            "config_hash256": config_hash256,
                            "spec_hash256": spec_hash256,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "champion",
                        "seat1_policy_id": "baseline",
                        "focal_seat": 0,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                        "decision_count": 5,
                        "tick_count": 5,
                        "no_progress_count": 0,
                        "termination_reason": "terminated",
                        "total_actions": 5,
                        "pass_actions": 0,
                        "main_move_actions": 5,
                        "pass_with_nonpass_available": 0,
                        "max_consecutive_main_moves": 3,
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
                            "config_hash256": config_hash256,
                            "spec_hash256": spec_hash256,
                        "focal_policy_id": "champion",
                        "opponent_policy_id": "baseline",
                        "seat0_policy_id": "baseline",
                        "seat1_policy_id": "champion",
                        "focal_seat": 1,
                        "outcome": "W",
                        "terminated": True,
                        "truncated": False,
                        "engine_status": 0,
                        "decision_count": 4,
                        "tick_count": 4,
                        "no_progress_count": 0,
                        "termination_reason": "terminated",
                        "total_actions": 4,
                        "pass_actions": 0,
                        "main_move_actions": 4,
                        "pass_with_nonpass_available": 0,
                        "max_consecutive_main_moves": 2,
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


def test_eval_entrypoint_rejects_summary_only_export_when_episode_contract_hashes_do_not_match(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(
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
                "decision_count": 5,
                "tick_count": 5,
                "no_progress_count": 0,
                "termination_reason": "terminated",
                "total_actions": 5,
                "pass_actions": 0,
                "main_move_actions": 5,
                "pass_with_nonpass_available": 0,
                "max_consecutive_main_moves": 3,
            },
            sort_keys=True,
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
            str(tmp_path / "summary.json"),
        ],
    )

    assert result.returncode != 0
    assert "config_hash256 does not match" in result.stderr


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
        "matchup_dir": "eval/dev_eval/update_1/b0_randomlegal",
        "episodes_path": "eval/dev_eval/update_1/b0_randomlegal/episodes.jsonl",
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
    stack_config = tmp_path / "configs" / "thesis_locked.yaml"

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
    stack_config = tmp_path / "configs" / "thesis_locked.yaml"
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

