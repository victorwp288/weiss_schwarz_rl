from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from weiss_rl.config import compute_config_hash256, load_stack_config
from weiss_rl.spec import spec_bundle_hash

REPO_ROOT = Path(__file__).resolve().parents[3]


def _mismatched_sha256(value: str) -> str:
    return ("0" if value[0] != "0" else "1") + value[1:]


def _write_stub_weiss_sim(
    tmp_path: Path,
    *,
    spec_hash: int = 123,
    pass_action_id: int = 8,
) -> dict[str, object]:
    bundle: dict[str, object] = {
        "policy_version": 3,
        "spec_hash": spec_hash,
        "observation": {
            "obs_encoding_version": 2,
            "dtype": "i32",
            "obs_len": 512,
        },
        "action": {
            "action_encoding_version": 1,
            "action_space_size": 9,
            "pass_action_id": pass_action_id,
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
                "rl = _Rl()",
            )
        ),
        encoding="utf-8",
    )
    return bundle


def _patch_periodic_dev_eval_config(tmp_path: Path) -> None:
    evaluation_path = tmp_path / "configs" / "evaluation_locked.yaml"
    evaluation_text = evaluation_path.read_text(encoding="utf-8")
    evaluation_text = evaluation_text.replace(
        "periodic_dev_eval_interval_updates: 50000",
        "periodic_dev_eval_interval_updates: 1",
    )
    evaluation_text = evaluation_text.replace(
        "periodic_dev_eval_paired_seeds: 64",
        "periodic_dev_eval_paired_seeds: 1",
    )
    evaluation_path.write_text(evaluation_text, encoding="utf-8")
    (tmp_path / "configs" / "seeds" / "dev_eval_seeds.txt").write_text("7\n", encoding="utf-8")


def _copy_repo_configs(tmp_path: Path) -> Path:
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    return tmp_path / "configs" / "rl_stack_locked.yaml"


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
    stack_config = _copy_repo_configs(tmp_path)

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
        "status": "unresolved",
        "version": "deterministic_v1",
        "final_policy_set_size": 10,
        "source_paths": {
            "snapshot_registry_json": None,
            "dev_eval_summaries_json": None,
        },
        "missing_inputs": ["snapshot_registry_json", "dev_eval_summaries_json"],
        "reason": "deterministic final policy set inputs were not provided",
    }
    assert (manifest_path.parent / "spec_bundle.json").is_file()
    assert (manifest_path.parent / "spec_hash256.txt").read_text(encoding="utf-8").strip() == spec_bundle_hash(bundle)
    assert "computed_run_id64:" in result.stdout
    assert "computed_run_id256:" in result.stdout
    assert "run_label:              spec_bundle_run" in result.stdout
    assert "run_dir_name:           spec_bundle_run" in result.stdout
    assert "Manifest scaffold only: no learner training or rollout collection was executed." in result.stdout
    assert "active weiss_sim runtime is missing stepping APIs" in result.stdout


def test_train_entrypoint_resolves_policy_set_selection_when_inputs_are_supplied(tmp_path: Path) -> None:
    bundle = _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
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


def test_train_entrypoint_uses_default_run_dir_when_no_label_override(tmp_path: Path) -> None:
    _write_stub_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)

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
    stack_config = _copy_repo_configs(tmp_path)

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


def test_train_entrypoint_runs_periodic_dev_eval_and_handles_empty_ids_pass_fallback(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(
        tmp_path,
        spec_hash=123,
        pass_action_id=3,
        empty_eval_legal_row=True,
    )
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)

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
        ],
    )

    assert result.returncode == 0, result.stderr
    assert "Periodic dev eval: update=1 opponent=b0_randomlegal" in result.stdout

    eval_root = tmp_path / "runs" / "periodic_dev_eval_run" / "eval" / "dev_eval" / "update_1"
    seed_usage = json.loads((eval_root / "seed_usage.json").read_text(encoding="utf-8"))
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
        "seed_usage_path": "eval/dev_eval/update_1/seed_usage.json",
    }
    assert diagnostics_payload["seat_results"]["seat0_wins"] == 2
    assert diagnostics_payload["seat_results"]["seat1_wins"] == 0
    assert (eval_root / "b0_randomlegal" / "matchup_summary.csv").is_file()


def test_train_entrypoint_periodic_dev_eval_writes_exact_current_checkpoint(tmp_path: Path) -> None:
    bundle = _write_runtime_weiss_sim(tmp_path, spec_hash=123)
    stack_config = _copy_repo_configs(tmp_path)
    _patch_periodic_dev_eval_config(tmp_path)

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
        ],
    )

    assert result.returncode == 0, result.stderr

    run_root = tmp_path / "runs" / "periodic_dev_eval_checkpoint_traceability"
    eval_root = run_root / "eval" / "dev_eval" / "update_1"
    checkpoint_path = run_root / "training" / "checkpoints" / "checkpoint_1.pt"
    seed_usage = json.loads((eval_root / "seed_usage.json").read_text(encoding="utf-8"))
    summary_payload = json.loads((eval_root / "b0_randomlegal" / "matchup_summary.json").read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    assert checkpoint_path.is_file()
    assert seed_usage["focal_policy"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert summary_payload["evaluation_context"]["checkpoint_path"] == "training/checkpoints/checkpoint_1.pt"
    assert checkpoint_payload["update_count"] == 1
