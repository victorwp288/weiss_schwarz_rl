from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from weiss_rl.workflows.cli_dispatch import _WORKFLOW_HANDLERS
from weiss_rl.workflows.cli_parser import build_workflow_parser

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_cli_b1_source_run(run_dir: Path, *, policy_id: str = "selected_candidate", update: int = 15) -> Path:
    checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": policy_id,
                        "update": update,
                        "weights_sha256": "a" * 64,
                        "path": f"training/snapshots/{policy_id}/weights.pt",
                    }
                ],
                "champion_snapshots": [],
                "pinned_snapshots": [policy_id],
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def test_package_cli_parser_commands_are_all_dispatchable() -> None:
    parser = build_workflow_parser()
    command_actions = [action for action in parser._actions if action.dest == "command"]

    assert len(command_actions) == 1
    assert set(command_actions[0].choices) == set(_WORKFLOW_HANDLERS)


def test_package_cli_train_b1_dry_run_uses_thesis_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_smoke",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "b1_smoke.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert "configs/thesis/b1_noleague.yaml" in payload["command"]
    assert "--runtime-mode" in payload["command"]
    assert "train_async_fast" in payload["command"]
    assert "--profile" in payload["command"]
    assert "fast" in payload["command"]


def test_package_cli_train_b1_gpu_probe_uses_cuda_probe_shape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_gpu_probe",
            "--profile",
            "gpu-probe",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "b1_gpu_probe.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert payload["profile"] == "gpu-probe"
    assert "--device" in payload["command"]
    assert "cuda" in payload["command"]
    assert "--num-envs" in payload["command"]
    assert "32" in payload["command"]
    assert "--unroll-length" in payload["command"]
    assert "16" in payload["command"]
    assert "training.profile_timers=true" in payload["command"]


def test_package_cli_train_b1_league_probe_uses_early_guard_shape(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_league_probe",
            "--profile",
            "league-probe",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "b1_league_probe.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-b1"
    assert payload["profile"] == "league-probe"
    assert "--device" in payload["command"]
    assert "cuda" in payload["command"]
    assert "--num-envs" in payload["command"]
    assert "288" in payload["command"]
    assert "--unroll-length" in payload["command"]
    assert "64" in payload["command"]
    assert "--max-updates" in payload["command"]
    assert "50" in payload["command"]
    assert "--checkpoint-interval-updates" in payload["command"]
    assert "5" in payload["command"]
    assert "system.collection_backend=process" in payload["command"]
    assert "training.profile_timers=true" in payload["command"]


def test_package_cli_train_b1_guided_seed_uses_guided_seed_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-b1-guided-seed",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "b1_guided_seed_smoke",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "b1_guided_seed_smoke.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-b1-guided-seed"
    assert "configs/thesis/b1_guided_seed.yaml" in payload["command"]


def test_package_cli_train_main_requires_b1_and_uses_main_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    b1_run = repo_root / "runs" / "b1_smoke"
    checkpoint_path = _write_cli_b1_source_run(b1_run, policy_id="b1_noleague_baseline")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_smoke",
            "--b1-run",
            str(b1_run),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "main_smoke.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-main"
    assert payload["init_policy_id"] == "b1_noleague_baseline"
    assert "configs/thesis/main_league.yaml" in payload["command"]
    assert b1_run.as_posix() in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_smoke_accepts_single_unaliased_b1_snapshot(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    b1_run = repo_root / "runs" / "b1_smoke"
    checkpoint_path = _write_cli_b1_source_run(b1_run, policy_id="policy_000001", update=1)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_smoke",
            "--b1-run",
            str(b1_run),
            "--profile",
            "smoke",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "main_smoke.json").read_text(encoding="utf-8"))
    assert payload["workflow"] == "train-main"
    assert payload["profile"] == "smoke"
    assert payload["init_policy_id"] == "policy_000001"
    assert "--init-from-checkpoint" in payload["command"]
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_strict_profile_rejects_unaliased_b1_snapshot(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    b1_run = repo_root / "runs" / "b1_smoke"
    _write_cli_b1_source_run(b1_run, policy_id="policy_000001", update=1)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_strict",
            "--b1-run",
            str(b1_run),
            "--profile",
            "thesis-local",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Could not resolve a B1 seed checkpoint from --b1-run" in result.stderr
    assert "Traceback" not in result.stderr


def test_package_cli_train_main_accepts_guided_seed_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    b1_run = repo_root / "runs" / "b1_smoke"
    seed_run = repo_root / "runs" / "b1_guided_seed_smoke"
    checkpoint_path = _write_cli_b1_source_run(b1_run)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_seeded_smoke",
            "--b1-run",
            str(b1_run),
            "--seed-run",
            str(seed_run),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_seeded_smoke.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main"
    assert "--b1-baseline-run-dir" in payload["command"]
    assert b1_run.as_posix() in payload["command"]
    assert "--seed-snapshot-run-dir" in payload["command"]
    assert seed_run.as_posix() in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_guided_bootstrap_uses_seed_and_warmstart_without_strict_b1(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_smoke",
            "--seed-run",
            str(seed_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_smoke.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert "configs/thesis/main_league_guided_bootstrap.yaml" in payload["command"]
    assert "--seed-snapshot-run-dir" in payload["command"]
    assert seed_run.as_posix() in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert init_checkpoint.as_posix() in payload["command"]
    assert "--b1-baseline-run-dir" not in payload["command"]


def test_package_cli_train_main_guided_bootstrap_accepts_optional_strict_b1_anchor(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    b1_run = repo_root / "runs" / "b1_noleague_candidate"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_with_b1",
            "--seed-run",
            str(seed_run),
            "--b1-run",
            str(b1_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_with_b1.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert "--b1-baseline-run-dir" in payload["command"]
    assert b1_run.as_posix() in payload["command"]


def test_package_cli_train_main_guided_bootstrap_vtrace_uses_clamped_stack(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_vtrace",
            "--seed-run",
            str(seed_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--vtrace-clamp",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_vtrace.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert payload["vtrace_clamp"] is True
    assert "configs/thesis/main_league_guided_bootstrap_vtrace.yaml" in payload["command"]


def test_package_cli_train_main_guided_bootstrap_seed_champions_uses_seedchampion_stack(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "b1_guided_seed"
    init_checkpoint = repo_root / "runs" / "teacherfade" / "training" / "checkpoints" / "best.pt"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_bootstrap_seedchampion",
            "--seed-run",
            str(seed_run),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--seed-champions",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_guided_bootstrap_seedchampion.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert payload["seed_champions"] is True
    assert payload["vtrace_clamp"] is False
    assert "configs/thesis/main_league_guided_bootstrap_seedchampion.yaml" in payload["command"]


def test_package_cli_train_main_guided_bootstrap_selected_resolves_init_policy_id(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_run = repo_root / "runs" / "guided_source"
    checkpoint_path = source_run / "training" / "checkpoints" / "checkpoint_90.pt"
    registry_path = source_run / "training" / "snapshots" / "registry.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "snapshots": [
                    {
                        "policy_id": "guided_bootstrap_selected",
                        "update": 90,
                        "weights_sha256": "a" * 64,
                        "path": "training/snapshots/guided_bootstrap_selected/weights.pt",
                    }
                ],
                "champion_snapshots": [],
                "pinned_snapshots": ["guided_bootstrap_selected"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "main_guided_selected",
            "--seed-run",
            str(source_run),
            "--init-from-run-dir",
            str(source_run),
            "--init-policy-id",
            "guided_bootstrap_selected",
            "--selected-seed-champion",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((repo_root / "runs" / "_workflow_plans" / "main_guided_selected.json").read_text())
    assert payload["workflow"] == "train-main-guided-bootstrap"
    assert payload["selected_seed_champion"] is True
    assert payload["init_policy_id"] == "guided_bootstrap_selected"
    assert (
        "configs/thesis/main_league_guided_bootstrap_selected_trajbc_direct_b2b3b4_anchor_nopublic.yaml"
        in payload["command"]
    )
    assert checkpoint_path.as_posix() in payload["command"]


def test_package_cli_train_main_guided_bootstrap_rejects_ambiguous_init_sources(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "ambiguous",
            "--seed-run",
            str(repo_root / "runs" / "seed"),
            "--init-from-checkpoint",
            str(repo_root / "runs" / "source" / "training" / "checkpoints" / "checkpoint_90.pt"),
            "--init-from-run-dir",
            str(repo_root / "runs" / "source"),
            "--init-policy-id",
            "guided_bootstrap_selected",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--init-from-checkpoint cannot be combined" in result.stderr
    assert "Traceback" not in result.stderr


def test_package_cli_train_main_guided_bootstrap_requires_init_source(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "train-main-guided-bootstrap",
            "--repo-root",
            str(repo_root),
            "--run-label",
            "missing_init",
            "--seed-run",
            str(repo_root / "runs" / "seed"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "requires either --init-from-checkpoint or --init-from-run-dir plus --init-policy-id" in result.stderr
    assert "Traceback" not in result.stderr


def test_package_cli_smoke_eval_uses_tiny_eval_budget(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "smoke-eval",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(repo_root / "runs" / "main_smoke"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_smoke_smoke-eval.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "smoke-eval"
    assert "configs/thesis/main_league.yaml" in payload["command"]
    assert payload["command"].count("--policy-id") == 5
    assert "B4 HeuristicPublicControl" in payload["command"]
    assert "--paired-seed-limit" in payload["command"]
    assert "--skip-readiness" in payload["command"]


def test_package_cli_b2_audit_wraps_standard_disagreement_script(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_smoke"
    episodes_jsonl = run_dir / "eval" / "final_eval" / "episodes.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "b2-audit",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--episodes-jsonl",
            str(episodes_jsonl),
            "--policy-id",
            "policy_000001",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_smoke_b2-audit.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "b2-audit"
    assert "python/scripts/b2_disagreement_audit.py" in payload["command"]
    assert "configs/thesis/final_eval.yaml" in payload["command"]
    assert "--episodes-jsonl" in payload["command"]
    assert episodes_jsonl.as_posix() in payload["command"]
    assert "--policy-id" in payload["command"]
    assert "policy_000001" in payload["command"]
    assert (run_dir / "eval" / "b2_disagreement").as_posix() in payload["command"]


def test_package_cli_guard_run_wraps_learning_progress_league_guard(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_probe"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guard-run",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--min-latest-anchor-score",
            "0.5",
            "--max-vtrace-rho-p99",
            "25",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_probe_guard-run.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "guard-run"
    assert "python/scripts/learning_progress_diagnostic.py" in payload["command"]
    assert "--league-guard" in payload["command"]
    assert "--run-dir" in payload["command"]
    assert run_dir.as_posix() in payload["command"]
    assert "--guard-min-latest-anchor-score" in payload["command"]
    assert "0.5" in payload["command"]
    assert "--guard-max-vtrace-rho-p99" in payload["command"]
    assert "25.0" in payload["command"]
    assert payload["command"].count("--guard-required-anchor") == 3
    assert "B4 HeuristicPublicControl" in payload["command"]


def test_package_cli_guard_run_custom_required_anchors_replace_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_probe"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guard-run",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--required-anchor",
            "B2 HeuristicPublic",
            "--required-anchor",
            "custom_anchor",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_probe_guard-run.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "guard-run"
    assert payload["command"].count("--guard-required-anchor") == 2
    assert "B2 HeuristicPublic" in payload["command"]
    assert "custom_anchor" in payload["command"]
    assert "B3 HeuristicPublicAggro" not in payload["command"]
    assert "B4 HeuristicPublicControl" not in payload["command"]


def test_package_cli_figures_dry_run_forwards_figure_options(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    run_dir = repo_root / "runs" / "main_probe"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "figures",
            "--repo-root",
            str(repo_root),
            "--run-dir",
            str(run_dir),
            "--fig-id",
            "seat_bias",
            "--format",
            "png",
            "--format",
            "pdf",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "main_probe_figures.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "figures"
    assert payload["command"] == [
        sys.executable,
        "python/scripts/make_figures.py",
        "--run-dir",
        run_dir.as_posix(),
        "--fig-id",
        "seat_bias",
        "--format",
        "png",
        "--format",
        "pdf",
    ]


def test_package_cli_guided_bootstrap_loop_wraps_segmented_controller(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    initial_run_dir = repo_root / "runs" / "guided_floor"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guided-bootstrap-loop",
            "--repo-root",
            str(repo_root),
            "--initial-run-dir",
            str(initial_run_dir),
            "--initial-policy-id",
            "guided_bootstrap_floor_selected",
            "--run-prefix",
            "floor_loop",
            "--segments",
            "2",
            "--segment-updates",
            "25",
            "--confirm-paired-seeds",
            "64",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "floor_loop_guided-bootstrap-loop.json").read_text(encoding="utf-8")
    )
    assert payload["workflow"] == "guided-bootstrap-loop"
    assert "python/scripts/segmented_b1_guided_bootstrap.py" in payload["command"]
    assert "--initial-run-dir" in payload["command"]
    assert initial_run_dir.as_posix() in payload["command"]
    assert "--stack-config" in payload["command"]
    assert "configs/thesis/main_league_guided_bootstrap_selected_anchor_floor.yaml" in payload["command"]
    assert "--confirm-paired-seeds" in payload["command"]
    assert "64" in payload["command"]


def test_package_cli_guarded_league_bootstrap_wraps_controller(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    seed_run = repo_root / "runs" / "guided_selected"
    init_checkpoint = seed_run / "training" / "checkpoints" / "checkpoint_25.pt"
    reference_summary = seed_run / "eval" / "targeted_confirm256" / "targeted_confirm256_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guarded-league-bootstrap",
            "--repo-root",
            str(repo_root),
            "--init-from-checkpoint",
            str(init_checkpoint),
            "--seed-snapshot-run-dir",
            str(seed_run),
            "--run-prefix",
            "guarded_selected",
            "--segments",
            "2",
            "--segment-updates",
            "10",
            "--confirm-paired-seeds",
            "64",
            "--publish-min-confirm-paired-seeds",
            "256",
            "--confirm-recent-candidate-count",
            "3",
            "--first-init-schedule-offset-updates",
            "0",
            "--reference-summary-json",
            str(reference_summary),
            "--reference-label",
            "selected_confirm256",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (repo_root / "runs" / "_workflow_plans" / "guarded_selected_guarded-league-bootstrap.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["workflow"] == "guarded-league-bootstrap"
    assert "python/scripts/guarded_league_bootstrap.py" in payload["command"]
    assert "--init-from-checkpoint" in payload["command"]
    assert init_checkpoint.as_posix() in payload["command"]
    assert "--seed-snapshot-run-dir" in payload["command"]
    assert seed_run.as_posix() in payload["command"]
    assert "--reference-summary-json" in payload["command"]
    assert reference_summary.as_posix() in payload["command"]
    assert "--first-init-schedule-offset-updates" in payload["command"]
    assert "0" in payload["command"]
    assert "--publish-min-confirm-paired-seeds" in payload["command"]
    assert "--confirm-recent-candidate-count" in payload["command"]
    assert "3" in payload["command"]
    assert "--max-reference-drop" in payload["command"]
    assert "0.04" in payload["command"]
    assert "--selected-alias-policy-id" in payload["command"]
    assert "main_league_selected" in payload["command"]
    assert payload["publish_min_confirm_paired_seeds"] == 256
    assert payload["confirm_recent_candidate_count"] == 3
    assert payload["selected_alias_policy_id"] == "main_league_selected"


def test_package_cli_guard_run_failure_exits_without_traceback(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "main_bad"
    logs_dir = run_dir / "training" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "training_metrics.jsonl").write_text(
        json.dumps({"update_count": 20, "loss": 1.0, "vtrace_rho_p99": 31.0}) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "periodic_dev_eval_summaries.json").write_text(
        json.dumps(
            {
                "train_u20_p4": {
                    "update_count": 20,
                    "aggregate_score": 0.50,
                    "anchor_scores": {
                        "B2 HeuristicPublic": 0.34,
                        "B3 HeuristicPublicAggro": 0.41,
                        "B4 HeuristicPublicControl": 0.47,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    for update in (10, 15, 20):
        gate_path = run_dir / "eval" / "promotion_gate" / f"update_{update}" / "promotion_gate.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(
            json.dumps(
                {
                    "focal_policy_id": f"policy_{update:06d}",
                    "decision": {"passed": False, "reasons": [{"code": "anchor_loss_guardrail_exceeded"}]},
                    "overall_posterior": {"mean": 0.5, "prob_gt_target": 0.1},
                }
            ),
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "weiss_rl.cli",
            "guard-run",
            "--run-dir",
            str(run_dir),
            "--max-vtrace-rho-p99",
            "25",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "league guard failed" in result.stderr
    assert "Traceback" not in result.stderr
