from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from weiss_rl.experiments.segmented_b1_guided_bootstrap import (
    SegmentedBootstrapConfig,
    SegmentRuntime,
    resolve_snapshot_checkpoint_path,
    run_segmented_b1_guided_bootstrap,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_registry(run_dir: Path, *, policy_id: str, update: int) -> None:
    _write_json(
        run_dir / "training" / "snapshots" / "registry.json",
        {
            "schema_version": 1,
            "snapshots": [
                {
                    "policy_id": policy_id,
                    "update": update,
                    "path": f"training/snapshots/{policy_id}/weights.pt",
                    "weights_sha256": f"sha-{policy_id}",
                }
            ],
            "pinned_snapshots": [policy_id],
        },
    )
    checkpoint_path = run_dir / "training" / "checkpoints" / f"checkpoint_{update}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_resolve_snapshot_checkpoint_path_uses_snapshot_update(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    _write_registry(run_dir, policy_id="guided_bootstrap_floor_selected", update=25)

    checkpoint = resolve_snapshot_checkpoint_path(
        run_dir=run_dir,
        policy_id="guided_bootstrap_floor_selected",
    )

    assert checkpoint == run_dir / "training" / "checkpoints" / "checkpoint_25.pt"


def test_segmented_bootstrap_dry_run_writes_first_segment_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "source"
    _write_registry(run_dir, policy_id="guided_bootstrap_floor_selected", update=25)

    summary = run_segmented_b1_guided_bootstrap(
        SegmentedBootstrapConfig(
            repo_root=tmp_path,
            initial_run_dir=run_dir,
            initial_policy_id="guided_bootstrap_floor_selected",
            run_prefix="floor_loop",
            segments=3,
            runtime=SegmentRuntime(segment_updates=25, num_envs=2, unroll_length=4, device="cpu"),
            dry_run=True,
        )
    )

    assert summary["status"] == "planned"
    assert len(summary["segments"]) == 1
    segment = summary["segments"][0]
    assert segment["source_policy_id"] == "guided_bootstrap_floor_selected"
    assert segment["source_checkpoint"].endswith("training/checkpoints/checkpoint_25.pt")
    assert "targeted_confirm_command_template" in segment
    written = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert written["status"] == "planned"


def test_segmented_bootstrap_reanchors_next_segment_to_published_alias(tmp_path: Path) -> None:
    initial_run_dir = tmp_path / "runs" / "source"
    _write_registry(initial_run_dir, policy_id="guided_bootstrap_floor_selected", update=25)
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "python/scripts/train.py" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, policy_id="policy_000005", update=25)
            _write_json(
                run_dir / "training" / "logs" / "periodic_dev_eval_summaries.json",
                {
                    "train_u25_p5": {
                        "update_count": 25,
                        "policy_version": 5,
                        "aggregate_score": 0.68,
                        "anchor_scores": {
                            "B2 HeuristicPublic": 0.58,
                            "B3 HeuristicPublicAggro": 0.57,
                            "B4 HeuristicPublicControl": 0.56,
                        },
                    }
                },
            )
        elif "python/scripts/select_b1_candidate.py" in command:
            output_json = tmp_path / _option(command, "--output-json")
            run_dir = tmp_path / _option(command, "--run-dir")
            selected = {
                "run_dir": run_dir.as_posix(),
                "run_name": run_dir.name,
                "snapshot_policy_id": "policy_000005",
                "update_count": 25,
                "selection_score": 0.57,
                "required_anchor_min": 0.56,
                "required_anchor_mean": 0.57,
                "eligible": True,
            }
            _write_json(
                output_json,
                {
                    "selected": selected,
                    "run_summaries": [{"run_name": run_dir.name, "latest_minus_best": -0.01}],
                },
            )
            if "--publish-selected-alias" in command:
                _write_registry(run_dir, policy_id="guided_bootstrap_floor_segmented_selected", update=25)
        return subprocess.CompletedProcess(command, 0)

    summary = run_segmented_b1_guided_bootstrap(
        SegmentedBootstrapConfig(
            repo_root=tmp_path,
            initial_run_dir=initial_run_dir,
            initial_policy_id="guided_bootstrap_floor_selected",
            run_prefix="floor_loop",
            segments=2,
            runtime=SegmentRuntime(segment_updates=25, num_envs=2, unroll_length=4, device="cpu"),
            confirm_paired_seeds=4,
        ),
        runner=fake_runner,
    )

    assert summary["status"] == "completed"
    assert len(summary["segments"]) == 2
    assert summary["segments"][1]["source_run_dir"].endswith("runs/floor_loop_seg01")
    assert summary["segments"][1]["source_policy_id"] == "guided_bootstrap_floor_segmented_selected"
    train_commands = [command for command, _env in calls if "python/scripts/train.py" in command]
    train_envs = [env for command, env in calls if "python/scripts/train.py" in command]
    assert _option(train_commands[1], "--init-from-checkpoint").endswith(
        "runs/floor_loop_seg01/training/checkpoints/checkpoint_25.pt"
    )
    assert all(env is not None and env["PYTHONHASHSEED"] == "0" for env in train_envs)
