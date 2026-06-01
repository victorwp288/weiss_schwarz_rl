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
from weiss_rl.experiments.segmented_b1_guided_bootstrap_outcomes import SegmentSelectionResult, stop_decision
from weiss_rl.experiments.segmented_b1_guided_bootstrap_plan import (
    build_segment_record,
    build_segmented_bootstrap_summary,
    populate_dry_run_segment_plan,
)
from weiss_rl.experiments.segmented_b1_guided_bootstrap_runner import run_segmented_bootstrap_step


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


def test_segmented_plan_helpers_build_summary_record_and_dry_run_commands(tmp_path: Path) -> None:
    config = SegmentedBootstrapConfig(
        repo_root=tmp_path,
        initial_run_dir=tmp_path / "runs" / "source",
        initial_policy_id="guided_bootstrap_floor_selected",
        run_prefix="floor_loop",
        segments=2,
        runtime=SegmentRuntime(segment_updates=25, num_envs=2, unroll_length=4, device="cpu"),
        confirm_paired_seeds=8,
        dry_run=True,
    )
    summary = build_segmented_bootstrap_summary(
        config=config,
        repo_root=tmp_path,
        created_unix=123.0,
        initial_selection_score=0.61,
    )
    segment_record = build_segment_record(
        config=config,
        repo_root=tmp_path,
        segment_index=1,
        segment_run_label="floor_loop_seg01",
        segment_run_dir=tmp_path / "runs" / "floor_loop_seg01",
        source_run_dir=config.initial_run_dir,
        source_policy_id=config.initial_policy_id,
        source_checkpoint_path=config.initial_run_dir / "training" / "checkpoints" / "checkpoint_25.pt",
        seed_run_dir=config.initial_run_dir,
        train_command=["python", "-m", "weiss_rl.training.train_entrypoint"],
        preselect_json=tmp_path / "diagnostics" / "floor_loop_seg01_candidate_preconfirm.json",
        final_json=tmp_path / "diagnostics" / "floor_loop_seg01_candidate_selection.json",
    )
    populate_dry_run_segment_plan(
        segment_record,
        config=config,
        segment_run_dir=tmp_path / "runs" / "floor_loop_seg01",
        preselect_json=tmp_path / "diagnostics" / "floor_loop_seg01_candidate_preconfirm.json",
        final_json=tmp_path / "diagnostics" / "floor_loop_seg01_candidate_selection.json",
    )

    assert summary["created_unix"] == 123.0
    assert summary["status"] == "planned"
    assert summary["initial_selection_score"] == 0.61
    assert segment_record["source_checkpoint"].endswith("training/checkpoints/checkpoint_25.pt")
    assert "segmented_confirm8_<selected-policy-id>" in segment_record["targeted_confirm_command_template"]["argv"]
    assert "--publish-selected-alias" in segment_record["final_selector_command"]["argv"]
    assert "--publish-selected-alias" not in segment_record["preselect_command"]["argv"]


def test_segmented_outcome_stop_decisions_preserve_status_and_reasons() -> None:
    ineligible = stop_decision(
        SegmentSelectionResult(
            selected={"eligible": False},
            selection_score=0.40,
            selected_minus_previous=None,
            latest_minus_best=None,
        ),
        max_selected_drop=0.02,
        stop_on_latest_falloff=False,
        max_latest_drop=0.05,
    )
    selected_drop = stop_decision(
        SegmentSelectionResult(
            selected={"eligible": True},
            selection_score=0.50,
            selected_minus_previous=-0.03,
            latest_minus_best=-0.01,
        ),
        max_selected_drop=0.02,
        stop_on_latest_falloff=True,
        max_latest_drop=0.05,
    )
    latest_falloff = stop_decision(
        SegmentSelectionResult(
            selected={"eligible": True},
            selection_score=0.60,
            selected_minus_previous=0.01,
            latest_minus_best=-0.06,
        ),
        max_selected_drop=0.02,
        stop_on_latest_falloff=True,
        max_latest_drop=0.05,
    )
    keep_running = stop_decision(
        SegmentSelectionResult(
            selected={"eligible": True},
            selection_score=0.60,
            selected_minus_previous=-0.01,
            latest_minus_best=-0.06,
        ),
        max_selected_drop=0.02,
        stop_on_latest_falloff=False,
        max_latest_drop=0.05,
    )

    assert ineligible.status == "stopped_ineligible"
    assert ineligible.stop_reason == "selected candidate did not meet required anchor threshold"
    assert selected_drop.status == "stopped_selected_drop"
    assert selected_drop.stop_reason == "selected score dropped by -0.0300, below -0.0200"
    assert latest_falloff.status == "stopped_latest_falloff"
    assert latest_falloff.stop_reason == "latest fell behind best by -0.0600, below -0.0500"
    assert keep_running.should_stop is False


def test_segmented_step_executes_commands_and_returns_stop_status(tmp_path: Path) -> None:
    initial_run_dir = tmp_path / "runs" / "source"
    _write_registry(initial_run_dir, policy_id="guided_bootstrap_floor_selected", update=25)
    config = SegmentedBootstrapConfig(
        repo_root=tmp_path,
        initial_run_dir=initial_run_dir,
        initial_policy_id="guided_bootstrap_floor_selected",
        run_prefix="floor_loop",
        runtime=SegmentRuntime(segment_updates=25, num_envs=2, unroll_length=4, device="cpu"),
        confirm_paired_seeds=4,
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "weiss_rl.training.train_entrypoint" in command:
            run_label = _option(command, "--run-label")
            run_dir = tmp_path / "runs" / run_label
            _write_registry(run_dir, policy_id="policy_000005", update=25)
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
            output_json = tmp_path / _option(command, "--output-json")
            run_dir = tmp_path / _option(command, "--run-dir")
            eligible = "--publish-selected-alias" not in command
            _write_json(
                output_json,
                {
                    "selected": {
                        "run_dir": run_dir.as_posix(),
                        "run_name": run_dir.name,
                        "snapshot_policy_id": "policy_000005",
                        "update_count": 25,
                        "selection_score": 0.40,
                        "eligible": eligible,
                    },
                    "run_summaries": [{"run_name": run_dir.name, "latest_minus_best": -0.01}],
                },
            )
        return subprocess.CompletedProcess(command, 0)

    result = run_segmented_bootstrap_step(
        config=config,
        repo_root=tmp_path,
        diagnostics_dir=tmp_path / "diagnostics",
        segment_index=1,
        current_run_dir=initial_run_dir,
        current_seed_run_dir=initial_run_dir,
        current_policy_id="guided_bootstrap_floor_selected",
        previous_score=0.57,
        runner=fake_runner,
    )

    assert result.should_stop is True
    assert result.terminal_status == "stopped_ineligible"
    assert result.terminal_reason == "selected candidate did not meet required anchor threshold"
    assert result.next_run_dir == initial_run_dir
    assert result.segment_record["status"] == "completed"
    assert result.segment_record["selected"]["eligible"] is False
    assert result.segment_record["preselect_command"]["argv"].count("--publish-selected-alias") == 0
    assert "--publish-selected-alias" in result.segment_record["final_selector_command"]["argv"]
    command_modules = [command[command.index("-m") + 1] for command, _env in calls]
    assert command_modules == [
        "weiss_rl.training.train_entrypoint",
        "weiss_rl.experiments.select_b1_candidate_entrypoint",
        "weiss_rl.eval.targeted_confirm_entrypoint",
        "weiss_rl.experiments.select_b1_candidate_entrypoint",
    ]
    fixed_envs = [
        env
        for command, env in calls
        if "weiss_rl.training.train_entrypoint" in command or "weiss_rl.eval.targeted_confirm_entrypoint" in command
    ]
    assert all(env is not None and env["PYTHONHASHSEED"] == "0" for env in fixed_envs)


def test_segmented_bootstrap_reanchors_next_segment_to_published_alias(tmp_path: Path) -> None:
    initial_run_dir = tmp_path / "runs" / "source"
    _write_registry(initial_run_dir, policy_id="guided_bootstrap_floor_selected", update=25)
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_runner(command: list[str], **_: Any) -> subprocess.CompletedProcess[Any]:
        calls.append((command, _.get("env")))
        if "weiss_rl.training.train_entrypoint" in command:
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
        elif "weiss_rl.experiments.select_b1_candidate_entrypoint" in command:
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
    train_commands = [command for command, _env in calls if "weiss_rl.training.train_entrypoint" in command]
    train_envs = [env for command, env in calls if "weiss_rl.training.train_entrypoint" in command]
    assert _option(train_commands[1], "--init-from-checkpoint").endswith(
        "runs/floor_loop_seg01/training/checkpoints/checkpoint_25.pt"
    )
    assert all(env is not None and env["PYTHONHASHSEED"] == "0" for env in train_envs)
