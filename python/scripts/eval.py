from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

from weiss_rl.artifacts import ArtifactLayout
from weiss_rl.artifacts.reproducibility import parse_seed_file
from weiss_rl.config import compute_config_hash256, load_stack_config, load_study_config
from weiss_rl.core.simulator_contract import load_verified_simulator_contract
from weiss_rl.core.spec import assert_spec_bundle_contract
from weiss_rl.diagnostics.cli_banner import print_startup_banner
from weiss_rl.diagnostics.tensorboard_logger import TensorBoardLogger, tensorboard_unavailable_reason
from weiss_rl.eval import (
    build_matchup_export,
    build_paper_readiness_summary,
    build_seat_advantage_diagnostics,
    load_dev_eval_summaries,
    load_eval_game_records,
    resolve_final_policy_set,
    run_final_eval,
    write_matchup_diagnostics_json,
    write_matchup_summary_csv,
    write_matchup_summary_json,
    write_paper_readiness_json,
)
from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.eval.policy_set import recommend_focal_policy_id
from weiss_rl.eval.simulator_runner import SimulatorEvalRunner, resolve_eval_policies
from weiss_rl.experiments.toy_public_demo import (
    PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES,
    PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS,
    public_demo_spec_bundle,
    public_demo_spec_hash256,
    public_demo_stop_rules,
    run_public_demo_final_eval,
)
from weiss_rl.metagame import build_sensitivity_report
from weiss_rl.plotting.paper_figures import render_paper_figures

_SHA256_HEX_LENGTH = 64
_GIT_COMMIT_HEX_LENGTH = 40


def _normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _expected_sha256(value: str, *, flag_name: str) -> str:
    if not value.strip():
        return ""
    normalized = _normalize_sha256(value)
    if not normalized:
        raise ValueError(f"{flag_name} must be a 64-character lowercase or uppercase SHA-256 hex string")
    return normalized


def _require_matching_hash(*, flag_name: str, expected: str, actual: str) -> None:
    if expected and expected != actual:
        raise RuntimeError(f"{flag_name} mismatch: expected {expected}, observed {actual}")


def _normalize_git_commit(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _GIT_COMMIT_HEX_LENGTH:
        return ""
    if any(char not in "0123456789abcdef" for char in normalized):
        return ""
    return normalized


def _resolve_run_label(parser: argparse.ArgumentParser, run_label: str, run_id_alias: str) -> str:
    normalized_label = run_label.strip()
    normalized_alias = run_id_alias.strip()
    if normalized_label and normalized_alias and normalized_label != normalized_alias:
        parser.error("--run-label and deprecated --run-id must match when both are provided")
    if normalized_alias:
        print("Warning: --run-id is deprecated; use --run-label instead.", file=sys.stderr)
    return normalized_label or normalized_alias


def _require_positive_int(parser: argparse.ArgumentParser, flag_name: str, value: int | None) -> int | None:
    if value is None:
        return None
    normalized = int(value)
    if normalized < 1:
        parser.error(f"{flag_name} must be >= 1")
    return normalized


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must contain an object at the top level")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_run_summary_or_default(layout: ArtifactLayout) -> dict[str, Any]:
    if layout.run_summary_path.is_file():
        return _load_json_object(layout.run_summary_path, label="run summary")
    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    return {
        "kind": "run_summary_v1",
        "artifact_schema_version": "run_artifacts_v2",
        "run_id256": str(manifest.get("run_id256", "")),
        "run_id64": str(manifest.get("run_id64", "")),
        "run_dir": layout.run_dir.as_posix(),
        "artifact_roots": {
            "training": layout.relative(layout.training_dir),
            "eval": layout.relative(layout.eval_dir),
            "replays": layout.relative(layout.replays_dir),
            "tensorboard": layout.relative(layout.tensorboard_dir),
            "figures": layout.relative(layout.figures_dir),
        },
        "manifest_path": layout.relative(layout.manifest_path),
        "environment_path": layout.relative(layout.environment_path),
        "determinism_report_path": layout.relative(layout.determinism_report_path),
        "paper_readiness_summary_path": layout.relative(layout.paper_readiness_summary_path),
        "seed_derivation": manifest.get("seed_derivation", {}),
        "runtime_mode": "interpolated_checkpoint",
        "paper_grade": False,
    }


def _load_determinism_report_or_default(layout: ArtifactLayout) -> dict[str, Any]:
    if layout.determinism_report_path.is_file():
        return _load_json_object(layout.determinism_report_path, label="determinism report")
    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    evaluation_pinning = manifest.get("evaluation_pinning", {})
    if not isinstance(evaluation_pinning, dict):
        evaluation_pinning = {}
    seed_derivation = manifest.get("seed_derivation", {})
    if not isinstance(seed_derivation, dict):
        seed_derivation = {}
    seed_files = manifest.get("seed_files", {})
    if not isinstance(seed_files, dict):
        seed_files = {}
    return {
        "kind": "determinism_report_v1",
        "artifact_schema_version": "run_artifacts_v2",
        "run_id256": str(manifest.get("run_id256", "")),
        "run_id64": str(manifest.get("run_id64", "")),
        "policy_selection_mode": "unresolved",
        "evaluation_pinning": evaluation_pinning,
        "seed_derivation": seed_derivation,
        "seed_files": seed_files,
        "device_policy": {
            "learner": "interpolated_checkpoint",
            "evaluation": evaluation_pinning.get("eval_device", "cpu"),
        },
        "replay_verification": {
            "path": layout.relative(layout.replay_verification_json()),
            "status": "pending",
        },
        "canonical_artifact_hashes": {},
    }


def _load_environment_or_default(layout: ArtifactLayout) -> dict[str, Any]:
    if layout.environment_path.is_file():
        return _load_json_object(layout.environment_path, label="environment manifest")
    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    package_names = ("weiss-rl", "weiss-sim", "torch", "numpy", "scipy", "matplotlib")
    return {
        "kind": "environment_manifest_v1",
        "artifact_schema_version": "run_artifacts_v2",
        "run_id256": str(manifest.get("run_id256", "")),
        "run_id64": str(manifest.get("run_id64", "")),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {name: _safe_package_version(name) for name in package_names},
    }


def _safe_package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _ensure_run_level_report_scaffolding(layout: ArtifactLayout) -> None:
    if not layout.environment_path.is_file():
        _write_json(layout.environment_path, _load_environment_or_default(layout))
    if not layout.run_summary_path.is_file():
        _write_json(layout.run_summary_path, _load_run_summary_or_default(layout))
    if not layout.determinism_report_path.is_file():
        _write_json(layout.determinism_report_path, _load_determinism_report_or_default(layout))


def _effective_manifest_git_commit(
    *,
    manifest: dict[str, Any],
    git_commit_override: str,
) -> str:
    current = _normalize_git_commit(str(manifest.get("git_commit", "")))
    if current:
        return current
    return _normalize_git_commit(git_commit_override)


def _persist_policy_selection_in_manifest(
    *,
    layout: ArtifactLayout,
    manifest: dict[str, Any],
    policy_ids: list[str],
    selection_details: dict[str, Any],
) -> None:
    manifest["policy_set_selection"] = list(policy_ids)
    merged_details = dict(selection_details)
    merged_details.setdefault("status", "resolved")
    merged_details["resolved_by"] = "canonical_eval_pipeline_v1"
    merged_details["policy_count"] = len(policy_ids)
    manifest["policy_set_selection_details"] = merged_details
    _write_json(layout.manifest_path, manifest)


def _policy_selection_mode(selection_details: dict[str, Any]) -> str:
    return str(selection_details.get("mode", "")).strip().lower()


def _resolve_selection_inputs_from_manifest(
    *,
    stack_root: Path,
    manifest: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    details = manifest.get("policy_set_selection_details")
    if not isinstance(details, dict):
        return None, None
    source_paths = details.get("source_paths")
    if not isinstance(source_paths, dict):
        return None, None

    def _resolve(path_value: Any) -> Path | None:
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = stack_root / candidate
        return candidate

    return _resolve(source_paths.get("snapshot_registry_json")), _resolve(source_paths.get("dev_eval_summaries_json"))


def _default_dev_eval_summaries_path(layout: ArtifactLayout) -> Path | None:
    for candidate in (
        layout.training_logs_dir / "dev_eval_summaries.json",
        layout.training_logs_dir / "periodic_dev_eval_summaries.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _run_summary_marks_canonical_eval_completed(layout: ArtifactLayout) -> bool:
    if not layout.run_summary_path.is_file():
        return False
    try:
        run_summary = _load_json_object(layout.run_summary_path, label="run summary")
    except Exception:
        return False
    return bool(run_summary.get("canonical_eval_completed", False))


def _authoritative_manifest_policy_selection(
    *,
    manifest: dict[str, Any],
    layout: ArtifactLayout,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
) -> tuple[list[str], dict[str, Any]] | None:
    if snapshot_registry_path is not None or dev_eval_summaries_path is not None:
        return None

    manifest_policy_ids = manifest.get("policy_set_selection")
    if not isinstance(manifest_policy_ids, list):
        return None
    resolved_from_manifest = [str(policy_id).strip() for policy_id in manifest_policy_ids if str(policy_id).strip()]
    if not resolved_from_manifest:
        return None

    details = manifest.get("policy_set_selection_details")
    status = ""
    selection_details: dict[str, Any] = {}
    if isinstance(details, dict):
        selection_details = dict(details)
        status = str(details.get("status", "")).strip().lower()
    if status == "unresolved":
        return None
    if _policy_selection_mode(selection_details) == "explicit_cli":
        return None

    has_completed_eval_artifacts = bool(
        layout.final_eval_summary_json().is_file() or _run_summary_marks_canonical_eval_completed(layout)
    )
    if not has_completed_eval_artifacts:
        return None

    selection_details.setdefault("mode", "manifest_policy_set_selection")
    selection_details.setdefault("status", "resolved")
    selection_details["policy_count"] = len(resolved_from_manifest)
    return resolved_from_manifest, selection_details


def _resolve_policy_ids_for_run(
    *,
    policy_ids: list[str],
    stack: Any,
    manifest: dict[str, Any],
    layout: ArtifactLayout,
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
) -> tuple[list[str], dict[str, Any], Path | None, Path | None]:
    manifest_snapshot_registry, manifest_dev_eval = _resolve_selection_inputs_from_manifest(
        stack_root=stack.root,
        manifest=manifest,
    )
    resolved_snapshot_registry: Path | None = (
        snapshot_registry_path or manifest_snapshot_registry or (layout.training_snapshots_dir / "registry.json")
    )
    if resolved_snapshot_registry is None or not resolved_snapshot_registry.is_file():
        resolved_snapshot_registry = None
    resolved_dev_eval = dev_eval_summaries_path or manifest_dev_eval
    if resolved_dev_eval is None or not resolved_dev_eval.is_file():
        resolved_dev_eval = _default_dev_eval_summaries_path(layout)

    explicit_policy_ids = [policy_id.strip() for policy_id in policy_ids if policy_id.strip()]
    if explicit_policy_ids:
        return (
            explicit_policy_ids,
            {"mode": "explicit_cli", "policy_count": len(explicit_policy_ids)},
            resolved_snapshot_registry,
            resolved_dev_eval,
        )

    authoritative_manifest_selection = _authoritative_manifest_policy_selection(
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
    )
    if authoritative_manifest_selection is not None:
        resolved_from_manifest, selection_details = authoritative_manifest_selection
        return resolved_from_manifest, selection_details, resolved_snapshot_registry, resolved_dev_eval

    manifest_policy_ids = manifest.get("policy_set_selection")
    resolved_from_manifest = (
        [str(policy_id).strip() for policy_id in manifest_policy_ids if str(policy_id).strip()]
        if isinstance(manifest_policy_ids, list)
        else []
    )

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    if resolved_snapshot_registry is not None and resolved_dev_eval is not None:
        resolved = resolve_final_policy_set(
            snapshot_registry_path=resolved_snapshot_registry,
            dev_eval_summaries_path=resolved_dev_eval,
            config=evaluation.final_policy_set_selection,
            final_policy_set_size=evaluation.final_policy_set_size,
        )
        return (
            resolved,
            {
                "mode": "deterministic_v1",
                "policy_count": len(resolved),
                "snapshot_registry_path": resolved_snapshot_registry.as_posix(),
                "dev_eval_summaries_path": resolved_dev_eval.as_posix(),
                "final_policy_set_size": int(evaluation.final_policy_set_size),
            },
            resolved_snapshot_registry,
            resolved_dev_eval,
        )

    if resolved_from_manifest:
        return (
            resolved_from_manifest,
            {
                "mode": "manifest_policy_set_selection_fallback",
                "policy_count": len(resolved_from_manifest),
            },
            resolved_snapshot_registry,
            resolved_dev_eval,
        )

    if resolved_snapshot_registry is None:
        raise FileNotFoundError(
            "final policy-set resolution requires a snapshot registry; checked "
            f"{snapshot_registry_path or manifest_snapshot_registry or (layout.training_snapshots_dir / 'registry.json')}"
        )
    if resolved_dev_eval is None:
        checked_paths = [
            path.as_posix()
            for path in (
                dev_eval_summaries_path,
                manifest_dev_eval,
                layout.training_logs_dir / "dev_eval_summaries.json",
                layout.training_logs_dir / "periodic_dev_eval_summaries.json",
            )
            if path is not None
        ]
        raise FileNotFoundError(
            "final policy-set resolution requires dev-eval summaries; checked "
            + (", ".join(checked_paths) if checked_paths else "<none>")
        )
    raise AssertionError("policy-set resolution should have returned or raised before reaching this point")


def _update_run_level_reports(
    *,
    layout: ArtifactLayout,
    run_dir: Path,
    policy_ids: list[str],
    selection_details: dict[str, Any],
    final_eval_payload: dict[str, Any],
    metagame_payload: dict[str, Any] | None,
    figure_paths: tuple[Path, ...],
    readiness_payload: dict[str, Any] | None,
) -> None:
    run_summary = _load_run_summary_or_default(layout)
    run_summary.update(
        {
            "final_eval_dir": layout.relative(layout.final_eval_dir),
            "policy_ids": list(policy_ids),
            "policy_set_selection_mode": selection_details.get("mode", "unknown"),
            "metagame_dir": None if metagame_payload is None else layout.relative(layout.metagame_dir),
            "figure_outputs": [layout.relative(path) for path in figure_paths],
            "paper_readiness_summary_path": layout.relative(layout.paper_readiness_summary_path),
            "paper_grade": bool(readiness_payload and readiness_payload.get("passed", False)),
            "canonical_eval_completed": True,
        }
    )
    _write_json(layout.run_summary_path, run_summary)

    determinism_report = _load_determinism_report_or_default(layout)
    replay_verification = _load_json_object(layout.replay_verification_json(), label="replay verification summary")
    artifact_hashes = _load_json_object(layout.final_eval_aggregate_hashes_json(), label="final eval artifact hashes")
    determinism_report.update(
        {
            "run_dir": run_dir.as_posix(),
            "policy_selection_mode": selection_details.get("mode", "unknown"),
            "replay_verification": {
                "path": layout.relative(layout.replay_verification_json()),
                "status": replay_verification.get("status", "unknown"),
                "sampled_episode_count": replay_verification.get("sampled_episode_count", 0),
                "verified_episode_count": replay_verification.get("verified_episode_count", 0),
                "failed_episode_count": replay_verification.get("failed_episode_count", 0),
            },
            "canonical_artifact_hashes": dict(cast(dict[str, Any], artifact_hashes.get("artifacts", {}))),
            "final_eval": {
                "path": layout.relative(layout.final_eval_summary_json()),
                "policy_ids": list(policy_ids),
                "selection": dict(selection_details),
                "matchup_count": len(cast(list[Any], final_eval_payload.get("matchups", []))),
            },
        }
    )
    _write_json(layout.determinism_report_path, determinism_report)


def _run_canonical_eval_pipeline(
    *,
    parser: argparse.ArgumentParser,
    stack: Any,
    run_dir: Path,
    final_eval_dir: Path | None,
    policy_ids: list[str],
    snapshot_registry_path: Path | None,
    dev_eval_summaries_path: Path | None,
    b1_baseline_run_dir: Path | None,
    bootstrap_samples: int,
    paired_seed_limit: int | None,
    stage1_paired_seeds: int | None,
    max_paired_seeds: int | None,
    skip_metagame: bool,
    study_config_path: Path | None,
    skip_figures: bool,
    skip_readiness: bool,
    git_commit_override: str,
) -> int:
    layout = ArtifactLayout.from_run_dir(run_dir)
    layout.ensure_directories()
    tensorboard_logger = TensorBoardLogger(layout.tensorboard_dir)
    if final_eval_dir is not None and final_eval_dir.resolve() != layout.final_eval_dir.resolve():
        parser.error(
            f"--final-eval-dir must match the canonical run directory layout for non-demo runs: {layout.final_eval_dir}"
        )

    manifest = _load_json_object(layout.manifest_path, label="run manifest")
    effective_git_commit = _effective_manifest_git_commit(
        manifest=manifest,
        git_commit_override=git_commit_override,
    )
    if effective_git_commit:
        manifest_git_commit = _normalize_git_commit(str(manifest.get("git_commit", "")))
        if manifest_git_commit:
            print(f"Eval provenance git commit: {manifest_git_commit}")
        else:
            print(f"Eval provenance git commit override (not persisted): {effective_git_commit}")
    run_id256 = str(manifest.get("run_id256", "")).strip()
    if len(run_id256) != 64:
        raise ValueError(f"run manifest is missing a valid run_id256: {layout.manifest_path}")

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")
    study_config = None
    if not skip_metagame:
        resolved_study_config = (
            (stack.root / "configs" / "study" / "metagame_sensitivity.yaml")
            if study_config_path is None
            else study_config_path.resolve()
        )
        study_config = load_study_config(resolved_study_config)

    (
        resolved_policy_ids,
        selection_details,
        resolved_registry_path,
        resolved_dev_eval_path,
    ) = _resolve_policy_ids_for_run(
        policy_ids=policy_ids,
        stack=stack,
        manifest=manifest,
        layout=layout,
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
    )
    _persist_policy_selection_in_manifest(
        layout=layout,
        manifest=manifest,
        policy_ids=resolved_policy_ids,
        selection_details=selection_details,
    )

    contract = load_verified_simulator_contract(
        stack.root,
        expected_spec_hash=str(manifest.get("spec_hash256", "")).strip(),
    )
    observation_dim = int(contract.spec_bundle["observation"]["obs_len"])
    action_dim = int(contract.spec_bundle["action"]["action_space_size"])
    pass_action_id = int(contract.spec_bundle["action"]["pass_action_id"])
    resolved_policies = resolve_eval_policies(
        stack=stack,
        policy_ids=resolved_policy_ids,
        run_dir=run_dir,
        observation_dim=observation_dim,
        action_dim=action_dim,
        spec_bundle=contract.spec_bundle,
        snapshot_registry_path=resolved_registry_path,
        b1_baseline_run_dir=b1_baseline_run_dir,
    )
    runner = SimulatorEvalRunner(
        stack=stack,
        policies=resolved_policies,
        artifact_layout=layout,
        run_id256=run_id256,
        spec_hash256=str(manifest["spec_hash256"]),
        action_dim=action_dim,
        pass_action_id=pass_action_id,
        require_sorted_legal_ids=bool(evaluation.eval_assert_sorted_legal_ids),
        replay_capture_rate=float(evaluation.replay_capture_rate_eval),
        regression_capture_count=int(evaluation.regression_capture_count),
    )

    seed_file_path = stack.seed_sets["report_eval"]
    all_paired_seeds = parse_seed_file(seed_file_path)
    if paired_seed_limit is not None:
        all_paired_seeds = all_paired_seeds[: int(paired_seed_limit)]
    if not all_paired_seeds:
        raise ValueError(f"report_eval seed file produced no usable seeds: {seed_file_path}")

    resolved_stage1 = int(
        stage1_paired_seeds or min(evaluation.final_matrix_stage1_paired_seeds, len(all_paired_seeds))
    )
    resolved_max = int(
        max_paired_seeds or min(evaluation.final_matrix_stage2_adaptive_max_paired_seeds, len(all_paired_seeds))
    )
    if resolved_stage1 > resolved_max:
        raise ValueError(f"stage1 paired seeds ({resolved_stage1}) cannot exceed max paired seeds ({resolved_max})")

    recommended_focal_policy_id = None
    if resolved_registry_path is not None and resolved_dev_eval_path is not None:
        try:
            from weiss_rl.league.registry import SnapshotRegistry

            recommended_focal_policy_id = recommend_focal_policy_id(
                snapshot_registry=SnapshotRegistry.load(resolved_registry_path),
                dev_eval_summaries=load_dev_eval_summaries(resolved_dev_eval_path),
                candidate_policy_ids=resolved_policy_ids,
            )
        except Exception:
            recommended_focal_policy_id = None

    try:
        if not tensorboard_logger.enabled:
            unavailable_reason = tensorboard_unavailable_reason()
            print(
                "TensorBoard logging is disabled for eval: "
                + ("SummaryWriter unavailable" if unavailable_reason is None else unavailable_reason),
                file=sys.stderr,
            )
        else:
            tensorboard_logger.log_text("eval/run/manifest", manifest)

        final_eval_payload = run_final_eval(
            output_dir=layout.final_eval_dir,
            runner=runner,
            paired_seeds=all_paired_seeds,
            stage1_paired_seeds=resolved_stage1,
            max_paired_seeds=resolved_max,
            stop_rules=evaluation.stop_rules,
            run_id256=run_id256,
            config_hash256=str(manifest["config_hash256"]),
            spec_hash256=str(manifest["spec_hash256"]),
            scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
            sample_count=int(bootstrap_samples),
            policy_ids=resolved_policy_ids,
            snapshot_registry_path=resolved_registry_path,
            dev_eval_summaries_path=resolved_dev_eval_path,
            selection_config=evaluation.final_policy_set_selection,
            final_policy_set_size=int(evaluation.final_policy_set_size),
            metadata={
                "pipeline": {
                    "kind": "canonical_eval_pipeline_v1",
                    "selection": dict(selection_details),
                    "seed_file": seed_file_path.as_posix(),
                    "paired_seed_limit": None if paired_seed_limit is None else int(paired_seed_limit),
                },
                "recommended_focal_policy_id": recommended_focal_policy_id,
            },
            seed_file_path=seed_file_path,
        )

        metagame_payload: dict[str, Any] | None = None
        if not skip_metagame:
            assert study_config is not None
            metagame_payload = build_sensitivity_report(
                final_eval_dir=layout.final_eval_dir,
                out_dir=layout.metagame_dir,
                metagame_config=study_config.metagame,
                sensitivity_config=study_config.sensitivity,
            )

        figure_paths: tuple[Path, ...] = ()
        if not skip_figures:
            figure_paths = render_paper_figures(run_dir)

        _ensure_run_level_report_scaffolding(layout)

        readiness_payload: dict[str, Any] | None = None
        if not skip_readiness:
            readiness_payload = build_paper_readiness_summary(
                run_dir=run_dir,
                focal_policy_id=recommended_focal_policy_id,
            )
            write_paper_readiness_json(layout.paper_readiness_summary_path, readiness_payload)

        _update_run_level_reports(
            layout=layout,
            run_dir=run_dir,
            policy_ids=resolved_policy_ids,
            selection_details=selection_details,
            final_eval_payload=final_eval_payload,
            metagame_payload=metagame_payload,
            figure_paths=figure_paths,
            readiness_payload=readiness_payload,
        )

        if tensorboard_logger.enabled:
            tensorboard_logger.log_final_eval_summary(final_eval_payload, step=0)
            if metagame_payload is not None:
                tensorboard_logger.log_metagame_summary(metagame_payload, metagame_dir=layout.metagame_dir, step=0)
            if readiness_payload is not None:
                tensorboard_logger.log_paper_readiness(readiness_payload, step=0)

        print(f"Canonical final_eval summary JSON: {layout.final_eval_summary_json()}")
        print(f"Canonical replay verification JSON: {layout.replay_verification_json()}")
        if metagame_payload is not None:
            print(f"Canonical metagame summary JSON: {layout.metagame_dir / 'summary.json'}")
        if figure_paths:
            print(f"Rendered {len(figure_paths)} paper figure files to {layout.figures_paper_dir}")
        if readiness_payload is not None:
            print(f"Paper readiness summary JSON: {layout.paper_readiness_summary_path}")
            print("Paper readiness: " + ("passed" if bool(readiness_payload.get("passed", False)) else "failed"))
        print(f"Resolved policy set: {resolved_policy_ids}")
        return 0
    finally:
        tensorboard_logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation entrypoint for canonical final_eval or summary-only reports"
    )
    parser.add_argument(
        "--stack-config",
        type=Path,
        required=True,
        help="Path to the stack config used for contract checks and evaluation settings",
    )
    parser.add_argument(
        "--spec-hash",
        type=str,
        default="",
        help="Expected compatibility spec hash or full spec bundle SHA-256 for contract validation",
    )
    parser.add_argument("--config-hash", type=str, default="", help="Config hash for contract validation")
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional startup banner/log label only; not persisted in summary exports",
    )
    parser.add_argument("--run-id", dest="run_id_alias", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--public-demo",
        action="store_true",
        help="Run the built-in public-safe toy final-eval path instead of canonical simulator-backed evaluation.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Canonical run directory for simulator-backed evaluation or staged public-demo artifacts",
    )
    parser.add_argument(
        "--final-eval-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for public-demo artifacts, or the canonical "
            "<run-dir>/eval/final_eval path for non-demo runs"
        ),
    )
    parser.add_argument(
        "--policy-id",
        action="append",
        default=None,
        help="Explicit policy ID to evaluate in canonical non-demo mode (repeatable)",
    )
    parser.add_argument(
        "--snapshot-registry-json",
        type=Path,
        default=None,
        help="Optional snapshot registry JSON for deterministic policy-set resolution in canonical non-demo mode",
    )
    parser.add_argument(
        "--dev-eval-summaries-json",
        type=Path,
        default=None,
        help="Optional dev-eval summaries JSON for deterministic policy-set resolution in canonical non-demo mode",
    )
    parser.add_argument(
        "--b1-baseline-run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing the real B1 no-league baseline artifacts when the selected policy set includes B1"
        ),
    )
    parser.add_argument(
        "--paired-seed-limit",
        type=int,
        default=None,
        help="Optional cap on the number of report_eval paired seeds used in canonical non-demo mode",
    )
    parser.add_argument(
        "--stage1-paired-seeds",
        type=int,
        default=None,
        help="Optional override for stage-1 paired seeds in canonical non-demo mode",
    )
    parser.add_argument(
        "--max-paired-seeds",
        type=int,
        default=None,
        help="Optional override for stage-2 max paired seeds in canonical non-demo mode",
    )
    parser.add_argument(
        "--skip-metagame",
        action="store_true",
        help="Skip metagame sensitivity generation in canonical non-demo mode",
    )
    parser.add_argument(
        "--study-config",
        type=Path,
        default=None,
        help="Optional study-only metagame/sensitivity config (defaults to configs/study/metagame_sensitivity.yaml)",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip paper figure rendering in canonical non-demo mode",
    )
    parser.add_argument(
        "--skip-readiness",
        action="store_true",
        help="Skip paper-readiness auditing in canonical non-demo mode",
    )
    parser.add_argument(
        "--git-commit-override",
        type=str,
        default="",
        help="Optional 40-hex git commit shown in eval logs when manifest provenance is missing; never persisted",
    )
    parser.add_argument(
        "--public-demo-paired-seeds",
        type=int,
        default=PUBLIC_DEMO_DEFAULT_PAIRED_SEEDS,
        help="Paired seed count for public-demo final_eval generation",
    )
    parser.add_argument(
        "--public-demo-bootstrap-samples",
        type=int,
        default=PUBLIC_DEMO_DEFAULT_BOOTSTRAP_SAMPLES,
        help="Bootstrap sample count for public-demo final_eval generation",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=None,
        help="Existing seat-swapped episodes.jsonl to summarize (no rollout generation)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Output path for summary JSON export in summary-only mode",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output path for summary CSV export in summary-only mode",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=None,
        help="Output path for seat diagnostics JSON export in summary-only mode",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Bootstrap sample count for uncertainty",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap RNG seed for summary-only mode")
    args = parser.parse_args()
    run_label = _resolve_run_label(parser, args.run_label, args.run_id_alias)

    _require_positive_int(parser, "--bootstrap-samples", args.bootstrap_samples)
    _require_positive_int(parser, "--public-demo-paired-seeds", args.public_demo_paired_seeds)
    _require_positive_int(parser, "--public-demo-bootstrap-samples", args.public_demo_bootstrap_samples)
    paired_seed_limit = _require_positive_int(parser, "--paired-seed-limit", args.paired_seed_limit)
    stage1_paired_seeds = _require_positive_int(parser, "--stage1-paired-seeds", args.stage1_paired_seeds)
    max_paired_seeds = _require_positive_int(parser, "--max-paired-seeds", args.max_paired_seeds)

    if args.public_demo:
        if args.run_dir is None:
            parser.error("--public-demo requires --run-dir")
        if args.episodes_jsonl is not None:
            parser.error("--public-demo cannot be combined with --episodes-jsonl")
    elif not args.skip_readiness and (args.skip_metagame or args.skip_figures):
        parser.error("--skip-metagame or --skip-figures requires --skip-readiness")
    elif args.run_dir is not None and args.episodes_jsonl is not None:
        parser.error("--run-dir cannot be combined with --episodes-jsonl outside --public-demo mode")
    elif args.episodes_jsonl is None and (
        args.summary_json is not None or args.summary_csv is not None or args.diagnostics_json is not None
    ):
        parser.error("--summary-json/--summary-csv/--diagnostics-json require --episodes-jsonl")

    stack = load_stack_config(args.stack_config)
    config_hash256 = compute_config_hash256(stack)
    _require_matching_hash(
        flag_name="--config-hash",
        expected=_expected_sha256(args.config_hash, flag_name="--config-hash"),
        actual=config_hash256,
    )

    policy = "hard_fail"

    contract = None
    if args.public_demo:
        public_demo_bundle = public_demo_spec_bundle()
        assert_spec_bundle_contract(args.spec_hash, public_demo_bundle)
        reported_spec_hash = public_demo_spec_hash256()
    else:
        contract = load_verified_simulator_contract(stack.root, expected_spec_hash=args.spec_hash)
        reported_spec_hash = contract.spec_hash256
    print_startup_banner(
        reported_spec_hash,
        config_hash256,
        run_label=run_label,
        spec_mismatch_policy=policy,
    )
    if contract is not None:
        print(
            "Verified runtime spec bundle: "
            f"compat={contract.simulator.get('compatibility_hash', '')} sha256={contract.spec_hash256}"
        )
    elif args.public_demo:
        print(
            "Verified public-demo spec bundle: "
            f"compat={public_demo_spec_bundle()['spec_hash']} sha256={reported_spec_hash}"
        )

    if args.public_demo:
        assert args.run_dir is not None
        run_dir = args.run_dir.resolve()
        final_eval_dir = args.final_eval_dir or (run_dir / "eval" / "final_eval")
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing run manifest: {manifest_path}")
        manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
        run_id256 = str(manifest.get("run_id256", ""))
        if not run_id256:
            raise ValueError(f"run manifest is missing run_id256: {manifest_path}")
        evaluation = stack.config.evaluation
        stop_rules = public_demo_stop_rules() if evaluation is None else evaluation.stop_rules
        payload = run_public_demo_final_eval(
            output_dir=final_eval_dir,
            run_dir=run_dir,
            paired_seed_file=stack.seed_sets["report_eval"],
            paired_seed_limit=int(args.public_demo_paired_seeds),
            sample_count=int(args.public_demo_bootstrap_samples),
            run_id256=run_id256,
            config_hash256=config_hash256,
            spec_hash256=reported_spec_hash,
            stop_rules=stop_rules,
        )
        print(f"Public-demo final_eval summary JSON: {final_eval_dir / 'summary.json'}")
        print(f"Public-demo policies: {payload['policy_ids']}")
        print(
            "Public demo evaluation completed. These artifacts are toy/demo only and do not represent thesis results."
        )
        return

    if args.run_dir is not None:
        raise SystemExit(
            _run_canonical_eval_pipeline(
                parser=parser,
                stack=stack,
                run_dir=args.run_dir.resolve(),
                final_eval_dir=None if args.final_eval_dir is None else args.final_eval_dir.resolve(),
                policy_ids=list(args.policy_id or ()),
                snapshot_registry_path=None
                if args.snapshot_registry_json is None
                else args.snapshot_registry_json.resolve(),
                dev_eval_summaries_path=None
                if args.dev_eval_summaries_json is None
                else args.dev_eval_summaries_json.resolve(),
                b1_baseline_run_dir=None if args.b1_baseline_run_dir is None else args.b1_baseline_run_dir.resolve(),
                bootstrap_samples=int(args.bootstrap_samples),
                paired_seed_limit=paired_seed_limit,
                stage1_paired_seeds=stage1_paired_seeds,
                max_paired_seeds=max_paired_seeds,
                skip_metagame=bool(args.skip_metagame),
                study_config_path=None if args.study_config is None else args.study_config.resolve(),
                skip_figures=bool(args.skip_figures),
                skip_readiness=bool(args.skip_readiness),
                git_commit_override=str(args.git_commit_override),
            )
        )

    if args.episodes_jsonl is None:
        print(f"Evaluation contract check complete; no episodes were summarized. Seed sets: {sorted(stack.seed_sets)}")
        return

    evaluation = stack.config.evaluation
    if evaluation is None:
        raise ValueError("stack config is missing evaluation settings")

    records = load_eval_game_records(args.episodes_jsonl)
    payload = build_matchup_export(
        records,
        stop_rules=evaluation.stop_rules,
        max_paired_seeds=evaluation.final_matrix_stage2_adaptive_max_paired_seeds,
        scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
        sample_count=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    summary_json = args.summary_json or args.episodes_jsonl.with_suffix(".summary.json")
    summary_csv = args.summary_csv or args.episodes_jsonl.with_suffix(".summary.csv")
    write_matchup_summary_json(summary_json, payload)
    write_matchup_summary_csv(summary_csv, payload)

    print(f"Evaluation summary JSON: {summary_json}")
    print(f"Evaluation summary CSV: {summary_csv}")
    print("Evaluation reports were derived from a pre-recorded episodes file; no rollouts were executed here.")

    if args.diagnostics_json is not None:
        diagnostics_payload = build_seat_advantage_diagnostics(records)
        write_matchup_diagnostics_json(args.diagnostics_json, diagnostics_payload)
        print(f"Evaluation diagnostics JSON: {args.diagnostics_json}")


if __name__ == "__main__":
    main()
