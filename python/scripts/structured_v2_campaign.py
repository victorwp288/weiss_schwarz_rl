from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from weiss_rl.experiments.structured_acceptance import build_structured_baseline_contract

_DEFAULT_SEEDS = (20260212, 20260213, 20260214)
_DEFAULT_B2_ANCHOR = "B2 HeuristicPublic"


def _workspace_root(repo_root: Path) -> Path:
    return repo_root.parent


def _summary_path(*, workspace_root: Path, campaign_label: str) -> Path:
    return workspace_root / "runs" / "launch_groups" / campaign_label / "summary.json"


def _train_script(repo_root: Path) -> Path:
    return repo_root / "python" / "scripts" / "train.py"


def _audit_script(repo_root: Path) -> Path:
    return repo_root / "python" / "scripts" / "b2_disagreement_audit.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the structured_v2 baseline->canary->B2-audit campaign for a fixed seed set"
    )
    parser.add_argument("--stack-config", type=Path, required=True)
    parser.add_argument("--campaign-label", type=str, required=True)
    parser.add_argument("--run-label-prefix", type=str, default="")
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--unroll-length", type=int, default=64)
    parser.add_argument("--baseline-max-updates", type=int, default=120)
    parser.add_argument("--canary-max-updates", type=int, default=120)
    parser.add_argument("--runtime-mode", type=str, default="train_async_fast")
    parser.add_argument("--profile", type=str, default="fast")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--frozen-baseline-run-dir",
        type=Path,
        default=Path(
            r"C:\Users\Bruger\Desktop\thesis-repo\weiss_schwarz_rl\runs\local_typed_gpu_mainmovefix_canary_u80"
        ),
    )
    parser.add_argument(
        "--frozen-audit-summary",
        type=Path,
        default=Path(
            r"C:\Users\Bruger\Desktop\thesis-repo\weiss_schwarz_rl\runs\audit_b2_mainmovefix_u300_v3\audit\summary.json"
        ),
    )
    parser.add_argument("--frozen-baseline-update", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _run_label(*, prefix: str, kind: str, seed: int, updates: int) -> str:
    return f"{prefix}_{kind}_seed{int(seed)}_u{int(updates)}"


def _repo_run_dir(repo_root: Path, run_label: str) -> Path:
    return repo_root / "runs" / run_label


def _command_env(*, repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    py_paths = [
        str(repo_root / "python"),
        str(_workspace_root(repo_root) / "weiss-schwarz-simulator" / "python"),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        py_paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    return env


def _write_summary(summary_file: Path, summary: dict[str, Any]) -> None:
    summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_step(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    dry_run: bool,
    summary_steps: list[dict[str, Any]],
    label: str,
    summary: dict[str, Any],
    summary_file: Path,
    seed: int,
) -> None:
    payload = {
        "label": label,
        "command": command,
        "cwd": cwd.as_posix(),
        "status": "planned" if dry_run else "running",
        "seed": int(seed),
    }
    summary["current_seed"] = int(seed)
    summary["current_stage"] = label
    summary_steps.append(payload)
    _write_summary(summary_file, summary)
    print(" ".join(command), flush=True)
    if dry_run:
        return
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = repr(exc)
        summary["status"] = "failed"
        summary["failed_step"] = label
        summary["failed_seed"] = int(seed)
        summary["error"] = repr(exc)
        _write_summary(summary_file, summary)
        raise
    payload["status"] = "completed"
    summary["last_completed_step"] = label
    _write_summary(summary_file, summary)


def _freeze_baseline_contract(
    *,
    baseline_run_dir: Path,
    baseline_update: int,
    audit_summary_path: Path,
) -> tuple[Path, dict[str, Any]]:
    dev_eval_summary_path = baseline_run_dir / "eval" / "dev_eval" / f"update_{int(baseline_update)}" / "summary.json"
    contract = build_structured_baseline_contract(
        baseline_run_dir=baseline_run_dir,
        baseline_update=int(baseline_update),
        dev_eval_summary=json.loads(dev_eval_summary_path.read_text(encoding="utf-8")),
        audit_summary_path=audit_summary_path,
        audit_summary=json.loads(audit_summary_path.read_text(encoding="utf-8")),
    )
    out_path = baseline_run_dir / "structured_v2" / "baseline_contract.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = contract.to_dict()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path, payload


def _find_counter(items: list[dict[str, Any]], **match: str) -> int:
    for item in items:
        if all(str(item.get(key, "")) == value for key, value in match.items()):
            return int(item.get("count", 0))
    return 0


def _focal_policy_id_from_dev_eval(summary: dict[str, Any]) -> str:
    anchors = summary.get("anchors", {})
    if isinstance(anchors, dict):
        b2 = anchors.get(_DEFAULT_B2_ANCHOR)
        if isinstance(b2, dict):
            focal = str(b2.get("focal_policy_id", "")).strip()
            if focal:
                return focal
        for value in anchors.values():
            if isinstance(value, dict):
                focal = str(value.get("focal_policy_id", "")).strip()
                if focal:
                    return focal
    raise RuntimeError("Could not resolve focal_policy_id from dev eval summary")


def _u120_acceptance_payload(
    *,
    baseline_contract: dict[str, Any],
    dev_eval_summary: dict[str, Any],
    audit_summary: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    targets = dict(baseline_contract.get("acceptance_targets", {}).get("u120", {}))
    anchor_scores = dict(dev_eval_summary.get("anchor_scores", {}))
    top_family_pairs = list(audit_summary.get("top_family_pairs", ()))
    top_action_pairs = list(audit_summary.get("top_action_label_pairs", ()))
    mismatch_counts = {
        "main_move_to_pass": _find_counter(
            top_family_pairs,
            policy_a_family="main_move",
            policy_b_family="pass",
        ),
        "main_move_to_main_play_character": _find_counter(
            top_family_pairs,
            policy_a_family="main_move",
            policy_b_family="main_play_character",
        ),
    }
    top_exact_pair = dict(top_action_pairs[0]) if top_action_pairs else {}
    expected_top = dict(targets.get("forbid_top_exact_pair", {}))
    checks = {
        "b2_nonzero": float(anchor_scores.get(_DEFAULT_B2_ANCHOR, 0.0)) > 0.0,
        "main_move_to_pass_reduced": mismatch_counts["main_move_to_pass"]
        <= int(targets.get("max_main_move_to_pass", 0)),
        "main_move_to_main_play_character_reduced": mismatch_counts["main_move_to_main_play_character"]
        <= int(targets.get("max_main_move_to_main_play_character", 0)),
        "top_exact_pair_changed": not (
            expected_top
            and str(top_exact_pair.get("policy_a_action_label", ""))
            == str(expected_top.get("policy_a_action_label", ""))
            and str(top_exact_pair.get("policy_b_action_label", ""))
            == str(expected_top.get("policy_b_action_label", ""))
        ),
    }
    return {
        "kind": "structured_v2_u120_acceptance_v1",
        "seed": int(seed),
        "aggregate_score": float(dev_eval_summary.get("aggregate_score", 0.0)),
        "anchor_scores": {str(key): float(value) for key, value in anchor_scores.items()},
        "mismatch_counts": mismatch_counts,
        "top_exact_pair": top_exact_pair,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "baseline_targets": targets,
    }


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2] if args.repo_root is None else Path(args.repo_root).resolve()
    python_exe = sys.executable
    workspace_root = _workspace_root(repo_root)
    summary_file = _summary_path(workspace_root=workspace_root, campaign_label=str(args.campaign_label))
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in (args.seed or list(_DEFAULT_SEEDS)))
    run_label_prefix = str(args.run_label_prefix).strip() or str(args.campaign_label).strip()
    baseline_contract_path, baseline_contract = _freeze_baseline_contract(
        baseline_run_dir=args.frozen_baseline_run_dir.resolve(),
        baseline_update=int(args.frozen_baseline_update),
        audit_summary_path=args.frozen_audit_summary.resolve(),
    )

    env = _command_env(repo_root=repo_root)
    steps: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "kind": "structured_v2_campaign_v1",
        "campaign_label": str(args.campaign_label),
        "repo_root": repo_root.as_posix(),
        "workspace_root": workspace_root.as_posix(),
        "stack_config": str(args.stack_config.resolve()),
        "baseline_contract_path": baseline_contract_path.as_posix(),
        "baseline_contract": baseline_contract,
        "dry_run": bool(args.dry_run),
        "seeds": list(seeds),
        "steps": steps,
        "status": "running" if not args.dry_run else "planned",
        "current_seed": None,
        "current_stage": None,
        "last_completed_step": None,
    }
    _write_summary(summary_file, summary)

    for seed in seeds:
        baseline_run_label = _run_label(
            prefix=run_label_prefix,
            kind="b1",
            seed=seed,
            updates=int(args.baseline_max_updates),
        )
        canary_run_label = _run_label(
            prefix=run_label_prefix,
            kind="canary",
            seed=seed,
            updates=int(args.canary_max_updates),
        )
        baseline_run_dir = _repo_run_dir(repo_root, baseline_run_label)
        canary_run_dir = _repo_run_dir(repo_root, canary_run_label)

        baseline_command = [
            python_exe,
            str(_train_script(repo_root)),
            "--stack-config",
            str(args.stack_config.resolve()),
            "--run-label",
            baseline_run_label,
            "--seed",
            str(seed),
            "--num-envs",
            str(int(args.num_envs)),
            "--unroll-length",
            str(int(args.unroll_length)),
            "--max-updates",
            str(int(args.baseline_max_updates)),
            "--runtime-mode",
            str(args.runtime_mode),
            "--profile",
            str(args.profile),
            "--device",
            str(args.device),
            "--override",
            'experiment.role="baseline_noleague"',
            "--override",
            "league.enabled=false",
        ]
        _run_step(
            command=baseline_command,
            cwd=repo_root,
            env=env,
            dry_run=bool(args.dry_run),
            summary_steps=steps,
            label=f"baseline_seed_{seed}",
            summary=summary,
            summary_file=summary_file,
            seed=seed,
        )

        canary_command = [
            python_exe,
            str(_train_script(repo_root)),
            "--stack-config",
            str(args.stack_config.resolve()),
            "--run-label",
            canary_run_label,
            "--seed",
            str(seed),
            "--num-envs",
            str(int(args.num_envs)),
            "--unroll-length",
            str(int(args.unroll_length)),
            "--max-updates",
            str(int(args.canary_max_updates)),
            "--runtime-mode",
            str(args.runtime_mode),
            "--profile",
            str(args.profile),
            "--device",
            str(args.device),
            "--b1-baseline-run-dir",
            str(baseline_run_dir),
        ]
        _run_step(
            command=canary_command,
            cwd=repo_root,
            env=env,
            dry_run=bool(args.dry_run),
            summary_steps=steps,
            label=f"canary_seed_{seed}",
            summary=summary,
            summary_file=summary_file,
            seed=seed,
        )

        audit_output_run_dir = _repo_run_dir(
            repo_root,
            f"{canary_run_label}_audit_b2_u{int(args.canary_max_updates)}",
        )
        dev_eval_summary_path = (
            canary_run_dir / "eval" / "dev_eval" / f"update_{int(args.canary_max_updates)}" / "summary.json"
        )
        episodes_jsonl = (
            canary_run_dir
            / "eval"
            / "dev_eval"
            / f"update_{int(args.canary_max_updates)}"
            / _DEFAULT_B2_ANCHOR
            / "episodes.jsonl"
        )
        if not args.dry_run:
            dev_eval_summary = json.loads(dev_eval_summary_path.read_text(encoding="utf-8"))
            focal_policy_id = _focal_policy_id_from_dev_eval(dev_eval_summary)
        else:
            dev_eval_summary = {}
            focal_policy_id = "train_u120_pX"
        audit_command = [
            python_exe,
            str(_audit_script(repo_root)),
            "--stack-config",
            str(args.stack_config.resolve()),
            "--run-dir",
            str(canary_run_dir),
            "--output-run-dir",
            str(audit_output_run_dir),
            "--episodes-jsonl",
            str(episodes_jsonl),
            "--policy-id",
            focal_policy_id,
        ]
        _run_step(
            command=audit_command,
            cwd=repo_root,
            env=env,
            dry_run=bool(args.dry_run),
            summary_steps=steps,
            label=f"audit_seed_{seed}",
            summary=summary,
            summary_file=summary_file,
            seed=seed,
        )
        if not args.dry_run:
            audit_summary_path = audit_output_run_dir / "audit" / "summary.json"
            acceptance_payload = _u120_acceptance_payload(
                baseline_contract=baseline_contract,
                dev_eval_summary=dev_eval_summary,
                audit_summary=json.loads(audit_summary_path.read_text(encoding="utf-8")),
                seed=seed,
            )
            acceptance_path = canary_run_dir / "structured_v2" / "u120_acceptance.json"
            acceptance_path.parent.mkdir(parents=True, exist_ok=True)
            acceptance_path.write_text(
                json.dumps(acceptance_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            steps.append(
                {
                    "label": f"acceptance_seed_{seed}",
                    "status": "completed",
                    "seed": int(seed),
                    "acceptance_path": acceptance_path.as_posix(),
                    "pass": bool(acceptance_payload["pass"]),
                }
            )
            summary["last_completed_step"] = f"acceptance_seed_{seed}"
            _write_summary(summary_file, summary)

    summary["status"] = "completed" if not args.dry_run else "planned"
    summary["current_stage"] = None
    summary["current_seed"] = None
    _write_summary(summary_file, summary)
    print(f"Wrote campaign summary: {summary_file}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
