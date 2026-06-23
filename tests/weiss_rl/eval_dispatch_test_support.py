from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
from weiss_rl.workflows.eval_support.eval_startup import EvalStartup, EvalValidatedArgs


def seed_stack(tmp_path: Path, **seed_sets: Path) -> SimpleNamespace:
    return SimpleNamespace(seed_sets=seed_sets or {"report_eval": tmp_path / "seeds.txt"})


def validated_args(
    run_label: str,
    *,
    paired_seed_limit: int | None = None,
    stage1_paired_seeds: int | None = None,
    max_paired_seeds: int | None = None,
) -> EvalValidatedArgs:
    return EvalValidatedArgs(
        run_label=run_label,
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
    )


def runtime_startup(stack: object, *, config_hash: str = "c", spec_hash: str = "e") -> EvalStartup:
    expanded_config_hash = config_hash * 64
    expanded_spec_hash = spec_hash * 64
    return EvalStartup(
        stack=stack,
        config_hash256=expanded_config_hash,
        reported_spec_hash=expanded_spec_hash,
        contract=SimpleNamespace(
            simulator={"compatibility_hash": "compat123"},
            spec_hash256=expanded_spec_hash,
        ),
    )


def public_demo_startup(stack: object) -> EvalStartup:
    return EvalStartup(
        stack=stack,
        config_hash256="c" * 64,
        reported_spec_hash="d" * 64,
        contract=None,
    )


def fail_route(name: str):
    return lambda **_kwargs: (_ for _ in ()).throw(AssertionError(f"{name} should not run"))


def dispatch_dependencies(**overrides: Any) -> EvalDispatchDependencies:
    defaults: dict[str, Any] = {
        "public_demo_spec_bundle_fn": lambda: {"spec_hash": "public_demo"},
        "public_demo_stop_rules_fn": None,
        "run_public_demo_final_eval_fn": None,
        "run_public_demo_eval_mode_fn": fail_route("public demo"),
        "run_canonical_eval_pipeline_fn": fail_route("canonical mode"),
        "run_summary_only_eval_mode_fn": fail_route("summary mode"),
        "load_eval_game_records_fn": None,
        "build_matchup_export_fn": None,
        "build_seat_advantage_diagnostics_fn": None,
        "write_matchup_diagnostics_json_fn": None,
        "write_matchup_summary_csv_fn": None,
        "write_matchup_summary_json_fn": None,
    }
    defaults.update(overrides)
    return EvalDispatchDependencies(**defaults)


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()
