from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from weiss_rl.workflows.eval_support.eval_dispatch_dependencies import EvalDispatchDependencies
from weiss_rl.workflows.eval_support.eval_startup_state import EvalStartup, EvalValidatedArgs


@dataclass(frozen=True, slots=True)
class EvalDispatchRequest:
    parser: argparse.ArgumentParser
    args: Any
    validated: EvalValidatedArgs
    startup: EvalStartup
    dependencies: EvalDispatchDependencies

    @property
    def is_public_demo(self) -> bool:
        return bool(self.args.public_demo)

    @property
    def has_run_dir(self) -> bool:
        return self.args.run_dir is not None

    @property
    def has_episodes_jsonl(self) -> bool:
        return self.args.episodes_jsonl is not None

    def public_demo_kwargs(self) -> dict[str, Any]:
        assert self.args.run_dir is not None
        return {
            "stack": self.startup.stack,
            "run_dir": self.args.run_dir.resolve(),
            "final_eval_dir": None if self.args.final_eval_dir is None else self.args.final_eval_dir.resolve(),
            "paired_seed_limit": int(self.args.public_demo_paired_seeds),
            "bootstrap_samples": int(self.args.public_demo_bootstrap_samples),
            "config_hash256": self.startup.config_hash256,
            "spec_hash256": self.startup.reported_spec_hash,
            "public_demo_stop_rules_fn": self.dependencies.public_demo_stop_rules_fn,
            "run_public_demo_final_eval_fn": self.dependencies.run_public_demo_final_eval_fn,
        }

    def canonical_kwargs(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "stack": self.startup.stack,
            "run_dir": self.args.run_dir.resolve(),
            "final_eval_dir": None if self.args.final_eval_dir is None else self.args.final_eval_dir.resolve(),
            "policy_ids": list(self.args.policy_id or ()),
            "snapshot_registry_path": None
            if self.args.snapshot_registry_json is None
            else self.args.snapshot_registry_json.resolve(),
            "dev_eval_summaries_path": None
            if self.args.dev_eval_summaries_json is None
            else self.args.dev_eval_summaries_json.resolve(),
            "b1_baseline_run_dir": None
            if self.args.b1_baseline_run_dir is None
            else self.args.b1_baseline_run_dir.resolve(),
            "bootstrap_samples": int(self.args.bootstrap_samples),
            "paired_seed_limit": self.validated.paired_seed_limit,
            "stage1_paired_seeds": self.validated.stage1_paired_seeds,
            "max_paired_seeds": self.validated.max_paired_seeds,
            "skip_metagame": bool(self.args.skip_metagame),
            "study_config_path": None if self.args.study_config is None else self.args.study_config.resolve(),
            "skip_figures": bool(self.args.skip_figures),
            "skip_readiness": bool(self.args.skip_readiness),
            "git_commit_override": str(self.args.git_commit_override),
        }

    def summary_only_kwargs(self) -> dict[str, Any]:
        return {
            "stack": self.startup.stack,
            "episodes_jsonl": self.args.episodes_jsonl,
            "summary_json": self.args.summary_json,
            "summary_csv": self.args.summary_csv,
            "diagnostics_json": self.args.diagnostics_json,
            "bootstrap_samples": int(self.args.bootstrap_samples),
            "bootstrap_seed": int(self.args.bootstrap_seed),
            "load_eval_game_records_fn": self.dependencies.load_eval_game_records_fn,
            "build_matchup_export_fn": self.dependencies.build_matchup_export_fn,
            "build_seat_advantage_diagnostics_fn": self.dependencies.build_seat_advantage_diagnostics_fn,
            "write_matchup_diagnostics_json_fn": self.dependencies.write_matchup_diagnostics_json_fn,
            "write_matchup_summary_csv_fn": self.dependencies.write_matchup_summary_csv_fn,
            "write_matchup_summary_json_fn": self.dependencies.write_matchup_summary_json_fn,
        }


def eval_dispatch_request(
    *,
    parser: argparse.ArgumentParser,
    args: Any,
    validated: EvalValidatedArgs,
    startup: EvalStartup,
    dependencies: EvalDispatchDependencies,
) -> EvalDispatchRequest:
    return EvalDispatchRequest(
        parser=parser,
        args=args,
        validated=validated,
        startup=startup,
        dependencies=dependencies,
    )


__all__ = ["EvalDispatchRequest", "eval_dispatch_request"]
