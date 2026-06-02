from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from weiss_rl.experiments.toy_public_demo import public_demo_stop_rules, run_public_demo_final_eval


def run_public_demo_eval_mode(
    *,
    stack: Any,
    run_dir: Path,
    final_eval_dir: Path | None,
    paired_seed_limit: int,
    bootstrap_samples: int,
    config_hash256: str,
    spec_hash256: str,
    public_demo_stop_rules_fn: Any = public_demo_stop_rules,
    run_public_demo_final_eval_fn: Any = run_public_demo_final_eval,
) -> None:
    resolved_final_eval_dir = final_eval_dir or (run_dir / "eval" / "final_eval")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    manifest = cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))
    run_id256 = str(manifest.get("run_id256", ""))
    if not run_id256:
        raise ValueError(f"run manifest is missing run_id256: {manifest_path}")

    evaluation = stack.config.evaluation
    stop_rules = public_demo_stop_rules_fn() if evaluation is None else evaluation.stop_rules
    payload = run_public_demo_final_eval_fn(
        output_dir=resolved_final_eval_dir,
        run_dir=run_dir,
        paired_seed_file=stack.seed_sets["report_eval"],
        paired_seed_limit=int(paired_seed_limit),
        sample_count=int(bootstrap_samples),
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        stop_rules=stop_rules,
    )
    print(f"Public-demo final_eval summary JSON: {resolved_final_eval_dir / 'summary.json'}")
    print(f"Public-demo policies: {payload['policy_ids']}")
    print("Public demo evaluation completed. These artifacts are toy/demo only and do not represent thesis results.")


__all__ = ["run_public_demo_eval_mode"]
