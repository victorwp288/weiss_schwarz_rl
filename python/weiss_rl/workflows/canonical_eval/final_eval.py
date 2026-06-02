from __future__ import annotations

from typing import Any, cast

from weiss_rl.eval.payoff_folding import PayoffFoldScheme
from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState


def run_canonical_final_eval_output(
    *,
    bootstrap_samples: int,
    run_state: CanonicalEvalRunState,
    runtime_state: CanonicalEvalRuntimeState,
    dependencies: Any,
) -> dict[str, Any]:
    layout = run_state.layout
    evaluation = run_state.evaluation
    manifest = run_state.manifest
    return dependencies.run_final_eval_fn(
        output_dir=layout.final_eval_dir,
        runner=runtime_state.runner,
        paired_seeds=runtime_state.paired_seeds,
        stage1_paired_seeds=runtime_state.stage1_paired_seeds,
        max_paired_seeds=runtime_state.max_paired_seeds,
        stop_rules=evaluation.stop_rules,
        run_id256=run_state.run_id256,
        config_hash256=str(manifest["config_hash256"]),
        spec_hash256=str(manifest["spec_hash256"]),
        scheme=cast(PayoffFoldScheme, evaluation.final_policy_set_selection.folding),
        sample_count=int(bootstrap_samples),
        policy_ids=runtime_state.policy_ids,
        snapshot_registry_path=runtime_state.snapshot_registry_path,
        dev_eval_summaries_path=runtime_state.dev_eval_summaries_path,
        selection_config=evaluation.final_policy_set_selection,
        final_policy_set_size=int(evaluation.final_policy_set_size),
        metadata={
            "pipeline": {
                "kind": "canonical_eval_pipeline_v1",
                "selection": dict(runtime_state.selection_details),
                "seed_file": runtime_state.seed_file_path.as_posix(),
                "paired_seed_limit": runtime_state.paired_seed_limit,
            },
            "recommended_focal_policy_id": runtime_state.recommended_focal_policy_id,
        },
        seed_file_path=runtime_state.seed_file_path,
    )


__all__ = ["run_canonical_final_eval_output"]
