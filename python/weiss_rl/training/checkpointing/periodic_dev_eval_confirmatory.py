"""Confirmatory periodic dev-eval execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class PeriodicDevEvalEffectiveSummary:
    summary: Mapping[str, Any]
    confirmatory_request: Mapping[str, Any] | None = None
    confirmatory_pair_count: int = 0


def checkpoint_tracker_best_record(tracker_payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    existing_best_record = tracker_payload.get("best")
    if not isinstance(existing_best_record, Mapping):
        return None
    return cast(Mapping[str, Any], existing_best_record)


def maybe_run_confirmatory_dev_eval(
    *,
    hooks: Any,
    stack: Any,
    learner: Any,
    summary_payload: Mapping[str, Any],
    contract: Any,
    artifacts: Any,
    training_paths: Any,
    device: Any,
    run_id256: str,
    config_hash256: str,
    spec_hash256: str,
    update_count: int,
) -> PeriodicDevEvalEffectiveSummary:
    existing_best_record = checkpoint_tracker_best_record(hooks.load_checkpoint_tracker(training_paths))
    confirmatory_request = hooks.confirmatory_dev_eval_request(
        stack=stack,
        existing_best_record=existing_best_record,
        dev_eval_summary=summary_payload,
    )
    if confirmatory_request is None:
        return PeriodicDevEvalEffectiveSummary(summary=summary_payload)

    seed_file, _validated_sources, base_paired_seeds, seed_file_sha256 = hooks.periodic_dev_eval_schedule(stack)
    confirmatory_pairs = hooks.expand_periodic_dev_eval_paired_seeds(
        base_paired_seeds,
        requested_pairs=int(confirmatory_request["target_pairs"]),
        seed_file_sha256=seed_file_sha256,
        update_count=update_count,
        policy_version=int(learner.get_policy_version()),
        scope="periodic_dev_eval_confirmatory",
    )
    effective_summary = hooks.run_periodic_dev_eval(
        stack=stack,
        contract=contract,
        artifacts=artifacts,
        training_paths=training_paths,
        learner=learner,
        device=device,
        run_id256=run_id256,
        config_hash256=config_hash256,
        spec_hash256=spec_hash256,
        artifact_dir_name="dev_eval_confirmatory",
        artifact_scope="periodic_dev_eval_confirmatory",
        paired_seeds_override=confirmatory_pairs,
        persist_summary=False,
        update_stall_monitor=False,
    )
    print(
        "Confirmatory dev eval: "
        f"update={learner.update_count} paired_seeds={len(confirmatory_pairs)} "
        f"aggregate={effective_summary['aggregate_score']:.4f} "
        f"reasons={','.join(cast(list[str], confirmatory_request['reasons']))} "
        f"seed_file={seed_file.name}"
    )
    return PeriodicDevEvalEffectiveSummary(
        summary=cast(Mapping[str, Any], effective_summary),
        confirmatory_request=confirmatory_request,
        confirmatory_pair_count=len(confirmatory_pairs),
    )


__all__ = [
    "PeriodicDevEvalEffectiveSummary",
    "checkpoint_tracker_best_record",
    "maybe_run_confirmatory_dev_eval",
]
