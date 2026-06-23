from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState


class SyntheticCanonicalLayout:
    def __init__(self, run_dir: Path) -> None:
        self.final_eval_dir = run_dir / "eval" / "final_eval"
        self.metagame_dir = run_dir / "eval" / "metagame"
        self.figures_paper_dir = run_dir / "figures" / "paper"
        self.paper_readiness_summary_path = run_dir / "paper_readiness_summary.json"

    def final_eval_summary_json(self) -> Path:
        return self.final_eval_dir / "summary.json"

    def replay_verification_json(self) -> Path:
        return self.final_eval_dir / "replay_verification.json"


def canonical_manifest() -> dict[str, str]:
    return {
        "run_id256": "ab" * 32,
        "config_hash256": "cd" * 32,
        "spec_hash256": "ef" * 32,
    }


def make_run_state(
    *,
    run_dir: Path,
    layout: object | None = None,
    evaluation: object | None = None,
    study_config: object | None = None,
    tensorboard_logger: object | None = None,
) -> CanonicalEvalRunState:
    return CanonicalEvalRunState(
        layout=layout or SyntheticCanonicalLayout(run_dir),
        tensorboard_logger=tensorboard_logger or SimpleNamespace(enabled=False),
        manifest=canonical_manifest(),
        run_id256="ab" * 32,
        evaluation=evaluation or SimpleNamespace(),
        study_config=study_config,
    )


def make_runtime_state(
    *,
    tmp_path: Path,
    policy_ids: list[str] | None = None,
    paired_seeds: list[int] | None = None,
    paired_seed_limit: int = 1,
    stage1_paired_seeds: int = 1,
    max_paired_seeds: int = 1,
    snapshot_registry_path: Path | None = None,
    dev_eval_summaries_path: Path | None = None,
    seed_file_path: Path | None = None,
    recommended_focal_policy_id: str = "policy_000100",
) -> CanonicalEvalRuntimeState:
    return CanonicalEvalRuntimeState(
        policy_ids=policy_ids or ["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=snapshot_registry_path,
        dev_eval_summaries_path=dev_eval_summaries_path,
        runner=object(),
        paired_seeds=paired_seeds or [101],
        paired_seed_limit=paired_seed_limit,
        stage1_paired_seeds=stage1_paired_seeds,
        max_paired_seeds=max_paired_seeds,
        seed_file_path=seed_file_path or tmp_path / "seeds.txt",
        recommended_focal_policy_id=recommended_focal_policy_id,
    )
