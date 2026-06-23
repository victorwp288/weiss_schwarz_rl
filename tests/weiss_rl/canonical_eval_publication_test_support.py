from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from weiss_rl.workflows.canonical_eval.state import CanonicalEvalRunState, CanonicalEvalRuntimeState
from weiss_rl.workflows.canonical_eval.supplemental_outputs import CanonicalEvalSupplementalOutputs


class FakeTensorBoardLogger:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.calls: list[tuple[str, object]] = []

    def log_text(self, tag: str, payload: object) -> None:
        self.calls.append(("text", (tag, payload)))

    def log_final_eval_summary(self, payload: object, *, step: int) -> None:
        self.calls.append(("final", (payload, step)))

    def log_metagame_summary(self, payload: object, *, metagame_dir: Path, step: int) -> None:
        self.calls.append(("metagame", (payload, metagame_dir, step)))

    def log_paper_readiness(self, payload: object, *, step: int) -> None:
        self.calls.append(("readiness", (payload, step)))


def canonical_publication_layout(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        metagame_dir=tmp_path / "run" / "eval" / "metagame",
        figures_paper_dir=tmp_path / "run" / "figures" / "paper",
        paper_readiness_summary_path=tmp_path / "run" / "paper_readiness_summary.json",
        final_eval_summary_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "summary.json",
        replay_verification_json=lambda: tmp_path / "run" / "eval" / "final_eval" / "replay_verification.json",
    )


def canonical_runtime_state(tmp_path: Path) -> CanonicalEvalRuntimeState:
    return CanonicalEvalRuntimeState(
        policy_ids=["B0 RandomLegal", "policy_000100"],
        selection_details={"status": "resolved"},
        snapshot_registry_path=None,
        dev_eval_summaries_path=None,
        runner=object(),
        paired_seeds=[101],
        paired_seed_limit=1,
        stage1_paired_seeds=1,
        max_paired_seeds=1,
        seed_file_path=tmp_path / "seeds.txt",
        recommended_focal_policy_id="policy_000100",
    )


def canonical_run_state(
    *,
    layout: object,
    tensorboard_logger: object | None = None,
    manifest: dict[str, Any] | None = None,
    run_id256: str = "ab" * 32,
) -> CanonicalEvalRunState:
    return CanonicalEvalRunState(
        layout=layout,
        tensorboard_logger=tensorboard_logger or SimpleNamespace(),
        manifest={"run_id256": run_id256} if manifest is None else manifest,
        run_id256=run_id256,
        evaluation=SimpleNamespace(),
        study_config=None,
    )


def canonical_supplemental_outputs(tmp_path: Path) -> CanonicalEvalSupplementalOutputs:
    return CanonicalEvalSupplementalOutputs(
        metagame_payload={"meta": "payload"},
        figure_paths=(tmp_path / "run" / "figures" / "paper" / "seat_bias.pdf",),
        readiness_payload={"passed": True},
    )
