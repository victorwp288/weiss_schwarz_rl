from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import weiss_rl.training.dev_eval as periodic_dev_eval
import weiss_rl.training.dev_eval.plan as periodic_dev_eval_plan
from weiss_rl.eval.simulator.harness import GameResult
from weiss_rl.training.periodic_dev_eval_run import run_periodic_dev_eval

from .training_dev_eval_test_support import make_dev_eval_stack


def test_run_periodic_dev_eval_writes_heuristic_policy_alignment_diagnostics(tmp_path) -> None:
    stack = make_dev_eval_stack(tmp_path, required_pairs=1)
    stack.config.training = SimpleNamespace(algorithm="impala")
    stack.config.evaluation.eval_assert_sorted_legal_ids = True
    stack.config.evaluation.stop_rules = SimpleNamespace(stop_delta_ci_half_width=0.01, stop_confidence=0.95)
    stack.config.evaluation.final_policy_set_selection = SimpleNamespace(folding="S2")
    artifacts = SimpleNamespace(run_dir=tmp_path / "runs" / "alignment")
    training_paths = SimpleNamespace(logs_dir=artifacts.run_dir / "training" / "logs")
    learner = SimpleNamespace(model=object(), update_count=25, get_policy_version=lambda: 2)
    checkpoint_path = artifacts.run_dir / "training" / "checkpoints" / "checkpoint_25.pt"

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            self._heuristic_policy = kwargs.get("heuristic_policy")

        def run_game(self, scheduled_game):
            return GameResult(
                episode_seed=scheduled_game.episode_seed,
                terminated=True,
                truncated=False,
                winner_seat=scheduled_game.focal_seat,
            )

        def policy_alignment_summary(self):
            if self._heuristic_policy is None:
                return None
            return {
                "schema": "policy_alignment_diagnostics_v1",
                "all_decisions": {"compared_steps": 7},
            }

    result = run_periodic_dev_eval(
        stack=stack,
        contract=SimpleNamespace(spec_bundle={"action": {"pass_action_id": 3}}),
        artifacts=artifacts,
        training_paths=training_paths,
        learner=learner,
        device=torch.device("cpu"),
        run_id256="0" * 64,
        config_hash256="1" * 64,
        spec_hash256="2" * 64,
        runner_cls=FakeRunner,
        ensure_current_checkpoint_fn=lambda **_kwargs: checkpoint_path,
        current_focal_policy_id_fn=lambda **_kwargs: "policy_000002",
        spec_dimensions_fn=lambda _contract: (4, 6),
        clone_cpu_eval_model_fn=lambda **_kwargs: object(),
        periodic_dev_eval_opponents_fn=lambda **_kwargs: [
            ("b0_randomlegal", "B0 RandomLegal", None, None),
            ("b2_heuristicpublic", "B2 HeuristicPublic", None, object()),
        ],
        persist_summary=False,
        update_stall_monitor_enabled=False,
    )

    heuristic_summary_path = (
        artifacts.run_dir / "eval" / "dev_eval" / "update_25" / "b2_heuristicpublic" / "matchup_summary.json"
    )
    random_summary_path = (
        artifacts.run_dir / "eval" / "dev_eval" / "update_25" / "b0_randomlegal" / "matchup_summary.json"
    )
    heuristic_payload = json.loads(heuristic_summary_path.read_text(encoding="utf-8"))
    random_payload = json.loads(random_summary_path.read_text(encoding="utf-8"))

    assert heuristic_payload["policy_alignment_diagnostics"]["all_decisions"]["compared_steps"] == 7
    assert result["periodic_dev_eval_plan"][0] == {
        "step_id": "validate_contract",
        "purpose": "Validate that dev-eval settings, seed sources, and runtime assumptions are usable.",
        "evidence": ["evaluation contract", "seed file", "validated seed sources"],
    }
    assert result["anchors"]["B2 HeuristicPublic"]["policy_alignment_diagnostics"]["schema"] == (
        "policy_alignment_diagnostics_v1"
    )
    assert "policy_alignment_diagnostics" not in random_payload


def test_periodic_dev_eval_plan_names_artifact_flow() -> None:
    payload = periodic_dev_eval.periodic_dev_eval_plan_payload()

    assert periodic_dev_eval.PERIODIC_DEV_EVAL_PLAN is periodic_dev_eval_plan.PERIODIC_DEV_EVAL_PLAN
    assert [step["step_id"] for step in payload] == [
        "validate_contract",
        "snapshot_eval_model",
        "resolve_anchor_panel",
        "run_matchups",
        "summarize_quality",
        "attach_diagnostics",
    ]
    assert "aggregate score" in payload[4]["evidence"]
