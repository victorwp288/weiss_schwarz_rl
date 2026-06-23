from __future__ import annotations

from .entrypoints_eval_selection_test_support import (
    EXPECTED_DETERMINISTIC_POLICY_IDS,
    prepare_policy_selection_run,
)
from .entrypoints_test_support import Path, json


def test_eval_pipeline_persists_policy_selection_before_run_final_eval(tmp_path: Path, monkeypatch) -> None:
    from weiss_rl.workflows import eval_entrypoint as eval_script

    stack, layout = prepare_policy_selection_run(tmp_path, "eval_pipeline_persist_before_final_eval")
    run_dir = layout.run_dir
    layout.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    layout.manifest_path.write_text(
        json.dumps(
            {
                "run_id256": "ab" * 32,
                "config_hash256": "cd" * 32,
                "spec_hash256": "ef" * 32,
                "policy_set_selection": [],
                "policy_set_selection_details": {
                    "status": "unresolved",
                    "reason": "selection_pending",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakeTensorBoardLogger:
        enabled = False

        def __init__(self, _log_dir: Path) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeContract:
        spec_bundle = {
            "observation": {"obs_len": 512},
            "action": {"action_space_size": 9, "pass_action_id": 8},
        }

    observed: dict[str, dict[str, object]] = {}

    def _fake_run_final_eval(**_kwargs: object) -> dict[str, object]:
        observed["manifest"] = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        raise RuntimeError("stop after manifest check")

    monkeypatch.setattr(eval_script, "TensorBoardLogger", _FakeTensorBoardLogger)
    monkeypatch.setattr(
        eval_script,
        "load_verified_simulator_contract",
        lambda *_args, **_kwargs: _FakeContract(),
    )
    monkeypatch.setattr(eval_script, "resolve_eval_policies", lambda **_kwargs: [])
    monkeypatch.setattr(eval_script, "SimulatorEvalRunner", lambda **_kwargs: object())
    monkeypatch.setattr(eval_script, "run_final_eval", _fake_run_final_eval)

    try:
        eval_script._run_canonical_eval_pipeline(
            parser=eval_script.argparse.ArgumentParser(),
            stack=stack,
            run_dir=run_dir,
            final_eval_dir=None,
            policy_ids=[],
            snapshot_registry_path=None,
            dev_eval_summaries_path=None,
            b1_baseline_run_dir=None,
            bootstrap_samples=8,
            paired_seed_limit=1,
            stage1_paired_seeds=1,
            max_paired_seeds=1,
            skip_metagame=True,
            study_config_path=None,
            skip_figures=True,
            skip_readiness=True,
            git_commit_override="",
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after manifest check"
    else:
        raise AssertionError("expected fake run_final_eval to stop the pipeline")

    persisted = observed["manifest"]
    assert persisted["policy_set_selection"] == EXPECTED_DETERMINISTIC_POLICY_IDS
    assert persisted["policy_set_selection_details"] == {
        "mode": "deterministic_v1",
        "policy_count": len(EXPECTED_DETERMINISTIC_POLICY_IDS),
        "resolved_by": "canonical_eval_pipeline_v1",
        "snapshot_registry_path": (layout.training_snapshots_dir / "registry.json").as_posix(),
        "dev_eval_summaries_path": (layout.training_logs_dir / "periodic_dev_eval_summaries.json").as_posix(),
        "final_policy_set_size": 10,
        "status": "resolved",
    }
