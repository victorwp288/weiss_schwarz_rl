from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weiss_rl.artifacts.reproducibility import hash_seed_file
from weiss_rl.config.models import ModelConfig, ModelDropoutConfig
from weiss_rl.eval.harness import GameResult
from weiss_rl.model import PolicyValueModel
from weiss_rl.training.dev_eval import (
    clone_cpu_eval_model,
    legal_ids_for_env_row,
    periodic_dev_eval_bootstrap_seed,
    periodic_dev_eval_rng_seed,
    periodic_dev_eval_schedule,
    periodic_dev_eval_seed_usage_payload,
    periodic_dev_eval_summaries_path,
    persist_periodic_dev_eval_summary,
    promotion_gate_bootstrap_seed,
    promotion_gate_rng_seed,
    resolve_periodic_dev_eval_seed_file,
    should_run_periodic_dev_eval,
    stall_monitor_state_path,
    update_stall_monitor,
    validate_periodic_dev_eval_contract,
)
from weiss_rl.training.periodic_dev_eval_run import run_periodic_dev_eval
from weiss_rl.training.script_entrypoint_hooks import run_periodic_dev_eval_with_script_hooks


def _stack(tmp_path, *, seed_file_name: str = "dev_seeds.txt", required_pairs: int = 2):
    seed_file = tmp_path / seed_file_name
    seed_file.write_text("11\n22\n33\n", encoding="utf-8")
    return SimpleNamespace(
        root=tmp_path,
        seed_sets={"dev_eval": seed_file},
        config=SimpleNamespace(
            evaluation=SimpleNamespace(
                eval_sampling_algorithm="pinned_cdf_pcg_v1",
                eval_device="cpu",
                eval_inference_mode=True,
                seat_swap=True,
                model_sampling_temperature=1.0,
                seed_files={"dev_eval": seed_file.name},
                periodic_dev_eval_paired_seeds=required_pairs,
                periodic_dev_eval_interval_updates=20,
            ),
            reproducibility=SimpleNamespace(seed_files={"dev_eval": seed_file.name}),
        ),
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        gru_hidden_size=8,
        encoder_mlp_width=8,
        encoder_mlp_layers=1,
        layer_norm=False,
        dropout=ModelDropoutConfig(family_a=0.0, ablation=0.0),
    )


def test_periodic_dev_eval_schedule_validates_sources_and_hashes_seed_file(tmp_path) -> None:
    stack = _stack(tmp_path)

    seed_file, sources = resolve_periodic_dev_eval_seed_file(stack)
    scheduled_seed_file, scheduled_sources, paired_seeds, seed_hash = periodic_dev_eval_schedule(stack)

    assert seed_file == tmp_path / "dev_seeds.txt"
    assert sources == {
        "stack.seed_sets.dev_eval": "dev_seeds.txt",
        "evaluation.seed_files.dev_eval": "dev_seeds.txt",
        "reproducibility.seed_files.dev_eval": "dev_seeds.txt",
    }
    assert scheduled_seed_file == seed_file
    assert scheduled_sources == sources
    assert paired_seeds == [11, 22]
    assert seed_hash == hash_seed_file(seed_file)


def test_periodic_dev_eval_schedule_rejects_mismatched_seed_sources(tmp_path) -> None:
    stack = _stack(tmp_path)
    other_seed_file = tmp_path / "other_seeds.txt"
    other_seed_file.write_text("11\n22\n", encoding="utf-8")
    stack.config.evaluation.seed_files["dev_eval"] = other_seed_file.name

    with pytest.raises(RuntimeError, match="Periodic dev eval seed file mismatch"):
        resolve_periodic_dev_eval_seed_file(stack)


def test_periodic_dev_eval_seed_usage_payload_preserves_artifact_contract(tmp_path) -> None:
    seed_file = tmp_path / "seeds" / "dev_eval.txt"
    seed_file.parent.mkdir()
    seed_file.write_text("11\n22\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "current"
    checkpoint_path = run_dir / "checkpoints" / "current.pt"
    evaluation = SimpleNamespace(
        seat_swap=True,
        eval_device="cpu",
        eval_inference_mode=True,
        eval_sampling_algorithm="pinned_cdf_pcg_v1",
        model_sampling_temperature=0.25,
        eval_assert_sorted_legal_ids=True,
    )

    payload = periodic_dev_eval_seed_usage_payload(
        seed_file=seed_file,
        seed_file_root=tmp_path,
        seed_file_sha256="abc123",
        validated_sources={"stack.seed_sets.dev_eval": "seeds/dev_eval.txt"},
        artifact_scope="periodic_dev_eval_confirmatory",
        scheduled_paired_seeds=[11, 22],
        paired_seeds=[11, 22, 33],
        evaluation=evaluation,
        focal_policy_id="policy_000010",
        update_count=10,
        policy_version=3,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        opponent_policy_id="b0_randomlegal",
        opponent_display_name="B0 RandomLegal",
    )

    assert payload == {
        "seed_set": "dev_eval",
        "seed_file": {
            "path": "seeds/dev_eval.txt",
            "sha256": "abc123",
            "validated_sources": {"stack.seed_sets.dev_eval": "seeds/dev_eval.txt"},
        },
        "artifact_scope": "periodic_dev_eval_confirmatory",
        "seed_schedule": {
            "configured_paired_seed_count": 2,
            "requested_paired_seed_count": 3,
            "expanded_beyond_seed_file": True,
        },
        "paired_seed_count": 3,
        "paired_seeds": [11, 22, 33],
        "protocol": {
            "seat_swap": True,
            "eval_device": "cpu",
            "eval_inference_mode": True,
            "eval_sampling_algorithm": "pinned_cdf_pcg_v1",
            "model_sampling_temperature": 0.25,
            "eval_assert_sorted_legal_ids": True,
        },
        "focal_policy": {
            "policy_id": "policy_000010",
            "update_count": 10,
            "policy_version": 3,
            "checkpoint_path": "checkpoints/current.pt",
        },
        "opponent_policy": {
            "policy_id": "b0_randomlegal",
            "display_name": "B0 RandomLegal",
        },
    }


def test_legal_ids_for_env_row_slices_packed_legality_rows() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([2, 4, 9, 11, 13], dtype=np.uint32),
            np.asarray([0, 2, 5], dtype=np.int64),
        )
    )

    row = legal_ids_for_env_row(batch=batch, env_index=1, require_sorted=True)

    assert row.dtype == np.uint32
    assert row.tolist() == [9, 11, 13]


def test_legal_ids_for_env_row_rejects_missing_offsets() -> None:
    batch = SimpleNamespace(ids_offsets=None)

    with pytest.raises(RuntimeError, match="Expected ids_offsets legality during periodic dev eval"):
        legal_ids_for_env_row(batch=batch, env_index=0, require_sorted=False)


def test_legal_ids_for_env_row_can_enforce_sorted_ids() -> None:
    batch = SimpleNamespace(
        ids_offsets=(
            np.asarray([5, 3], dtype=np.uint32),
            np.asarray([0, 2], dtype=np.int64),
        )
    )

    assert legal_ids_for_env_row(batch=batch, env_index=0, require_sorted=False).tolist() == [5, 3]
    with pytest.raises(ValueError, match="strictly increasing"):
        legal_ids_for_env_row(batch=batch, env_index=0, require_sorted=True)


def test_periodic_dev_eval_contract_preserves_public_failures(tmp_path) -> None:
    stack = _stack(tmp_path)
    validate_periodic_dev_eval_contract(stack)
    stack.config.evaluation.eval_device = "cuda"

    with pytest.raises(RuntimeError, match="evaluation.eval_device='cpu'"):
        validate_periodic_dev_eval_contract(stack)


def test_periodic_dev_eval_contract_accepts_model_argmax_sampling(tmp_path) -> None:
    stack = _stack(tmp_path)
    stack.config.evaluation.eval_sampling_algorithm = "model_argmax_pinned_v1"

    validate_periodic_dev_eval_contract(stack)


def test_periodic_and_promotion_rng_seed_helpers_are_stable_and_distinct() -> None:
    scheduled_game = SimpleNamespace(
        pair_index=3,
        swap_index=1,
        episode_seed=123456,
        seat0_policy_id="a",
        seat1_policy_id="b",
    )

    first = periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=0)
    assert periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=0) == first
    assert periodic_dev_eval_rng_seed(scheduled_game=scheduled_game, seat=1) != first
    assert promotion_gate_rng_seed(scheduled_game=scheduled_game, seat=0) != first
    assert periodic_dev_eval_bootstrap_seed(update_count=10, policy_version=2) != promotion_gate_bootstrap_seed(
        update_count=10,
        policy_version=2,
    )


def test_periodic_dev_eval_paths_and_interval_predicate(tmp_path) -> None:
    stack = _stack(tmp_path)
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")

    assert should_run_periodic_dev_eval(stack, update_count=40)
    assert not should_run_periodic_dev_eval(stack, update_count=41)
    stack.config.evaluation.periodic_dev_eval_interval_updates = 0
    assert not should_run_periodic_dev_eval(stack, update_count=40)
    assert periodic_dev_eval_summaries_path(paths) == tmp_path / "logs" / "periodic_dev_eval_summaries.json"
    assert stall_monitor_state_path(paths) == tmp_path / "logs" / "stall_monitor.json"


def test_clone_cpu_eval_model_copies_weights_guidance_and_eval_mode(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del tmp_path
    stack = SimpleNamespace(config=SimpleNamespace(model=_model_config()))
    learner_model = PolicyValueModel(observation_dim=4, action_dim=3, config=stack.config.model)
    learner_model.train()
    guidance_calls: list[tuple[PolicyValueModel, dict[str, float]]] = []
    guidance_payload = {
        "public_heuristic_logit_bias_scale": 0.25,
        "public_heuristic_actor_logit_bias_scale": 0.75,
    }
    monkeypatch.setattr(
        "weiss_rl.training.dev_eval.model_guidance_payload",
        lambda model: guidance_payload if model is learner_model else {},
    )
    monkeypatch.setattr(
        "weiss_rl.training.dev_eval.restore_model_guidance_from_payload",
        lambda model, payload: guidance_calls.append((model, dict(payload))),
    )

    clone = clone_cpu_eval_model(
        learner_model=learner_model,
        observation_dim=4,
        action_dim=3,
        stack=stack,
    )

    assert clone.training is False
    assert guidance_calls == [(clone, guidance_payload)]
    for source, copied in zip(learner_model.parameters(), clone.parameters(), strict=True):
        assert copied.device.type == "cpu"
        assert torch.equal(copied, source.detach().cpu())
        assert copied.data_ptr() != source.detach().cpu().data_ptr()


def test_clone_cpu_eval_model_requires_model_config() -> None:
    learner_model = PolicyValueModel(observation_dim=4, action_dim=3, config=_model_config())
    stack = SimpleNamespace(config=SimpleNamespace(model=None))

    with pytest.raises(RuntimeError, match="missing the model config block"):
        clone_cpu_eval_model(
            learner_model=learner_model,
            observation_dim=4,
            action_dim=3,
            stack=stack,
        )


def test_persist_periodic_dev_eval_summary_uses_policy_id_as_key(tmp_path) -> None:
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")

    persist_periodic_dev_eval_summary(
        training_paths=paths,
        payload={
            "policy_id": "policy_000010",
            "aggregate_score": 0.625,
            "anchor_scores": {"B0 RandomLegal": 0.75},
            "update_count": 10,
            "policy_version": 3,
        },
    )
    persist_periodic_dev_eval_summary(
        training_paths=paths,
        payload={"policy_id": "", "aggregate_score": 1.0},
    )

    payload = json.loads((paths.logs_dir / "periodic_dev_eval_summaries.json").read_text(encoding="utf-8"))
    assert payload == {
        "policy_000010": {
            "aggregate_score": 0.625,
            "anchor_scores": {"B0 RandomLegal": 0.75},
            "update_count": 10,
            "policy_version": 3,
        }
    }


def test_update_stall_monitor_tracks_consecutive_stall_risk(tmp_path) -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            curriculum=SimpleNamespace(
                stall_monitor=SimpleNamespace(
                    enabled=True,
                    truncation_rate_threshold=0.25,
                    consecutive_evals=2,
                )
            )
        )
    )
    paths = SimpleNamespace(logs_dir=tmp_path / "logs")
    summary = {
        "anchors": {
            "B0 RandomLegal": {
                "summary": {
                    "games": 4,
                    "truncations": 1,
                    "no_progress_timeouts": 0,
                    "natural_timeouts": 0,
                }
            },
            "B2 HeuristicPublic": {
                "summary": {
                    "games": 4,
                    "truncations": 0,
                    "no_progress_timeouts": 2,
                    "natural_timeouts": 1,
                }
            },
        }
    }

    first = update_stall_monitor(stack=stack, training_paths=paths, update_count=20, summary_payload=summary)
    second = update_stall_monitor(stack=stack, training_paths=paths, update_count=40, summary_payload=summary)

    assert first is not None
    assert first["stall_risk"] is False
    assert first["consecutive_trigger_count"] == 1
    assert first["worst_anchor"] == "B2 HeuristicPublic"
    assert first["stall_indicator_kind"] == "no_progress_timeout"
    assert second is not None
    assert second["stall_risk"] is True
    assert second["consecutive_trigger_count"] == 2
    persisted = json.loads((paths.logs_dir / "stall_monitor.json").read_text(encoding="utf-8"))
    assert persisted == second


def test_run_periodic_dev_eval_writes_heuristic_policy_alignment_diagnostics(tmp_path) -> None:
    stack = _stack(tmp_path, required_pairs=1)
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
    assert result["anchors"]["B2 HeuristicPublic"]["policy_alignment_diagnostics"]["schema"] == (
        "policy_alignment_diagnostics_v1"
    )
    assert "policy_alignment_diagnostics" not in random_payload


def test_run_periodic_dev_eval_hook_accepts_new_and_legacy_stall_monitor_flag_names() -> None:
    captured: list[dict[str, object]] = []

    class Api:
        _PeriodicDevEvalRunner = object()
        _ensure_current_checkpoint = object()
        _current_focal_policy_id = object()
        _spec_dimensions = object()
        _clone_cpu_eval_model = object()
        _periodic_dev_eval_opponents = object()
        _persist_periodic_dev_eval_summary = object()
        _update_stall_monitor = object()
        _write_json = object()

        @staticmethod
        def run_periodic_dev_eval(**kwargs):
            captured.append(kwargs)
            return {"ok": True}

    base_kwargs = {
        "stack": object(),
        "contract": object(),
        "artifacts": object(),
        "training_paths": object(),
        "learner": object(),
        "device": torch.device("cpu"),
        "run_id256": "0" * 64,
        "config_hash256": "1" * 64,
        "spec_hash256": "2" * 64,
    }

    run_periodic_dev_eval_with_script_hooks(Api, **base_kwargs, update_stall_monitor=False)
    run_periodic_dev_eval_with_script_hooks(Api, **base_kwargs, update_stall_monitor_enabled=False)

    assert captured[0]["update_stall_monitor_enabled"] is False
    assert captured[1]["update_stall_monitor_enabled"] is False
