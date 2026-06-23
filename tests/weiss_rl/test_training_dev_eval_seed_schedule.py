from __future__ import annotations

from types import SimpleNamespace

import pytest
from weiss_rl.artifacts.reproducibility import hash_seed_file
from weiss_rl.training.dev_eval import (
    periodic_dev_eval_schedule,
    periodic_dev_eval_seed_usage_payload,
    resolve_periodic_dev_eval_seed_file,
)

from .training_dev_eval_test_support import make_dev_eval_stack


def test_periodic_dev_eval_schedule_validates_sources_and_hashes_seed_file(tmp_path) -> None:
    stack = make_dev_eval_stack(tmp_path)

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
    stack = make_dev_eval_stack(tmp_path)
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
