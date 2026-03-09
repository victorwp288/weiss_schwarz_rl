from __future__ import annotations

import json

from weiss_rl.manifest import (
    DEFAULT_DISCOUNT_GAMMA,
    DEFAULT_ENV_WRAPPER,
    DEFAULT_REWARD_MODE,
    DEFAULT_REWARD_PERSPECTIVE,
    DEFAULT_STEP_DEFINITION,
    RunManifest,
    make_smoke_run_manifest,
)


def test_make_smoke_run_manifest_uses_canonical_family_a_defaults() -> None:
    manifest = make_smoke_run_manifest(
        run_id="run_123",
        stack_config="configs/rl_stack_locked.yaml",
        spec_hash="spec",
        config_hash="config",
        component_count=10,
        seed_set_count=3,
    )

    assert manifest.env_wrapper == DEFAULT_ENV_WRAPPER
    assert manifest.step_definition == DEFAULT_STEP_DEFINITION
    assert manifest.reward_mode == DEFAULT_REWARD_MODE
    assert manifest.reward_perspective == DEFAULT_REWARD_PERSPECTIVE
    assert manifest.discount_gamma == DEFAULT_DISCOUNT_GAMMA
    assert manifest.note == "Smoke run: config loading only (no training executed)."


def test_run_manifest_write_json_round_trips(tmp_path) -> None:
    manifest = RunManifest(run_id="run_456", note="ok")
    out_path = tmp_path / "manifest.json"

    manifest.write_json(out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_456"
    assert payload["note"] == "ok"
    assert payload["created_at_utc"]
