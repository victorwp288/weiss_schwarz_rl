from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from weiss_rl.config import apply_stack_overrides, compute_config_hash256, load_stack_config
from weiss_rl.tests._config_paths import canonical_stack_config_path, repo_root
from weiss_rl.training import startup


def _complete_stack() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            environment=SimpleNamespace(),
            training=SimpleNamespace(),
            model=SimpleNamespace(),
            system=SimpleNamespace(
                learner_device="cpu",
                profile=SimpleNamespace(local_iteration="balanced"),
            ),
            reproducibility=SimpleNamespace(seed_derivation=SimpleNamespace(base_seed64=1234)),
        )
    )


def _runtime_module(*, version: str = "1.2.0", rl: object | None = None, omit: tuple[str, ...] = ()) -> SimpleNamespace:
    values = {
        "__version__": version,
        "fast": object(),
        "inspect": object(),
        "make_pool": object(),
        "EnvPoolBuffers": object(),
        "export_spec_bundle": object(),
        "OBS_LEN": 378,
        "ACTION_SPACE_SIZE": 527,
        "SPEC_HASH": 8590000130,
        "PASS_ACTION_ID": 51,
        "rl": (
            rl
            if rl is not None
            else SimpleNamespace(
                reset_rl=object(),
                step_rl=object(),
                step_rl_sample_from_logits=object(),
                step_rl_sample_from_logits_with_logp=object(),
            )
        ),
    }
    for name in omit:
        values.pop(name, None)
    return SimpleNamespace(**values)


def test_startup_resolves_profile_device_and_seed_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _complete_stack()

    assert startup.resolve_runtime_profile(stack, "") == "balanced"
    assert startup.resolve_runtime_profile(stack, "fast") == "fast"
    assert startup.resolve_seed(stack, None) == 1234
    assert startup.resolve_seed(stack, 99) == 99

    monkeypatch.setattr(startup.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(startup.torch.cuda, "device_count", lambda: 2)
    stack.config.system.learner_device = "cuda:auto"

    assert startup.resolve_device(stack, "") == torch.device("cuda:0")


def test_apply_training_flag_overrides_returns_same_stack_without_training_block() -> None:
    stack = SimpleNamespace(config=SimpleNamespace(training=None))

    assert (
        startup.apply_training_flag_overrides(
            stack,
            enable_profile_timers=True,
            enable_torch_profiler=True,
        )
        is stack
    )


def test_apply_training_flag_overrides_updates_flags_before_hashing() -> None:
    stack = load_stack_config(canonical_stack_config_path())
    base_hash = compute_config_hash256(stack)

    updated = startup.apply_training_flag_overrides(
        stack,
        enable_profile_timers=True,
        enable_torch_profiler=True,
    )

    assert updated.config.training is not None
    assert updated.config.training.profile_timers is True
    assert updated.config.training.torch_profiler is True
    assert compute_config_hash256(updated) != base_hash


def test_apply_training_flag_overrides_is_noop_when_requested_flags_are_already_set() -> None:
    stack = load_stack_config(canonical_stack_config_path())
    updated = startup.apply_training_flag_overrides(
        stack,
        enable_profile_timers=True,
        enable_torch_profiler=True,
    )

    assert (
        startup.apply_training_flag_overrides(
            updated,
            enable_profile_timers=True,
            enable_torch_profiler=True,
        )
        is updated
    )


def test_manifest_scaffold_reason_lists_missing_required_blocks() -> None:
    stack = SimpleNamespace(
        config=SimpleNamespace(
            environment=None,
            training=SimpleNamespace(),
            model=None,
        )
    )

    assert startup.manifest_scaffold_only_reason(stack) == "missing config blocks: environment, model"
    assert startup.runtime_training_prerequisite_failure(stack) is None


def test_runtime_prerequisite_failure_reports_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _complete_stack()

    def _missing_module(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(startup.importlib, "import_module", _missing_module)

    assert (
        startup.runtime_training_prerequisite_failure(stack) == "weiss_sim is not importable in the active interpreter"
    )


def test_runtime_prerequisite_failure_reports_missing_stepping_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _complete_stack()
    missing_runtime = _runtime_module(omit=("rl", "PASS_ACTION_ID"))
    missing_rl_methods = _runtime_module(
        rl=SimpleNamespace(
            reset_rl=object(),
            step_rl=object(),
            step_rl_sample_from_logits=object(),
        )
    )

    monkeypatch.setattr(startup.importlib, "import_module", lambda name: missing_runtime)
    assert (
        startup.runtime_training_prerequisite_failure(stack)
        == "active weiss_sim runtime is missing stepping APIs: PASS_ACTION_ID, rl"
    )

    monkeypatch.setattr(startup.importlib, "import_module", lambda name: missing_rl_methods)
    assert (
        startup.runtime_training_prerequisite_failure(stack)
        == "active weiss_sim.rl is missing runtime methods: step_rl_sample_from_logits_with_logp"
    )


def test_runtime_prerequisite_failure_reports_old_simulator_version(monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _complete_stack()

    monkeypatch.setattr(startup.importlib, "import_module", lambda name: _runtime_module(version="0.8.1"))

    assert (
        startup.runtime_training_prerequisite_failure(stack) == "active weiss_sim version 0.8.1 is below required 1.2.0"
    )


def test_canonical_b1_noleague_prerequisites_accept_clean_model_actor_config() -> None:
    stack = load_stack_config(repo_root() / "configs" / "thesis" / "b1_noleague.yaml")

    assert startup.noleague_training_prerequisite_failure(stack) is None


def test_canonical_b1_noleague_prerequisites_accept_tighter_sync_probe() -> None:
    stack = load_stack_config(
        repo_root() / "configs" / "thesis" / "ablations" / "b1_noleague_tight_sync20_temp050.yaml"
    )

    assert startup.noleague_training_prerequisite_failure(stack) is None


def test_canonical_b1_noleague_prerequisites_reject_heuristic_actor_inheritance() -> None:
    stack = load_stack_config(repo_root() / "configs" / "thesis" / "b1_noleague.yaml")
    broken = apply_stack_overrides(
        stack,
        {
            "training.actor_policy_backend": "heuristic_public",
            "training.actor_heuristic_fraction": 1.0,
        },
    )

    assert startup.noleague_training_prerequisite_failure(broken) == (
        "B1 NoLeague must use training.actor_policy_backend=model"
    )


def test_canonical_b1_noleague_prerequisites_reject_inherited_opponent_sampling() -> None:
    stack = load_stack_config(repo_root() / "configs" / "thesis" / "b1_noleague.yaml")
    broken = apply_stack_overrides(
        stack,
        {
            "league.sampling.heuristic_public_mix_fraction": 1.0,
            "league.sampling.noleague_baseline_mix_fraction": 0.25,
        },
    )

    reason = startup.noleague_training_prerequisite_failure(broken)

    assert reason is not None
    assert "heuristic_public_mix_fraction=1" in reason
    assert "noleague_baseline_mix_fraction=0.25" in reason


def test_canonical_b1_noleague_prerequisites_reject_teacher_warmstart() -> None:
    stack = load_stack_config(repo_root() / "configs" / "thesis" / "b1_noleague.yaml")
    broken = apply_stack_overrides(
        stack,
        {
            "training.teacher_aux.mode": "warmstart_only",
            "training.structured_warmstart.enabled": True,
            "training.structured_warmstart.updates": 1,
        },
    )

    assert startup.noleague_training_prerequisite_failure(broken) == (
        "B1 NoLeague must set training.teacher_aux.mode=off"
    )

    warmstart_only = apply_stack_overrides(
        stack,
        {
            "training.structured_warmstart.enabled": True,
            "training.structured_warmstart.updates": 1,
        },
    )

    assert (
        startup.noleague_training_prerequisite_failure(warmstart_only)
        == "B1 NoLeague must disable training.structured_warmstart"
    )
