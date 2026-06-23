from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from weiss_rl.training.loop.setup import MinimalTrainingSetupHooks


class FakeModel:
    def __init__(self, recorder: MinimalSetupHookRecorder | None = None) -> None:
        self.recorder = recorder
        self.device: object | None = None

    def to(self, target_device: object) -> FakeModel:
        if self.recorder is not None:
            self.recorder.calls.append(("model_to", {"device": target_device}))
        self.device = target_device
        return self

    def state_dict(self) -> dict[str, object]:
        if self.recorder is not None:
            self.recorder.calls.append(("state_dict", {}))
        return {"weight": 1}


class FakeLearner:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.update_count = 0
        self.init_schedule_offset_updates = -1


class MinimalSetupHookRecorder:
    def __init__(
        self,
        *,
        training_paths: object,
        model: FakeModel,
        learner: FakeLearner,
        compiled_model: object | None = None,
        runtime_config: object | None = None,
        resume_state: object | None = None,
        init_state: object | None = None,
        config_hash256: str = "config-hash",
        canonical_config: dict[str, object] | None = None,
        snapshot_metrics: dict[str, float] | None = None,
        fail_restore: bool = False,
        fail_initialize: bool = False,
        fail_seed_snapshot: bool = False,
    ) -> None:
        self.training_paths = training_paths
        self.model = model
        self.learner = learner
        self.compiled_model = object() if compiled_model is None else compiled_model
        self.runtime_config = object() if runtime_config is None else runtime_config
        self.resume_state = resume_state
        self.init_state = init_state
        self.config_hash256 = config_hash256
        self.canonical_config = {} if canonical_config is None else canonical_config
        self.snapshot_metrics = {"snapshot_publish_latency_ms": 1.0} if snapshot_metrics is None else snapshot_metrics
        self.fail_restore = fail_restore
        self.fail_initialize = fail_initialize
        self.fail_seed_snapshot = fail_seed_snapshot
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.model.recorder = self

    def hooks(self) -> MinimalTrainingSetupHooks:
        return MinimalTrainingSetupHooks(
            spec_dimensions=self.spec_dimensions,
            training_paths=self.training_paths_fn,
            validate_algorithm_model_contract=self.validate_algorithm_model_contract,
            build_policy_value_model=self.build_policy_value_model,
            maybe_compile_learner_model=self.maybe_compile_learner_model,
            build_training_learner=self.build_training_learner,
            restore_learner_from_checkpoint=self.restore_learner_from_checkpoint,
            initialize_learner_from_checkpoint=self.initialize_learner_from_checkpoint,
            compute_config_hash256=self.compute_config_hash256,
            ensure_noleague_baseline_anchor=self.ensure_noleague_baseline_anchor,
            import_seed_snapshot_pool=self.import_seed_snapshot_pool,
            canonical_config_dict=self.canonical_config_dict,
            build_runtime_config=self.build_runtime_config,
            queue_runtime_cls=self.queue_runtime_cls(),
        )

    def queue_runtime_cls(self) -> type:
        recorder = self

        class RecordingRuntime:
            def __init__(self, **kwargs: object) -> None:
                recorder.calls.append(("runtime", kwargs))
                self.kwargs = kwargs

            def maybe_publish_snapshot(self, **kwargs: object) -> dict[str, float]:
                recorder.calls.append(("snapshot", kwargs))
                return recorder.snapshot_metrics

        return RecordingRuntime

    def spec_dimensions(self, received_contract: object) -> tuple[int, int]:
        self.calls.append(("spec_dimensions", {"contract": received_contract}))
        return 3, 9

    def training_paths_fn(self, run_dir: Path) -> object:
        self.calls.append(("training_paths", {"run_dir": run_dir}))
        return self.training_paths

    def validate_algorithm_model_contract(self, **kwargs: object) -> None:
        self.calls.append(("validate", kwargs))

    def build_policy_value_model(self, **kwargs: object) -> FakeModel:
        self.calls.append(("build_model", kwargs))
        return self.model

    def maybe_compile_learner_model(self, **kwargs: object) -> object:
        self.calls.append(("compile", kwargs))
        return self.compiled_model

    def build_training_learner(self, **kwargs: object) -> FakeLearner:
        self.calls.append(("build_learner", kwargs))
        return self.learner

    def restore_learner_from_checkpoint(self, **kwargs: object) -> object:
        if self.fail_restore:
            raise AssertionError("resume restore should not run")
        self.calls.append(("restore", kwargs))
        if self.resume_state is None:
            raise AssertionError("resume state was not configured")
        self.learner.update_count = int(self.resume_state.update_count)
        return self.resume_state

    def initialize_learner_from_checkpoint(self, **kwargs: object) -> object:
        if self.fail_initialize:
            raise AssertionError("init-from-checkpoint should not run in this setup")
        self.calls.append(("init", kwargs))
        if self.init_state is None:
            raise AssertionError("init state was not configured")
        return self.init_state

    def compute_config_hash256(self, received_stack: object) -> str:
        self.calls.append(("config_hash", {"stack": received_stack}))
        return self.config_hash256

    def ensure_noleague_baseline_anchor(self, **kwargs: object) -> None:
        self.calls.append(("baseline", kwargs))

    def import_seed_snapshot_pool(self, **kwargs: object) -> None:
        if self.fail_seed_snapshot:
            raise AssertionError("seed snapshot import should not run")
        self.calls.append(("seed_snapshot", kwargs))

    def canonical_config_dict(self, received_stack: object) -> dict[str, object]:
        self.calls.append(("canonical", {"stack": received_stack}))
        return self.canonical_config

    def build_runtime_config(self, **kwargs: object) -> object:
        self.calls.append(("runtime_config", kwargs))
        return self.runtime_config


def minimal_training_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        scalars_path=tmp_path / "training_metrics.jsonl",
        performance_log_path=tmp_path / "performance.jsonl",
    )


def minimal_training_stack(
    *,
    tmp_path: Path,
    algorithm: str = "ppo",
    recurrent_core: str = "none",
    encoder_kind: str = "flat",
    rewards_config: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        root=tmp_path,
        config=SimpleNamespace(
            training=SimpleNamespace(algorithm=algorithm),
            model=SimpleNamespace(recurrent_core=recurrent_core, encoder_kind=encoder_kind),
            environment=object(),
            rewards=object() if rewards_config is None else rewards_config,
        ),
    )


def minimal_training_contract(*, include_observation: bool = False) -> SimpleNamespace:
    spec_bundle: dict[str, object] = {"action": {"pass_action_id": 8}}
    if include_observation:
        spec_bundle["observation"] = {"shape": [3]}
    return SimpleNamespace(spec_bundle=spec_bundle)
