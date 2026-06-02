from __future__ import annotations

from .test_impala_learner import (
    Any,
    FakeGradScaler,
    ImpalaLearner,
    NaNGradientModel,
    NaNLogitModel,
    Path,
    _simple_training_batch,
    begin_impala_update_scope,
    cast,
    json,
    pytest,
)


def test_begin_impala_update_scope_counts_normal_updates_and_checkpoint_metadata(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=1,
        profile_timers=True,
    )

    scope = begin_impala_update_scope(
        learner=learner,
        batch=_simple_training_batch(),
        count_learner_update=True,
        include_training_metrics=True,
        checkpoint_on_interval=True,
    )

    assert learner.update_count == 1
    assert learner.total_samples_processed == 2
    assert learner.policy_version == 1
    assert (tmp_path / "checkpoints" / "checkpoint_metadata_1.json").is_file()
    assert scope.metrics["loss"] == pytest.approx(0.0)
    assert scope.metrics["entropy_coef"] == pytest.approx(float(learner.entropy_coef))
    assert scope.metrics["throughput_samples_per_sec"] >= 0.0
    assert scope.metrics["throughput_updates_per_sec"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics == {}


def test_impala_learner_writes_checkpoint_metadata_using_update_count(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=2,
    )

    for _ in range(4):
        result = learner.update({})
        assert result["loss"] == 0.0

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "checkpoint_metadata_2.json").is_file()
    assert (checkpoint_dir / "checkpoint_metadata_4.json").is_file()
    assert learner.get_policy_version() == 2


def test_impala_learner_checkpoint_metadata_records_scope_update_and_policy_version(tmp_path: Path) -> None:
    learner = ImpalaLearner(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_interval_updates=3,
    )

    for _ in range(3):
        learner.update({})

    checkpoint_metadata = json.loads(
        (tmp_path / "checkpoints" / "checkpoint_metadata_3.json").read_text(encoding="utf-8")
    )
    assert checkpoint_metadata == {
        "format": "checkpoint_metadata",
        "parameters_included": False,
        "policy_version": 1,
        "update_count": 3,
    }


def test_impala_learner_writes_fault_bundle_on_nonfinite_forward_logits(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNLogitModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner forward_logits; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "forward_logits"
    assert payload["context"]["forward_logits_nonfinite_indices"]["data"] == [[0, 0, 0], [1, 0, 0]]


def test_impala_learner_writes_fault_bundle_on_nonfinite_gradients(tmp_path: Path) -> None:
    fault_dir = tmp_path / "faults"
    learner = ImpalaLearner(model=NaNGradientModel(), fault_dir=fault_dir)

    with pytest.raises(RuntimeError, match="non-finite learner gradients; wrote fault bundle to ") as excinfo:
        learner.update(_simple_training_batch())

    [fault_path] = sorted(fault_dir.glob("learner_numeric_fault_*.json"))
    assert str(fault_path) in str(excinfo.value)

    payload = json.loads(fault_path.read_text(encoding="utf-8"))
    assert payload["component"] == "impala_learner"
    assert payload["stage"] == "gradients"
    assert "logit_bias" in payload["context"]["bad_gradient_names"]


def test_impala_learner_amp_overflow_is_reported_without_raising() -> None:
    learner = ImpalaLearner(model=NaNGradientModel())
    cast(Any, learner)._grad_scaler = FakeGradScaler(overflow=True)

    metrics = learner.update(_simple_training_batch())

    assert metrics["amp_grad_overflow"] == 1.0
    assert metrics["loss_scale"] == pytest.approx(4.0)
    assert learner.update_count == 1
