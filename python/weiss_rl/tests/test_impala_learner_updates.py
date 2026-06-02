from __future__ import annotations

from .test_impala_learner import (
    Any,
    FakeGradScaler,
    ForwardProxyModel,
    ImpalaLearner,
    ScopedOptimizerUpdateSpec,
    SimpleNamespace,
    TinyPolicyValueModel,
    VTraceTargets,
    _simple_training_batch,
    begin_impala_update_scope,
    build_scoped_impala_loss,
    cast,
    finalize_impala_update_scope,
    has_impala_training_inputs,
    impala_auxiliary_update,
    impala_normal_update,
    impala_paired_outcome_update,
    impala_paired_swing_update,
    impala_update_training_step,
    log_impala_update_metrics_if_due,
    missing_impala_training_input_fields,
    nn,
    np,
    pytest,
    resolve_impala_update_vtrace_result,
    run_impala_optimizer_step,
    run_impala_training_optimizer_step,
    run_scoped_impala_optimizer_update,
    set_impala_model_train_mode,
    summarize_precomputed_vtrace_update_metrics,
    time,
    torch,
    validate_impala_training_inputs,
)


def test_run_impala_optimizer_step_reports_no_grad_without_stepping() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=torch.tensor(2.0),
        base_metrics={"loss": 2.0},
        context={},
        scale_loss_on_nonfinite_gradients=False,
    )

    assert metrics["loss"] == pytest.approx(2.0)
    assert metrics["optimizer_no_grad"] == pytest.approx(1.0)
    assert metrics["amp_grad_overflow"] == pytest.approx(0.0)
    assert metrics["loss_scale"] == pytest.approx(0.0)
    assert metrics["grad_norm"] == pytest.approx(0.0)


def test_run_impala_optimizer_step_preserves_standard_amp_backoff_policy() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan))
    learner = ImpalaLearner(model=model)
    cast(Any, learner)._grad_scaler = FakeGradScaler(scale=8.0)

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=model.weight.sum(),
        base_metrics={"loss": 1.0},
        context={},
        scale_loss_on_nonfinite_gradients=True,
    )

    assert metrics["amp_grad_overflow"] == pytest.approx(1.0)
    assert metrics["loss_scale"] == pytest.approx(4.0)
    assert np.isnan(metrics["grad_norm"])


def test_run_impala_optimizer_step_preserves_auxiliary_amp_update_policy() -> None:
    model = nn.Linear(1, 1, bias=False)
    model.weight.register_hook(lambda grad: torch.full_like(grad, torch.nan))
    learner = ImpalaLearner(model=model)
    cast(Any, learner)._grad_scaler = FakeGradScaler(scale=8.0)

    metrics = run_impala_optimizer_step(
        learner=learner,
        batch={},
        loss=model.weight.sum(),
        base_metrics={"loss": 1.0},
        context={},
        scale_loss_on_nonfinite_gradients=False,
    )

    assert metrics["amp_grad_overflow"] == pytest.approx(1.0)
    assert metrics["loss_scale"] == pytest.approx(8.0)
    assert np.isnan(metrics["grad_norm"])


def test_finalize_impala_update_scope_merges_and_clears_profile_timers() -> None:
    learner = ImpalaLearner(profile_timers=True)
    cast(Any, learner)._active_timing_metrics = {"timer_custom_ms": 3.5}

    metrics = finalize_impala_update_scope(
        learner=learner,
        metrics={"loss": 1.0},
        started_at=time.perf_counter(),
    )

    assert metrics["loss"] == pytest.approx(1.0)
    assert metrics["timer_custom_ms"] == pytest.approx(3.5)
    assert metrics["timer_learner_total_ms"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics is None


def test_auxiliary_update_scope_preserves_update_count_and_training_metrics_policy() -> None:
    learner = ImpalaLearner(profile_timers=True)
    learner.update_count = 7

    scope = begin_impala_update_scope(
        learner=learner,
        batch=_simple_training_batch(),
        count_learner_update=False,
        include_training_metrics=False,
        checkpoint_on_interval=False,
    )

    assert learner.update_count == 7
    assert learner.total_samples_processed == 2
    assert scope.metrics == {}
    assert cast(Any, learner)._active_timing_metrics == {}


def test_set_impala_model_train_mode_sets_compiled_model_too() -> None:
    model = TinyPolicyValueModel(action_dim=2)
    compiled_model = ForwardProxyModel(model)
    model.eval()
    compiled_model.eval()
    learner = ImpalaLearner(model=model, compiled_model=compiled_model)

    set_impala_model_train_mode(learner)

    assert model.training is True
    assert compiled_model.training is True


def test_build_scoped_impala_loss_sets_train_mode_times_and_preserves_outputs() -> None:
    model = TinyPolicyValueModel(action_dim=2)
    compiled_model = ForwardProxyModel(model)
    learner = ImpalaLearner(model=model, compiled_model=compiled_model, profile_timers=True)
    model.eval()
    compiled_model.eval()
    timings: list[tuple[str, float]] = []
    cast(Any, learner)._record_timing_ms = lambda name, duration: timings.append((name, duration))
    loss = model.policy.weight.sum()
    metrics = {"custom_loss": 1.0}
    context = {"custom_context": torch.tensor(1.0)}
    calls: list[str] = []

    stage = build_scoped_impala_loss(
        learner=learner,
        loss_timer_name="learner_custom_loss",
        build_loss=lambda: (
            calls.append("loss") or loss,
            metrics,
            context,
        ),
    )

    assert calls == ["loss"]
    assert model.training is True
    assert compiled_model.training is True
    assert stage.loss is loss
    assert stage.metrics is metrics
    assert stage.context is context
    assert [name for name, _duration in timings] == ["learner_custom_loss"]
    assert timings[0][1] >= 0.0


def test_run_scoped_impala_optimizer_update_preserves_auxiliary_scope_and_timing_contract() -> None:
    model = nn.Linear(1, 1, bias=False)
    learner = ImpalaLearner(model=model, profile_timers=True)
    learner.update_count = 4
    model.eval()
    calls: list[str] = []

    metrics = run_scoped_impala_optimizer_update(
        learner=learner,
        batch=_simple_training_batch(),
        spec=ScopedOptimizerUpdateSpec(
            missing_model_message="missing model",
            loss_timer_name="learner_custom_loss",
        ),
        build_loss=lambda: (
            calls.append("loss") or model.weight.sum(),
            {"custom_loss": 1.0},
            {"context": torch.tensor(1.0)},
        ),
    )

    assert calls == ["loss"]
    assert model.training is True
    assert learner.update_count == 4
    assert learner.total_samples_processed == 2
    assert metrics["custom_loss"] == pytest.approx(1.0)
    assert "grad_norm" in metrics
    assert metrics["timer_learner_custom_loss_ms"] >= 0.0
    assert metrics["timer_learner_backward_ms"] >= 0.0
    assert metrics["timer_learner_optimizer_ms"] >= 0.0
    assert metrics["timer_learner_total_ms"] >= 0.0
    assert cast(Any, learner)._active_timing_metrics is None


def test_run_scoped_impala_optimizer_update_rejects_missing_model_before_loss_build() -> None:
    learner = ImpalaLearner(model=None)
    calls: list[str] = []

    with pytest.raises(ValueError, match="custom missing model"):
        run_scoped_impala_optimizer_update(
            learner=learner,
            batch=_simple_training_batch(),
            spec=ScopedOptimizerUpdateSpec(
                missing_model_message="custom missing model",
                loss_timer_name="learner_custom_loss",
            ),
            build_loss=lambda: (
                calls.append("loss") or torch.tensor(1.0),
                {},
                {},
            ),
        )

    assert calls == []
    assert learner.update_count == 0
    assert learner.total_samples_processed == 0


def test_run_impala_auxiliary_optimizer_update_uses_auxiliary_loss_and_scoped_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"auxiliary": True}
    loss = torch.tensor(1.25)
    loss_metrics = {"auxiliary_metric": 1.25}
    loss_context = {"auxiliary_context": torch.tensor(2.0)}
    calls: list[tuple[str, Any]] = []

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert (
            kwargs["spec"].missing_model_message == "ImpalaLearner requires a model to run an auxiliary optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_auxiliary_loss_and_metrics"
        built_loss, built_metrics, built_context = kwargs["build_loss"]()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        return {"loss": float(built_loss.item()), **built_metrics}

    learner = SimpleNamespace(
        model=object(),
        _auxiliary_loss_and_metrics=lambda source_batch: (
            calls.append(("auxiliary_loss", source_batch)) or loss,
            loss_metrics,
            loss_context,
        ),
    )
    monkeypatch.setattr(
        impala_auxiliary_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_auxiliary_update.run_impala_auxiliary_optimizer_update(learner=learner, batch=batch)

    assert result == {"loss": pytest.approx(1.25), "auxiliary_metric": 1.25}
    assert [name for name, _payload in calls] == ["scoped", "auxiliary_loss"]
    assert calls[1] == ("auxiliary_loss", batch)


def test_run_impala_auxiliary_optimizer_update_rejects_missing_model_before_auxiliary_loss() -> None:
    learner = SimpleNamespace(
        model=None,
        _auxiliary_loss_and_metrics=lambda _batch: pytest.fail("auxiliary loss should not be built"),
    )

    with pytest.raises(ValueError, match="ImpalaLearner requires a model to run an auxiliary optimizer step"):
        impala_auxiliary_update.run_impala_auxiliary_optimizer_update(
            learner=learner,
            batch=_simple_training_batch(),
        )


def test_run_impala_normal_update_runs_training_step_diagnostics_logging_and_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = SimpleNamespace(name="learner")
    batch = {"training": True}
    scope_metrics = {"loss": 0.0, "throughput": 10.0}
    vtrace_result = object()
    calls: list[tuple[str, Any]] = []

    def fake_begin_scope(**kwargs: Any) -> SimpleNamespace:
        calls.append(("begin", kwargs))
        assert kwargs == {
            "learner": learner,
            "batch": batch,
            "count_learner_update": True,
            "include_training_metrics": True,
            "checkpoint_on_interval": True,
        }
        return SimpleNamespace(started_at=12.5, metrics=scope_metrics)

    def fake_training_step(**kwargs: Any) -> dict[str, float]:
        calls.append(("training", kwargs))
        assert kwargs == {"learner": learner, "batch": batch}
        return {"loss": 1.5, "grad_norm": 0.25}

    def fake_summarize(**kwargs: Any) -> dict[str, float]:
        calls.append(("summarize", kwargs))
        assert kwargs == {"learner": learner, "batch": batch, "vtrace_result": vtrace_result}
        return {"vtrace_rho_p50": 0.75}

    def fake_log(**kwargs: Any) -> bool:
        calls.append(("log", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["metrics"] is scope_metrics
        assert kwargs["metrics"] == {
            "loss": 1.5,
            "throughput": 10.0,
            "grad_norm": 0.25,
            "vtrace_rho_p50": 0.75,
        }
        return True

    def fake_finalize(**kwargs: Any) -> dict[str, float]:
        calls.append(("finalize", kwargs))
        assert kwargs == {"learner": learner, "metrics": scope_metrics, "started_at": 12.5}
        return {"final_loss": scope_metrics["loss"], "final_vtrace": scope_metrics["vtrace_rho_p50"]}

    monkeypatch.setattr(impala_normal_update, "begin_impala_update_scope", fake_begin_scope)
    monkeypatch.setattr(impala_normal_update, "resolve_impala_update_vtrace_result", lambda source_batch: vtrace_result)
    monkeypatch.setattr(impala_normal_update, "has_impala_training_inputs", lambda source_batch: True)
    monkeypatch.setattr(impala_normal_update, "run_impala_training_optimizer_step", fake_training_step)
    monkeypatch.setattr(impala_normal_update, "summarize_precomputed_vtrace_update_metrics", fake_summarize)
    monkeypatch.setattr(impala_normal_update, "log_impala_update_metrics_if_due", fake_log)
    monkeypatch.setattr(impala_normal_update, "finalize_impala_update_scope", fake_finalize)

    result = impala_normal_update.run_impala_normal_update(learner=learner, batch=batch)

    assert result == {"final_loss": 1.5, "final_vtrace": 0.75}
    assert [name for name, _payload in calls] == ["begin", "training", "summarize", "log", "finalize"]


def test_run_impala_normal_update_skips_training_step_without_training_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = SimpleNamespace(name="learner")
    batch = {"metadata_only": True}
    scope_metrics = {"loss": 0.0}
    calls: list[str] = []

    monkeypatch.setattr(
        impala_normal_update,
        "begin_impala_update_scope",
        lambda **kwargs: calls.append("begin") or SimpleNamespace(started_at=3.0, metrics=scope_metrics),
    )
    monkeypatch.setattr(
        impala_normal_update,
        "resolve_impala_update_vtrace_result",
        lambda source_batch: calls.append("vtrace") or None,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "has_impala_training_inputs",
        lambda source_batch: calls.append("has_training") or False,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "run_impala_training_optimizer_step",
        lambda **_kwargs: pytest.fail("training optimizer step should be skipped"),
    )
    monkeypatch.setattr(
        impala_normal_update,
        "summarize_precomputed_vtrace_update_metrics",
        lambda **kwargs: calls.append("summarize") or {"vtrace_rows": 0.0},
    )
    monkeypatch.setattr(
        impala_normal_update,
        "log_impala_update_metrics_if_due",
        lambda **kwargs: calls.append("log") or False,
    )
    monkeypatch.setattr(
        impala_normal_update,
        "finalize_impala_update_scope",
        lambda **kwargs: calls.append("finalize") or dict(kwargs["metrics"]),
    )

    result = impala_normal_update.run_impala_normal_update(learner=learner, batch=batch)

    assert result == {"loss": 0.0, "vtrace_rows": 0.0}
    assert calls == ["begin", "vtrace", "has_training", "summarize", "log", "finalize"]


def test_run_impala_paired_swing_optimizer_update_validates_full_surface_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    learner = SimpleNamespace(model=object())

    monkeypatch.setattr(
        impala_paired_swing_update,
        "run_scoped_impala_optimizer_update",
        lambda **_kwargs: calls.append("scoped"),
    )

    with pytest.raises(ValueError, match="full_surface_top_action_retention_coef must be >= 0"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_coef=-0.1,
        )
    with pytest.raises(ValueError, match="full_surface_top_action_retention_margin must be >= 0"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_margin=-0.1,
        )
    with pytest.raises(ValueError, match="full_surface_retention_batch is required"):
        impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
            learner=learner,
            batch={},
            margin=1,
            coef=1,
            positive_action_source="positive",
            negative_action_source="negative",
            full_surface_top_action_retention_coef=0.5,
        )

    assert calls == []


def test_run_impala_paired_swing_optimizer_update_composes_full_surface_retention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"paired": True}
    retention_batch = {"retention": True}
    swing_loss = torch.tensor(2.0)
    retention_loss = torch.tensor(0.5)
    calls: list[tuple[str, Any]] = []

    def paired_swing_loss(source_batch: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("swing", (source_batch, kwargs)))
        return swing_loss, {"swing_metric": 2.0}, {"swing_context": "base"}

    def full_surface_retention(
        source_batch: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("retention", (source_batch, kwargs)))
        return retention_loss, {"retention_metric": 0.5}, {"retention_context": "extra"}

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["spec"].missing_model_message == (
            "ImpalaLearner requires a model to run a paired-swing optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_paired_swing_loss_and_metrics"
        loss, metrics, context = kwargs["build_loss"]()
        assert loss.item() == pytest.approx(2.5)
        assert metrics == {"swing_metric": 2.0, "retention_metric": 0.5}
        assert context == {"swing_context": "base", "retention_context": "extra"}
        return {"loss": float(loss.item()), **metrics}

    learner = SimpleNamespace(
        model=object(),
        _paired_swing_loss_and_metrics=paired_swing_loss,
        _paired_swing_full_surface_top_action_retention_loss_and_metrics=full_surface_retention,
    )
    monkeypatch.setattr(
        impala_paired_swing_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_swing_update.run_impala_paired_swing_optimizer_update(
        learner=learner,
        batch=batch,
        margin=1,
        coef=0.75,
        positive_action_source="teacher_positive",
        negative_action_source="learner_negative",
        loss_scope="span",
        compare_to="baseline",
        margin_retention_coef=0.25,
        margin_retention_margin=0.5,
        top_action_retention_coef=0.125,
        top_action_retention_margin=0.75,
        full_surface_retention_batch=retention_batch,
        full_surface_top_action_retention_coef=0.4,
        full_surface_top_action_retention_margin=0.6,
        full_surface_top_action_retention_mode="target_action",
    )

    assert result == {"loss": pytest.approx(2.5), "swing_metric": 2.0, "retention_metric": 0.5}
    assert [name for name, _payload in calls] == ["scoped", "swing", "retention"]
    assert calls[1][0] == "swing"
    assert calls[1][1][0] is batch
    assert calls[1][1][1] == {
        "margin": 1.0,
        "coef": 0.75,
        "positive_action_source": "teacher_positive",
        "negative_action_source": "learner_negative",
        "loss_scope": "span",
        "compare_to": "baseline",
        "margin_retention_coef": 0.25,
        "margin_retention_margin": 0.5,
        "top_action_retention_coef": 0.125,
        "top_action_retention_margin": 0.75,
    }
    assert calls[2] == (
        "retention",
        (
            retention_batch,
            {
                "coef": 0.4,
                "margin": 0.6,
                "mode": "target_action",
            },
        ),
    )


def test_run_impala_paired_outcome_preference_optimizer_update_forwards_casted_replay_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"preference": True}
    loss = torch.tensor(1.75)
    loss_metrics = {"preference_metric": 1.75}
    loss_context = {"preference_context": torch.tensor(3.0)}
    calls: list[tuple[str, Any]] = []

    def paired_outcome_loss(
        source_batch: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        calls.append(("preference_loss", (source_batch, kwargs)))
        return loss, loss_metrics, loss_context

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        calls.append(("scoped", kwargs))
        assert kwargs["learner"] is learner
        assert kwargs["batch"] is batch
        assert kwargs["spec"].missing_model_message == (
            "ImpalaLearner requires a model to run a paired outcome preference optimizer step"
        )
        assert kwargs["spec"].loss_timer_name == "learner_paired_outcome_preference_loss_and_metrics"
        built_loss, built_metrics, built_context = kwargs["build_loss"]()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        return {"loss": float(built_loss.item()), **built_metrics}

    learner = SimpleNamespace(
        model=object(),
        _paired_outcome_preference_loss_and_metrics=paired_outcome_loss,
    )
    monkeypatch.setattr(
        impala_paired_outcome_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_outcome_update.run_impala_paired_outcome_preference_optimizer_update(
        learner=learner,
        batch=batch,
        beta="0.7",
        coef="0.25",
        aggregation=123,
        group_balance=1,
        retention_coef="0.5",
        retention_margin="0.125",
        retention_role=456,
        retention_reference_top_only=1,
        top_action_retention_coef="0.75",
        top_action_retention_margin="0.875",
        top_action_retention_role=789,
        top_action_retention_reference_top_only=1,
    )

    assert result == {"loss": pytest.approx(1.75), "preference_metric": 1.75}
    assert [name for name, _payload in calls] == ["scoped", "preference_loss"]
    assert calls[1][1][0] is batch
    assert calls[1][1][1] == {
        "beta": 0.7,
        "coef": 0.25,
        "aggregation": "123",
        "group_balance": True,
        "retention_coef": 0.5,
        "retention_margin": 0.125,
        "retention_role": "456",
        "retention_reference_top_only": True,
        "top_action_retention_coef": 0.75,
        "top_action_retention_margin": 0.875,
        "top_action_retention_role": "789",
        "top_action_retention_reference_top_only": True,
    }


def test_run_impala_paired_outcome_preference_optimizer_update_uses_default_replay_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"preference": True}
    captured_kwargs: dict[str, Any] = {}
    learner = SimpleNamespace(
        model=object(),
        _paired_outcome_preference_loss_and_metrics=lambda source_batch, **kwargs: (
            captured_kwargs.update(kwargs) or torch.tensor(0.5),
            {"preference_metric": 0.5},
            {},
        ),
    )

    def fake_scoped_update(**kwargs: Any) -> dict[str, float]:
        loss, metrics, _context = kwargs["build_loss"]()
        return {"loss": float(loss.item()), **metrics}

    monkeypatch.setattr(
        impala_paired_outcome_update,
        "run_scoped_impala_optimizer_update",
        fake_scoped_update,
    )

    result = impala_paired_outcome_update.run_impala_paired_outcome_preference_optimizer_update(
        learner=learner,
        batch=batch,
        beta=0.3,
        coef=0.2,
    )

    assert result == {"loss": pytest.approx(0.5), "preference_metric": 0.5}
    assert captured_kwargs == {
        "beta": 0.3,
        "coef": 0.2,
        "aggregation": "mean",
        "group_balance": False,
        "retention_coef": 0.0,
        "retention_margin": 0.0,
        "retention_role": "preferred",
        "retention_reference_top_only": False,
        "top_action_retention_coef": 0.0,
        "top_action_retention_margin": 0.0,
        "top_action_retention_role": "all",
        "top_action_retention_reference_top_only": False,
    }


def test_impala_update_training_input_helpers_preserve_missing_field_contract() -> None:
    learner = SimpleNamespace(
        _has_legal_actions=lambda batch: False,
        _has_raw_vtrace_inputs=lambda batch: False,
    )
    batch = {"obs": np.zeros((1, 1, 2), dtype=np.float32)}

    assert has_impala_training_inputs(batch) is True
    assert resolve_impala_update_vtrace_result(batch) is None
    assert missing_impala_training_input_fields(learner=learner, batch=batch) == [
        "actions",
        "legal_actions",
        "vtrace_result_or_raw_inputs",
    ]
    with pytest.raises(
        ValueError,
        match=(
            "batch must include obs, actions, legality, and either vtrace_result or raw vtrace inputs "
            "for learner updates; missing actions, legal_actions, vtrace_result_or_raw_inputs"
        ),
    ):
        validate_impala_training_inputs(learner=learner, batch=batch)


def test_impala_update_training_input_helpers_accept_raw_vtrace_and_summarize_precomputed_targets() -> None:
    learner = SimpleNamespace(
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        _has_legal_actions=lambda batch: True,
        _has_raw_vtrace_inputs=lambda batch: True,
    )
    vtrace_result = VTraceTargets(
        vs=np.zeros((2, 1), dtype=np.float32),
        pg_advantages=np.zeros((2, 1), dtype=np.float32),
        rhos=np.asarray([[0.5], [2.0]], dtype=np.float32),
    )
    batch = {
        "obs": np.zeros((2, 1, 2), dtype=np.float32),
        "actions": np.zeros((2, 1), dtype=np.int64),
        "vtrace_result": vtrace_result,
        "vtrace_rho_bar": 1.5,
        "vtrace_c_bar": 0.75,
    }

    assert missing_impala_training_input_fields(learner=learner, batch=batch) == []
    validate_impala_training_inputs(learner=learner, batch=batch)
    assert (
        summarize_precomputed_vtrace_update_metrics(
            learner=learner,
            batch=batch,
            vtrace_result=None,
        )
        == {}
    )

    metrics = summarize_precomputed_vtrace_update_metrics(
        learner=learner,
        batch=batch,
        vtrace_result=vtrace_result,
    )

    assert metrics["vtrace_rho_p50"] == pytest.approx(1.25)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(0.5)


def test_log_impala_update_metrics_if_due_preserves_interval_and_timestamp_contract() -> None:
    calls: list[tuple[dict[str, float], dict[str, bool]]] = []
    metrics = {"loss": 1.0}
    batch = {"batch": True}
    learner = SimpleNamespace(
        logger=object(),
        update_count=6,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda logged_metrics, logged_batch: calls.append((logged_metrics, logged_batch)),
    )

    logged = log_impala_update_metrics_if_due(
        learner=learner,
        batch=batch,
        metrics=metrics,
        now=123.5,
    )

    assert logged is True
    assert calls == [(metrics, batch)]
    assert learner.last_log_time == pytest.approx(123.5)
    assert learner.last_log_update == 6


def test_log_impala_update_metrics_if_due_skips_without_logger_or_interval() -> None:
    calls: list[str] = []
    metrics = {"loss": 1.0}
    batch = {"batch": True}
    no_logger = SimpleNamespace(
        logger=None,
        update_count=6,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda _metrics, _batch: calls.append("no_logger"),
    )
    off_interval = SimpleNamespace(
        logger=object(),
        update_count=5,
        logging_interval_updates=3,
        last_log_time=0.0,
        last_log_update=0,
        _log_metrics=lambda _metrics, _batch: calls.append("off_interval"),
    )

    assert log_impala_update_metrics_if_due(learner=no_logger, batch=batch, metrics=metrics, now=1.0) is False
    assert log_impala_update_metrics_if_due(learner=off_interval, batch=batch, metrics=metrics, now=1.0) is False
    assert calls == []
    assert no_logger.last_log_time == pytest.approx(0.0)
    assert no_logger.last_log_update == 0
    assert off_interval.last_log_time == pytest.approx(0.0)
    assert off_interval.last_log_update == 0


def test_run_impala_training_optimizer_step_validates_builds_loss_and_scales_nonfinite_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = {"training_step_batch": True}
    learner = SimpleNamespace(model=object())
    loss = torch.tensor(2.0)
    loss_metrics = {"loss": 2.0}
    loss_context = {"context": torch.tensor(1.0)}
    calls: list[str] = []

    def fake_validate_impala_training_inputs(*, learner: Any, batch: Any) -> None:
        assert learner is training_learner
        assert batch is training_batch
        calls.append("validate")

    def fake_build_scoped_impala_loss(*, learner: Any, loss_timer_name: str, build_loss: Any) -> SimpleNamespace:
        assert learner is training_learner
        assert loss_timer_name == "learner_loss_and_metrics"
        built_loss, built_metrics, built_context = build_loss()
        assert built_loss is loss
        assert built_metrics is loss_metrics
        assert built_context is loss_context
        calls.append("build")
        return SimpleNamespace(loss=built_loss, metrics=built_metrics, context=built_context)

    def fake_run_impala_optimizer_step(**kwargs: Any) -> dict[str, float]:
        assert kwargs["learner"] is training_learner
        assert kwargs["batch"] is training_batch
        assert kwargs["loss"] is loss
        assert kwargs["base_metrics"] is loss_metrics
        assert kwargs["context"] is loss_context
        assert kwargs["scale_loss_on_nonfinite_gradients"] is True
        calls.append("optimizer")
        return {"loss": 2.0, "grad_norm": 0.5}

    training_learner = learner
    training_batch = batch
    learner._loss_and_metrics_with_context = lambda source_batch: (
        calls.append("loss") or loss,
        loss_metrics,
        loss_context,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "validate_impala_training_inputs",
        fake_validate_impala_training_inputs,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "build_scoped_impala_loss",
        fake_build_scoped_impala_loss,
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "run_impala_optimizer_step",
        fake_run_impala_optimizer_step,
    )

    metrics = run_impala_training_optimizer_step(learner=learner, batch=batch)

    assert calls == ["validate", "loss", "build", "optimizer"]
    assert metrics == {"loss": 2.0, "grad_norm": 0.5}


def test_run_impala_training_optimizer_step_rejects_missing_model_after_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    learner = SimpleNamespace(model=None)

    monkeypatch.setattr(
        impala_update_training_step,
        "validate_impala_training_inputs",
        lambda *, learner, batch: calls.append("validate"),
    )
    monkeypatch.setattr(
        impala_update_training_step,
        "build_scoped_impala_loss",
        lambda **_kwargs: calls.append("build"),
    )

    with pytest.raises(ValueError, match="ImpalaLearner requires a model to run an optimizer step"):
        run_impala_training_optimizer_step(learner=learner, batch={})

    assert calls == ["validate"]


def test_impala_learner_mixed_precision_flag_disables_amp_on_cpu() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), mixed_precision=True)

    metrics = learner.update(_simple_training_batch())

    assert metrics["loss"] != 0.0
    assert learner._amp_enabled is False
    assert learner._grad_scaler is None
