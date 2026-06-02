from __future__ import annotations

from .test_impala_learner import (
    Any,
    ImpalaLearner,
    ImpalaMetricAssemblyRequest,
    LegalActionBatch,
    SeatAwareTinyPolicyValueModel,
    SimpleNamespace,
    TinyPolicyValueModel,
    TrunkStructuredTeacherModel,
    VTraceTargets,
    _masked_action_logp_and_entropy,
    _packed_meta_from_ids,
    _simple_training_batch,
    _teacher_aux_catalog,
    assemble_impala_loss_core_metrics,
    attach_resolved_vtrace_context,
    cast,
    compute_impala_loss_and_metrics_with_context,
    compute_impala_loss_core,
    compute_impala_objective_losses,
    compute_impala_objective_stage,
    compute_impala_vtrace_stage,
    impala_loss_metrics_stage,
    impala_loss_vtrace_stage,
    np,
    packed_scores_action_logp_and_entropy,
    prepare_impala_loss_inputs,
    pytest,
    resolve_impala_action_reductions,
    resolve_impala_value_loss_mask,
    resolve_impala_vtrace_clip_config,
    resolve_impala_vtrace_targets,
    torch,
)


def test_compute_impala_objective_losses_uses_current_logp_for_retention_and_policy_logp_for_pg() -> None:
    result = compute_impala_objective_losses(
        policy_action_logp=torch.zeros((2, 1), dtype=torch.float32),
        retention_action_logp=torch.as_tensor([[-0.25], [-2.0]], dtype=torch.float32),
        actions=torch.as_tensor([[0], [1]], dtype=torch.long),
        advantages=torch.ones((2, 1), dtype=torch.float32),
        values=torch.zeros((2, 1), dtype=torch.float32),
        targets=torch.zeros((2, 1), dtype=torch.float32),
        entropy=torch.zeros((2, 1), dtype=torch.float32),
        loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
        value_loss_mask=None,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        trajectory_retention_valid=torch.as_tensor([[False], [True]], dtype=torch.bool),
        trajectory_retention_coef=0.5,
    )

    assert result.policy_loss.item() == pytest.approx(0.0)
    assert result.value_loss.item() == pytest.approx(0.0)
    assert result.trajectory_retention_metrics["trajectory_retention_loss"] == pytest.approx(2.0)
    assert result.trajectory_retention_metrics["trajectory_retention_weighted_loss"] == pytest.approx(1.0)
    assert result.total_loss.item() == pytest.approx(1.0)
    assert result.value_loss_mask.tolist() == [[1.0], [1.0]]


def test_compute_impala_objective_losses_respects_explicit_value_mask_and_entropy_term() -> None:
    result = compute_impala_objective_losses(
        policy_action_logp=torch.as_tensor([[-0.5], [-4.0]], dtype=torch.float32),
        retention_action_logp=torch.as_tensor([[-0.5], [-4.0]], dtype=torch.float32),
        actions=torch.as_tensor([[0], [1]], dtype=torch.long),
        advantages=torch.as_tensor([[2.0], [10.0]], dtype=torch.float32),
        values=torch.as_tensor([[0.0], [3.0]], dtype=torch.float32),
        targets=torch.as_tensor([[2.0], [1.0]], dtype=torch.float32),
        entropy=torch.as_tensor([[0.25], [99.0]], dtype=torch.float32),
        loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
        value_loss_mask=torch.as_tensor([[0.0], [1.0]], dtype=torch.float32),
        value_loss_coef=0.5,
        entropy_coef=0.1,
        trajectory_retention_valid=None,
        trajectory_retention_coef=0.0,
    )

    assert result.policy_loss.item() == pytest.approx(1.0)
    assert result.value_loss.item() == pytest.approx(4.0)
    assert result.entropy_mean.item() == pytest.approx(0.25)
    assert result.total_loss.item() == pytest.approx(2.975)
    assert result.value_loss_mask.tolist() == [[0.0], [1.0]]
    assert result.trajectory_retention_metrics == {}


def test_assemble_impala_loss_core_metrics_maps_stage_outputs_to_metric_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total_loss = torch.tensor(3.0, dtype=torch.float32)
    policy_loss = torch.tensor(0.5, dtype=torch.float32)
    value_loss = torch.tensor(1.25, dtype=torch.float32)
    entropy_mean = torch.tensor(0.125, dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    value_loss_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    actions = torch.tensor([[2], [3]], dtype=torch.long)
    action_logp = torch.tensor([[-0.2], [-0.7]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-0.1], [-0.6]], dtype=torch.float32)
    rewards = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
    advantages = torch.tensor([[0.25], [-0.5]], dtype=torch.float32)
    targets = torch.tensor([[0.75], [-0.25]], dtype=torch.float32)
    rhos = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    logits = torch.zeros((2, 1, 5), dtype=torch.float32)
    legal_mask = torch.ones((2, 1, 5), dtype=torch.bool)
    packed_legal = (
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    packed_view = object()
    factorized_result = object()
    batch = {"metric_stage_batch": True}
    resolved_mask = torch.ones_like(legal_mask)
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    timing_calls: list[tuple[str, float]] = []

    def resolve_legal_mask(source_batch: Any, *, expected_shape: torch.Size, action_dim: int) -> torch.Tensor:
        resolver_calls.append((source_batch, expected_shape, action_dim))
        return resolved_mask

    def record_timing(name: str, duration: float) -> None:
        timing_calls.append((name, duration))

    learner = SimpleNamespace(
        entropy_scope="family",
        pass_action_id=4,
        _resolve_legal_mask=resolve_legal_mask,
        _record_timing_ms=record_timing,
    )
    inputs = SimpleNamespace(
        obs=torch.zeros((2, 1, 3), dtype=torch.float32),
        loss_mask=loss_mask,
        actions=actions,
        emit_structured_metrics=True,
        logits=logits,
        legal_mask=legal_mask,
        packed_legal=packed_legal,
        packed_view=packed_view,
        factorized_result=factorized_result,
    )
    objective_losses = SimpleNamespace(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        value_loss_mask=value_loss_mask,
        trajectory_retention_metrics={"trajectory_retention_rows": 2.0},
    )
    policy_anchor_stage = SimpleNamespace(policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25})
    teacher_finalization = SimpleNamespace(teacher_metrics={"teacher_aux_loss": 0.5})
    resolved_vtrace = SimpleNamespace(
        behavior_logp_for_mask=behavior_logp,
        rewards_for_metrics=rewards,
        advantages=advantages,
        targets=targets,
        rhos_for_metrics=rhos,
    )
    clip_config = SimpleNamespace(rho_bar=1.5, c_bar=1.25)
    batch_values: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_values.append((source_batch, key))
        return None

    captured: dict[str, Any] = {}

    def fake_assemble_impala_loss_metrics(
        request: ImpalaMetricAssemblyRequest,
        *,
        batch_value: Any,
        record_timing_ms: Any,
    ) -> dict[str, float]:
        captured["request"] = request
        captured["batch_value"] = batch_value
        captured["record_timing_ms"] = record_timing_ms
        assert request.resolve_legal_mask is not None
        assert request.resolve_legal_mask(batch, torch.Size((2, 1)), 5) is resolved_mask
        return {"loss": 3.0, "metric_stage": 1.0}

    monkeypatch.setattr(
        impala_loss_metrics_stage,
        "assemble_impala_loss_metrics",
        fake_assemble_impala_loss_metrics,
    )

    metrics = assemble_impala_loss_core_metrics(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        objective_losses=cast(Any, objective_losses),
        policy_anchor_stage=cast(Any, policy_anchor_stage),
        teacher_finalization=cast(Any, teacher_finalization),
        resolved_vtrace=resolved_vtrace,
        clip_config=clip_config,
        action_logp=action_logp,
        action_catalog="catalog",
        batch_value=batch_value,
    )

    request = cast(ImpalaMetricAssemblyRequest, captured["request"])
    assert metrics == {"loss": 3.0, "metric_stage": 1.0}
    assert captured["batch_value"] is batch_value
    assert captured["record_timing_ms"] is record_timing
    assert request.total_loss is total_loss
    assert request.policy_loss is policy_loss
    assert request.value_loss is value_loss
    assert request.entropy_mean is entropy_mean
    assert request.entropy_scope == "family"
    assert request.loss_mask is loss_mask
    assert request.value_loss_mask is value_loss_mask
    assert request.actions is actions
    assert request.action_logp is action_logp
    assert request.behavior_logp_for_mask is behavior_logp
    assert request.rewards_for_metrics is rewards
    assert request.advantages is advantages
    assert request.targets is targets
    assert request.rhos_for_metrics is rhos
    assert request.rho_bar == pytest.approx(1.5)
    assert request.c_bar == pytest.approx(1.25)
    assert request.action_catalog == "catalog"
    assert request.pass_action_id == 4
    assert request.trajectory_retention_metrics == {"trajectory_retention_rows": 2.0}
    assert request.policy_anchor_metrics == {"policy_anchor_weighted_loss": 0.25}
    assert request.teacher_metrics == {"teacher_aux_loss": 0.5}
    assert request.emit_structured_metrics is True
    assert request.logits is logits
    assert request.legal_mask is legal_mask
    assert request.packed_legal is packed_legal
    assert request.packed_view is packed_view
    assert request.factorized_result is factorized_result
    assert request.batch is batch
    assert request.expected_shape == torch.Size((2, 1))
    assert request.action_dim == 5
    assert resolver_calls == [(batch, torch.Size((2, 1)), 5)]
    assert batch_values == []
    assert timing_calls == []


def test_resolve_impala_vtrace_clip_config_prefers_batch_overrides_then_learner_defaults() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=1.5, vtrace_c_bar=0.75)

    defaults = resolve_impala_vtrace_clip_config(
        learner=learner,
        batch={},
        batch_value=lambda source, key: source.get(key),
    )
    overrides = resolve_impala_vtrace_clip_config(
        learner=learner,
        batch={"vtrace_rho_bar": 2.25, "vtrace_c_bar": 0.5},
        batch_value=lambda source, key: source.get(key),
    )

    assert defaults.rho_bar == pytest.approx(1.5)
    assert defaults.c_bar == pytest.approx(0.75)
    assert overrides.rho_bar == pytest.approx(2.25)
    assert overrides.c_bar == pytest.approx(0.5)


def test_attach_resolved_vtrace_context_and_value_mask_keep_detached_loss_diagnostics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    values = torch.zeros((2, 1), dtype=torch.float32)
    batch = {"value_train_mask": np.asarray([[False], [True]], dtype=np.bool_)}
    resolved_vtrace = SimpleNamespace(
        targets=torch.ones((2, 1), dtype=torch.float32, requires_grad=True),
        advantages=torch.full((2, 1), 2.0, dtype=torch.float32, requires_grad=True),
        rhos_for_metrics=torch.full((2, 1), 3.0, dtype=torch.float32, requires_grad=True),
        rewards_for_metrics=torch.full((2, 1), 4.0, dtype=torch.float32, requires_grad=True),
    )
    context: dict[str, Any] = {}

    attach_resolved_vtrace_context(
        context=context,
        resolved_vtrace=resolved_vtrace,
        loss_mask=torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True),
    )
    value_mask = resolve_impala_value_loss_mask(
        learner=learner,
        batch=batch,
        expected_shape=torch.Size((2, 1)),
        like=values,
        batch_value=lambda source, key: source.get(key),
    )

    assert context["targets"].requires_grad is False
    assert context["advantages"].requires_grad is False
    assert context["vtrace_rhos"].requires_grad is False
    assert context["rewards"].requires_grad is False
    assert context["policy_train_mask"].requires_grad is False
    assert value_mask is not None
    assert value_mask.tolist() == [[0.0], [1.0]]


def test_compute_impala_vtrace_stage_resolves_targets_and_attaches_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_logp = torch.tensor([[-0.2], [-0.7]], dtype=torch.float32, requires_grad=True)
    resolved_action_logp = torch.tensor([[-0.3], [-0.8]], dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 1), dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32, requires_grad=True)
    context: dict[str, Any] = {}
    inputs = SimpleNamespace(
        vtrace_result="vtrace-result",
        values=values,
        loss_mask=loss_mask,
        context=context,
    )
    batch = {"vtrace_rho_bar": 2.0, "vtrace_c_bar": 0.5}
    float_target = object()
    resolve_bootstrap_value = object()
    learner = SimpleNamespace(
        vtrace_rho_bar=1.0,
        vtrace_c_bar=1.0,
        _float_target=float_target,
        _resolve_vtrace_bootstrap_value=resolve_bootstrap_value,
    )
    resolved_vtrace = SimpleNamespace(
        action_logp=resolved_action_logp,
        behavior_logp_for_mask=torch.zeros((2, 1), dtype=torch.float32),
        targets=torch.ones((2, 1), dtype=torch.float32, requires_grad=True),
        advantages=torch.full((2, 1), 2.0, dtype=torch.float32, requires_grad=True),
        rhos_for_metrics=torch.full((2, 1), 3.0, dtype=torch.float32, requires_grad=True),
        rewards_for_metrics=torch.full((2, 1), 4.0, dtype=torch.float32, requires_grad=True),
    )
    captured: dict[str, Any] = {}

    def fake_resolve_impala_vtrace_targets(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return resolved_vtrace

    monkeypatch.setattr(
        impala_loss_vtrace_stage,
        "resolve_impala_vtrace_targets",
        fake_resolve_impala_vtrace_targets,
    )
    batch_value_calls: list[tuple[Any, str]] = []

    def batch_value(source_batch: Any, key: str) -> Any:
        batch_value_calls.append((source_batch, key))
        return source_batch.get(key)

    stage = compute_impala_vtrace_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        action_logp=action_logp,
        batch_value=batch_value,
    )

    assert stage.retention_action_logp is action_logp
    assert stage.action_logp is resolved_action_logp
    assert stage.clip_config.rho_bar == pytest.approx(2.0)
    assert stage.clip_config.c_bar == pytest.approx(0.5)
    assert stage.resolved_vtrace is resolved_vtrace
    assert captured["batch"] is batch
    assert captured["vtrace_result"] == "vtrace-result"
    assert captured["values"] is values
    assert captured["action_logp"] is action_logp
    assert captured["loss_mask"] is loss_mask
    assert captured["rho_bar"] == pytest.approx(2.0)
    assert captured["c_bar"] == pytest.approx(0.5)
    assert captured["float_target"] is float_target
    assert captured["resolve_bootstrap_value"] is resolve_bootstrap_value
    assert captured["batch_value"] is batch_value
    assert batch_value_calls == [(batch, "vtrace_rho_bar"), (batch, "vtrace_c_bar")]
    torch.testing.assert_close(context["targets"], resolved_vtrace.targets)
    torch.testing.assert_close(context["advantages"], resolved_vtrace.advantages)
    torch.testing.assert_close(context["vtrace_rhos"], resolved_vtrace.rhos_for_metrics)
    torch.testing.assert_close(context["rewards"], resolved_vtrace.rewards_for_metrics)
    torch.testing.assert_close(context["policy_train_mask"], loss_mask)
    assert context["targets"].requires_grad is False
    assert context["advantages"].requires_grad is False
    assert context["vtrace_rhos"].requires_grad is False
    assert context["rewards"].requires_grad is False
    assert context["policy_train_mask"].requires_grad is False


def test_compute_impala_objective_stage_preserves_context_and_objective_contract() -> None:
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=2),
        trajectory_retention_coef=0.5,
        value_loss_coef=0.25,
        entropy_coef=0.1,
    )
    batch = {"value_train_mask": np.asarray([[False], [True]], dtype=np.bool_)}
    obs = torch.zeros((2, 1, 2), dtype=torch.float32)
    context: dict[str, Any] = {}
    inputs = SimpleNamespace(
        obs=obs,
        actions=torch.tensor([[0], [1]], dtype=torch.long),
        values=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        loss_mask=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
        trajectory_retention_valid=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        factorized_result=SimpleNamespace(top_action_ids=torch.tensor([[0], [0]], dtype=torch.long)),
        context=context,
    )
    resolved_vtrace = SimpleNamespace(
        advantages=torch.tensor([[2.0], [3.0]], dtype=torch.float32),
        targets=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
    )
    policy_action_logp = torch.tensor([[-0.25], [-0.75]], dtype=torch.float32)
    retention_action_logp = torch.tensor([[-0.5], [-1.0]], dtype=torch.float32)
    entropy = torch.tensor([[0.1], [0.2]], dtype=torch.float32)

    stage = compute_impala_objective_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        entropy=entropy,
        resolved_vtrace=resolved_vtrace,
        batch_value=lambda source, key: source.get(key),
    )
    direct = compute_impala_objective_losses(
        policy_action_logp=policy_action_logp,
        retention_action_logp=retention_action_logp,
        actions=inputs.actions,
        advantages=resolved_vtrace.advantages,
        values=inputs.values,
        targets=resolved_vtrace.targets,
        entropy=entropy,
        loss_mask=inputs.loss_mask,
        value_loss_mask=context["value_train_mask"],
        value_loss_coef=float(learner.value_loss_coef),
        entropy_coef=float(learner.entropy_coef),
        trajectory_retention_valid=inputs.trajectory_retention_valid,
        trajectory_retention_coef=float(learner.trajectory_retention_coef),
        top_action_ids=inputs.factorized_result.top_action_ids,
    )

    torch.testing.assert_close(stage.losses.total_loss, direct.total_loss)
    torch.testing.assert_close(stage.losses.policy_loss, direct.policy_loss)
    torch.testing.assert_close(stage.losses.value_loss, direct.value_loss)
    torch.testing.assert_close(stage.losses.entropy_mean, direct.entropy_mean)
    torch.testing.assert_close(stage.losses.trajectory_retention_loss, direct.trajectory_retention_loss)
    assert stage.losses.trajectory_retention_metrics == pytest.approx(direct.trajectory_retention_metrics)
    assert context["value_train_mask"].requires_grad is False
    assert context["value_train_mask"].tolist() == [[0.0], [1.0]]
    assert context["trajectory_retention_loss"].requires_grad is False


def test_compute_impala_loss_core_finalizes_vtrace_objective_context_and_metrics() -> None:
    torch.manual_seed(0)
    learner = ImpalaLearner(
        model=TinyPolicyValueModel(action_dim=2),
        structured_metrics_mode="off",
        trajectory_retention_coef=0.25,
        value_loss_coef=1.0,
        entropy_coef=0.0,
    )
    batch = _simple_training_batch()
    batch["policy_train_mask"] = np.asarray([[True], [False]], dtype=np.bool_)
    batch["value_train_mask"] = np.asarray([[False], [True]], dtype=np.bool_)
    batch["trajectory_retention_valid"] = np.asarray([[False], [True]], dtype=np.bool_)

    inputs = prepare_impala_loss_inputs(learner=learner, batch=batch, batch_value=lambda source, key: source.get(key))
    reductions = resolve_impala_action_reductions(
        factorized_result=inputs.factorized_result,
        logits=inputs.logits,
        packed_logits=inputs.packed_logits,
        legal_mask=inputs.legal_mask,
        packed_legal=inputs.packed_legal,
        actions=inputs.actions,
        entropy_scope=learner.entropy_scope,
        pass_action_id=learner.pass_action_id,
        action_catalog=getattr(learner.model, "action_catalog", None),
        record_timing_ms=learner._record_timing_ms,
    )
    inputs.context["action_logp"] = reductions.action_logp.detach()
    inputs.context["entropy"] = reductions.entropy.detach()

    result = compute_impala_loss_core(
        learner=learner,
        batch=batch,
        inputs=inputs,
        action_logp=reductions.action_logp,
        entropy=reductions.entropy,
        batch_value=lambda source, key: source.get(key),
    )

    assert result.context is inputs.context
    assert "targets" in result.context
    assert "advantages" in result.context
    assert "vtrace_rhos" in result.context
    assert "rewards" in result.context
    assert "trajectory_retention_loss" in result.context
    assert result.context["policy_train_mask"].tolist() == [[1.0], [0.0]]
    assert result.context["value_train_mask"].tolist() == [[0.0], [1.0]]
    assert result.metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert result.metrics["value_train_fraction"] == pytest.approx(0.5)
    assert result.metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert result.metrics["trajectory_retention_weighted_loss"] > 0.0
    assert result.metrics["loss"] == pytest.approx(float(result.total_loss.detach()))


def test_compute_impala_loss_pipeline_records_action_reductions_and_core_context() -> None:
    torch.manual_seed(0)
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), structured_metrics_mode="off")
    batch = _simple_training_batch()

    loss, metrics, context = compute_impala_loss_and_metrics_with_context(
        learner=learner,
        batch=batch,
        batch_value=lambda source, key: source.get(key),
    )

    assert metrics["loss"] == pytest.approx(float(loss.detach()))
    assert context["action_logp"].shape == torch.Size((2, 1))
    assert context["entropy"].shape == torch.Size((2, 1))
    assert "targets" in context
    assert "advantages" in context
    assert "vtrace_rhos" in context
    assert "policy_train_mask" in context
    assert not context["action_logp"].requires_grad
    assert not context["entropy"].requires_grad
    assert metrics["policy_train_fraction"] == pytest.approx(1.0)


def test_impala_learner_raw_vtrace_inputs_use_current_policy_logp_for_importance_weights() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": (action_logp - 2.0).cpu().numpy().astype(np.float32),
        "behavior_values": values.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, metrics = learner._loss_and_metrics(raw_batch)

    assert metrics["vtrace_rho_p50"] > 7.0
    assert metrics["vtrace_rho_p95"] > 7.0
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(1.0)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)


def test_impala_learner_raw_vtrace_uses_behavior_logp_on_non_train_rows_dense() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), vtrace_rho_bar=10.0, vtrace_c_bar=10.0)
    batch = _simple_training_batch()

    with torch.no_grad():
        logits, _values = learner._forward_time_major(torch.from_numpy(batch["obs"]))
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(batch["legal_mask"]),
            torch.from_numpy(batch["actions"]),
            pass_action_id=None,
        )
    behavior_logp = action_logp.clone()
    behavior_logp[1, 0] = behavior_logp[1, 0] - 3.0

    raw_batch = {
        "obs": batch["obs"],
        "actions": batch["actions"],
        "legal_mask": batch["legal_mask"],
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    torch.testing.assert_close(context["vtrace_rhos"][0, 0], torch.tensor(1.0))
    torch.testing.assert_close(context["vtrace_rhos"][1, 0], torch.tensor(1.0))
    assert context["policy_train_mask"].tolist() == [[1.0], [0.0]]


def test_resolve_impala_vtrace_targets_preserves_off_policy_train_rows_and_masks_non_train_rows() -> None:
    values = torch.zeros((2, 1), dtype=torch.float32)
    action_logp = torch.tensor([[0.0], [-1.0]], dtype=torch.float32)
    behavior_logp = torch.tensor([[-2.0], [-4.0]], dtype=torch.float32)
    loss_mask = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    def float_target(value: Any, *, expected_shape: torch.Size, like: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        assert tensor.shape == expected_shape
        return tensor

    def resolve_bootstrap(_batch: Any, *, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        return torch.zeros((batch_size,), dtype=like.dtype, device=like.device)

    resolved = resolve_impala_vtrace_targets(
        batch={
            "rewards": torch.zeros((2, 1), dtype=torch.float32),
            "discounts": torch.ones((2, 1), dtype=torch.float32),
            "behavior_logp": behavior_logp,
        },
        vtrace_result=None,
        values=values,
        action_logp=action_logp,
        loss_mask=loss_mask,
        rho_bar=10.0,
        c_bar=10.0,
        float_target=float_target,
        resolve_bootstrap_value=resolve_bootstrap,
        batch_value=lambda batch, key: batch.get(key),
    )

    torch.testing.assert_close(resolved.action_logp, torch.tensor([[0.0], [-4.0]]))
    torch.testing.assert_close(resolved.behavior_logp_for_mask, behavior_logp)
    assert resolved.rhos_for_metrics[0, 0] == pytest.approx(float(np.exp(2.0)))
    assert resolved.rhos_for_metrics[1, 0] == pytest.approx(1.0)
    assert resolved.targets.requires_grad is False
    assert resolved.advantages.requires_grad is False


def test_impala_learner_trains_value_on_non_policy_rows_by_default() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["policy_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_train_fraction"] == pytest.approx(1.0)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert context["value_train_mask"].tolist() == [[1.0], [1.0]]


def test_impala_learner_accepts_explicit_value_train_mask() -> None:
    torch.manual_seed(0)

    model = TinyPolicyValueModel(observation_dim=2, action_dim=2)
    with torch.no_grad():
        model.value.weight.zero_()
        model.value.bias.zero_()
    learner = ImpalaLearner(model=model, value_loss_coef=1.0, entropy_coef=0.0)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "policy_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "value_train_mask": np.asarray([[True], [False]], dtype=np.bool_),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.0], [2.0]], dtype=np.float32),
            pg_advantages=np.asarray([[0.0], [0.0]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    assert metrics["value_train_fraction"] == pytest.approx(0.5)
    assert metrics["value_loss"] == pytest.approx(0.0)
    assert context["value_train_mask"].tolist() == [[1.0], [0.0]]


def test_impala_learner_raw_vtrace_inputs_use_current_learner_values_for_targets() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, -0.5]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)

    with torch.no_grad():
        forward = learner._forward_time_major(torch.from_numpy(obs))
        logits = forward.logits
        assert logits is not None
        values = forward.values
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )

    log_rho = -0.2
    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": (action_logp - log_rho).cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), 123.0, dtype=np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
        "vtrace_rho_bar": 2.4,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    expected_rho = float(np.exp(log_rho))
    expected_targets = values.detach() * (1.0 - expected_rho)
    assert torch.allclose(context["targets"], expected_targets, atol=1.0e-6)


def test_impala_learner_raw_vtrace_inputs_can_bootstrap_from_current_model() -> None:
    torch.manual_seed(0)

    learner = ImpalaLearner(model=SeatAwareTinyPolicyValueModel(action_dim=2))
    obs = np.asarray([[[1.0, 0.0]]], dtype=np.float32)
    actions = np.asarray([[0]], dtype=np.int64)
    legal_mask = np.ones((1, 1, 2), dtype=np.uint8)
    to_play_seat = np.asarray([[0]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    bootstrap_obs = np.asarray([[2.0, 0.0]], dtype=np.float32)
    bootstrap_actor = np.asarray([1], dtype=np.int64)
    final_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            to_play_seat=to_play_seat,
            initial_hidden_state=initial_hidden_state,
        )
        logits = forward.logits
        assert logits is not None
        action_logp, _entropy = _masked_action_logp_and_entropy(
            logits,
            torch.from_numpy(legal_mask),
            torch.from_numpy(actions),
            pass_action_id=None,
        )
        model = cast(Any, learner.model)
        expected_bootstrap = model.value_seat_aware(
            torch.from_numpy(bootstrap_obs),
            torch.from_numpy(bootstrap_actor),
            torch.from_numpy(final_hidden_state),
        )

    raw_batch = {
        "obs": obs,
        "actions": actions,
        "legal_mask": legal_mask,
        "to_play_seat": to_play_seat,
        "actor": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((1, 1), dtype=np.float32),
        "discounts": np.ones((1, 1), dtype=np.float32),
        "behavior_logp": action_logp.cpu().numpy().astype(np.float32),
        "behavior_values": np.full((1, 1), -77.0, dtype=np.float32),
        "bootstrap_value": np.full((1,), 123.0, dtype=np.float32),
        "bootstrap_obs": bootstrap_obs,
        "bootstrap_actor": bootstrap_actor,
        "final_hidden_state": final_hidden_state,
        "vtrace_rho_bar": 1.0,
        "vtrace_c_bar": 1.0,
    }

    _loss, _metrics, context = learner._loss_and_metrics_with_context(raw_batch)

    assert torch.allclose(context["targets"], expected_bootstrap.reshape(1, 1), atol=1.0e-6)


def test_impala_learner_packed_raw_vtrace_rho_is_one_when_behavior_matches_policy() -> None:
    action_catalog = _teacher_aux_catalog()
    model = TrunkStructuredTeacherModel(action_catalog)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.policy.bias.zero_()
        model.policy.bias[0] = -1.0
        model.policy.bias[5] = 2.5
        model.policy.bias[10] = -0.5
        model.policy.bias[11] = 1.5
        model.policy.bias[12] = -2.0
        model.policy.bias[action_catalog.pass_action_id] = -3.0
    learner = ImpalaLearner(
        model=model,
        profile_timers=True,
        structured_metrics_mode="off",
        teacher_aux_mode="off",
        pass_action_id=action_catalog.pass_action_id,
        vtrace_rho_bar=10.0,
        vtrace_c_bar=10.0,
    )
    learner._active_timing_metrics = {}
    packed_ids = np.asarray(
        [0, 5, action_catalog.pass_action_id, 10, 11, 12, action_catalog.pass_action_id], dtype=np.uint32
    )
    packed_offsets = np.asarray([0, 3, 7], dtype=np.uint32)
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)
    obs = np.asarray([[[1.0, 0.0]], [[0.25, -0.5]]], dtype=np.float32)
    actions = np.asarray([[5], [11]], dtype=np.int64)
    to_play_seat = np.asarray([[0], [1]], dtype=np.int64)
    initial_hidden_state = np.zeros((1, 2, 1), dtype=np.float32)
    legal_actions = LegalActionBatch.from_packed(
        packed_ids,
        packed_offsets,
        meta=packed_meta,
        action_space=action_catalog.action_space_size,
    )

    with torch.no_grad():
        forward = learner._forward_time_major(
            torch.from_numpy(obs),
            initial_hidden_state=initial_hidden_state,
            to_play_seat=to_play_seat,
            legal_actions=legal_actions,
        )
        assert forward.packed_logits is not None
        behavior_logp, _entropy = packed_scores_action_logp_and_entropy(
            forward.packed_logits,
            torch.as_tensor(packed_ids, dtype=torch.long),
            torch.as_tensor(packed_offsets, dtype=torch.long),
            torch.from_numpy(actions),
            pass_action_id=action_catalog.pass_action_id,
        )
    learner._active_timing_metrics = {}
    batch = {
        "obs": obs,
        "actions": actions,
        "legal_actions": legal_actions,
        "legal_ids": packed_ids,
        "legal_offsets": packed_offsets,
        "legal_action_meta": packed_meta,
        "to_play_seat": to_play_seat,
        "initial_hidden_state": initial_hidden_state,
        "rewards": np.zeros((2, 1), dtype=np.float32),
        "discounts": np.ones((2, 1), dtype=np.float32),
        "behavior_logp": behavior_logp.cpu().numpy().astype(np.float32),
        "bootstrap_value": np.zeros((1,), dtype=np.float32),
    }

    _loss, metrics, context = learner._loss_and_metrics_with_context(batch)

    torch.testing.assert_close(context["action_logp"], behavior_logp)
    torch.testing.assert_close(context["vtrace_rhos"], torch.ones_like(context["vtrace_rhos"]))
    assert metrics["target_behavior_logp_delta_abs_p99"] == pytest.approx(0.0)
    assert metrics["target_behavior_train_logp_delta_abs_p99"] == pytest.approx(0.0)
