from __future__ import annotations

from .test_impala_learner import (
    Any,
    ImpalaLearner,
    ImpalaMetricAssemblyRequest,
    ImpalaStructuredSummaryRequest,
    SimpleNamespace,
    TinyPolicyValueModel,
    VTraceTargets,
    _chosen_action_outcome_metrics,
    _FiniteRecorder,
    _mulligan_metric_catalog,
    _packed_ids_from_mask,
    _packed_meta_from_ids,
    _structured_metric_catalog,
    assemble_impala_loss_metrics,
    build_impala_loss_metrics,
    cast,
    compute_impala_structured_policy_summary,
    finalize_impala_loss_context,
    finalize_impala_loss_context_stage,
    impala_loss_context_stage,
    np,
    pytest,
    summarize_structured_policy_metrics,
    torch,
)


def test_finalize_impala_loss_context_records_losses_and_finite_checks() -> None:
    learner = _FiniteRecorder()
    context: dict[str, Any] = {}
    factorized_result = SimpleNamespace(family_log_probs=torch.log_softmax(torch.ones((1, 1, 3)), dim=-1))
    policy_loss = torch.tensor(0.5)
    value_loss = torch.tensor(1.0)
    entropy_mean = torch.tensor(0.25)
    total_loss = torch.tensor(1.375)
    policy_anchor_loss = torch.tensor(0.125)

    finalize_impala_loss_context(
        learner=learner,
        batch={"batch": True},
        context=context,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
        total_loss=total_loss,
        policy_anchor_loss=policy_anchor_loss,
        factorized_result=factorized_result,
    )

    assert context["policy_loss"] is not policy_loss
    torch.testing.assert_close(context["policy_loss"], policy_loss)
    torch.testing.assert_close(context["value_loss"], value_loss)
    torch.testing.assert_close(context["entropy_mean"], entropy_mean)
    torch.testing.assert_close(context["policy_anchor_loss"], policy_anchor_loss)
    torch.testing.assert_close(context["total_loss"], total_loss)
    torch.testing.assert_close(context["factorized_family_log_probs"], factorized_result.family_log_probs)
    assert [name for name, _tensor in learner.calls] == [
        "policy_loss",
        "value_loss",
        "entropy_mean",
        "total_loss",
    ]


def test_finalize_impala_loss_context_stage_maps_objective_anchor_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = object()
    batch = {"context_stage_batch": True}
    context: dict[str, Any] = {"existing": torch.tensor(1.0)}
    factorized_result = SimpleNamespace(family_log_probs=torch.zeros((1, 1, 3), dtype=torch.float32))
    inputs = SimpleNamespace(
        context=context,
        factorized_result=factorized_result,
    )
    policy_loss = torch.tensor(0.5, dtype=torch.float32)
    value_loss = torch.tensor(1.25, dtype=torch.float32)
    entropy_mean = torch.tensor(0.125, dtype=torch.float32)
    total_loss = torch.tensor(3.0, dtype=torch.float32)
    policy_anchor_loss = torch.tensor(0.25, dtype=torch.float32)
    objective_losses = SimpleNamespace(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_mean=entropy_mean,
    )
    policy_anchor_stage = SimpleNamespace(policy_anchor_loss=policy_anchor_loss)
    captured: dict[str, Any] = {}

    def fake_finalize_impala_loss_context(**kwargs: Any) -> None:
        captured.update(kwargs)
        kwargs["context"]["finalized"] = torch.tensor(1.0)

    monkeypatch.setattr(
        impala_loss_context_stage,
        "finalize_impala_loss_context",
        fake_finalize_impala_loss_context,
    )

    result_context = finalize_impala_loss_context_stage(
        learner=learner,
        batch=batch,
        inputs=cast(Any, inputs),
        total_loss=total_loss,
        objective_losses=cast(Any, objective_losses),
        policy_anchor_stage=cast(Any, policy_anchor_stage),
    )

    assert result_context is context
    assert captured["learner"] is learner
    assert captured["batch"] is batch
    assert captured["context"] is context
    assert captured["policy_loss"] is policy_loss
    assert captured["value_loss"] is value_loss
    assert captured["entropy_mean"] is entropy_mean
    assert captured["total_loss"] is total_loss
    assert captured["policy_anchor_loss"] is policy_anchor_loss
    assert captured["factorized_result"] is factorized_result
    torch.testing.assert_close(context["existing"], torch.tensor(1.0))
    torch.testing.assert_close(context["finalized"], torch.tensor(1.0))


def test_assemble_impala_loss_metrics_preserves_base_metric_inputs_and_backfill_fields() -> None:
    batch = {
        "terminal_outcome_backfill_count": 3,
        "terminal_outcome_backfill_total_micros": 12.5,
        "terminal_outcome_trace_backfill_count": 2,
        "terminal_outcome_trace_backfill_total_micros": 7.25,
    }

    metrics = assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=torch.tensor(1.5),
            policy_loss=torch.tensor(0.25),
            value_loss=torch.tensor(2.0),
            entropy_mean=torch.tensor(0.125),
            entropy_scope="candidate",
            loss_mask=torch.as_tensor([[1.0], [0.0]], dtype=torch.float32),
            value_loss_mask=torch.as_tensor([[1.0], [1.0]], dtype=torch.float32),
            actions=torch.as_tensor([[0], [1]], dtype=torch.long),
            action_logp=torch.as_tensor([[-0.2], [-0.3]], dtype=torch.float32),
            behavior_logp_for_mask=torch.as_tensor([[-0.1], [-0.5]], dtype=torch.float32),
            rewards_for_metrics=torch.as_tensor([[1.0], [-1.0]], dtype=torch.float32),
            advantages=torch.as_tensor([[0.5], [-0.25]], dtype=torch.float32),
            targets=torch.as_tensor([[1.25], [-0.75]], dtype=torch.float32),
            rhos_for_metrics=torch.as_tensor([[1.0], [2.0]], dtype=torch.float32),
            rho_bar=1.5,
            c_bar=1.25,
            action_catalog=object(),
            pass_action_id=1,
            trajectory_retention_metrics={"trajectory_retention_rows": 1.0},
            policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25},
            teacher_metrics={"teacher_aux_loss": 0.5},
            emit_structured_metrics=True,
            batch=batch,
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
        record_timing_ms=lambda _name, _duration: pytest.fail("non-structured catalog must not summarize"),
    )

    assert metrics["loss"] == pytest.approx(1.5)
    assert metrics["policy_loss"] == pytest.approx(0.25)
    assert metrics["value_loss"] == pytest.approx(2.0)
    assert metrics["entropy"] == pytest.approx(0.125)
    assert metrics["terminal_outcome_backfill_count"] == pytest.approx(3.0)
    assert metrics["terminal_outcome_backfill_total_micros"] == pytest.approx(12.5)
    assert metrics["terminal_outcome_trace_backfill_count"] == pytest.approx(2.0)
    assert metrics["terminal_outcome_trace_backfill_total_micros"] == pytest.approx(7.25)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(0.25)
    assert metrics["teacher_aux_loss"] == pytest.approx(0.5)
    assert "structured_exact_action_concentration" not in metrics


def test_assemble_impala_loss_metrics_merges_structured_summary_with_dense_fallback() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, action_catalog.action_space_size), -20.0)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, [0, 7, action_catalog.pass_action_id]] = True
    legal_mask[1, 0, [4, 7, action_catalog.pass_action_id]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, action_catalog.pass_action_id] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, action_catalog.pass_action_id] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    timings: list[tuple[str, float]] = []
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    batch = {"marker": True}

    metrics = assemble_impala_loss_metrics(
        ImpalaMetricAssemblyRequest(
            total_loss=torch.tensor(0.0),
            policy_loss=torch.tensor(0.0),
            value_loss=torch.tensor(0.0),
            entropy_mean=torch.tensor(0.0),
            entropy_scope="family",
            loss_mask=torch.ones((2, 1), dtype=torch.float32),
            value_loss_mask=torch.ones((2, 1), dtype=torch.float32),
            actions=torch.as_tensor([[0], [4]], dtype=torch.long),
            action_logp=torch.zeros((2, 1), dtype=torch.float32),
            behavior_logp_for_mask=None,
            rewards_for_metrics=torch.zeros((2, 1), dtype=torch.float32),
            advantages=torch.zeros((2, 1), dtype=torch.float32),
            targets=torch.zeros((2, 1), dtype=torch.float32),
            rhos_for_metrics=torch.ones((2, 1), dtype=torch.float32),
            rho_bar=1.0,
            c_bar=1.0,
            action_catalog=action_catalog,
            pass_action_id=action_catalog.pass_action_id,
            emit_structured_metrics=True,
            logits=logits,
            legal_mask=None,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                None,
            ),
            batch=batch,
            expected_shape=torch.Size((2, 1)),
            action_dim=action_catalog.action_space_size,
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
                resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
            ),
        ),
        batch_value=lambda source_batch, key: source_batch.get(key),
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )

    expected_structured = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    assert resolver_calls == [(batch, torch.Size((2, 1)), action_catalog.action_space_size)]
    assert metrics["entropy_scope_family_active"] == pytest.approx(1.0)
    assert metrics["structured_exact_action_concentration"] == pytest.approx(
        expected_structured["structured_exact_action_concentration"]
    )
    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(
        expected_structured["structured_main_move_0_2_top1_rate"]
    )
    assert [name for name, _duration in timings] == ["learner_structured_summary"]


def test_summarize_structured_policy_metrics_reports_mainmove_pressure() -> None:
    action_catalog = _structured_metric_catalog()
    main_move_02_action = next(
        action_id
        for action_id in range(action_catalog.action_space_size)
        if (
            action_catalog.decode(action_id).family == "main_move"
            and action_catalog.decode(action_id).from_slot == 0
            and action_catalog.decode(action_id).to_slot == 2
        )
    )
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)

    legal_mask[0, 0, [0, main_move_02_action, 25]] = True
    logits[0, 0, 0] = 0.0
    logits[0, 0, main_move_02_action] = 2.0
    logits[0, 0, 25] = 1.0

    legal_mask[1, 0, [0, main_move_02_action, 25]] = True
    logits[1, 0, 0] = 3.0
    logits[1, 0, main_move_02_action] = 0.0
    logits[1, 0, 25] = 1.0

    metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)

    assert metrics["structured_main_move_0_2_top1_rate"] == pytest.approx(0.5)
    assert 0.0 < metrics["structured_main_move_share_when_play_available"] < 1.0
    assert (
        metrics["structured_main_play_character_mass"]
        + metrics["structured_main_move_mass"]
        + metrics["structured_pass_mass"]
    ) == pytest.approx(1.0)
    assert 0.0 < metrics["structured_exact_action_concentration"] <= 1.0


def test_summarize_structured_policy_metrics_matches_packed_meta_path() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5

    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    dense_metrics = summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    packed_metrics = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )

    assert packed_metrics == pytest.approx(dense_metrics)


def test_compute_impala_structured_policy_summary_resolves_dense_mask_when_packed_meta_missing() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    timings: list[tuple[str, float]] = []
    resolver_calls: list[tuple[Any, torch.Size, int]] = []
    batch = object()

    metrics = compute_impala_structured_policy_summary(
        ImpalaStructuredSummaryRequest(
            logits=logits,
            legal_mask=None,
            action_catalog=action_catalog,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                None,
            ),
            batch=batch,
            expected_shape=torch.Size((2, 1)),
            action_dim=26,
            resolve_legal_mask=lambda source_batch, expected_shape, action_dim: (
                resolver_calls.append((source_batch, expected_shape, action_dim)) or legal_mask
            ),
        ),
        record_timing_ms=lambda name, duration: timings.append((name, duration)),
    )

    assert resolver_calls == [(batch, torch.Size((2, 1)), 26)]
    assert metrics == pytest.approx(
        summarize_structured_policy_metrics(logits, legal_mask, action_catalog=action_catalog)
    )
    assert [name for name, _duration in timings] == ["learner_structured_summary"]
    assert timings[0][1] >= 0.0


def test_compute_impala_structured_policy_summary_keeps_packed_meta_path_without_dense_mask() -> None:
    action_catalog = _structured_metric_catalog()
    logits = torch.full((2, 1, 26), -20.0)
    legal_mask = torch.zeros((2, 1, 26), dtype=torch.bool)
    legal_mask[0, 0, [0, 7, 25]] = True
    legal_mask[1, 0, [4, 7, 25]] = True
    logits[0, 0, 0] = 1.5
    logits[0, 0, 7] = 2.0
    logits[0, 0, 25] = 0.5
    logits[1, 0, 4] = 2.5
    logits[1, 0, 7] = 0.0
    logits[1, 0, 25] = 0.5
    packed_ids, packed_offsets = _packed_ids_from_mask(legal_mask.numpy().astype(np.uint8, copy=False))
    packed_meta = _packed_meta_from_ids(action_catalog, packed_ids)

    metrics = compute_impala_structured_policy_summary(
        ImpalaStructuredSummaryRequest(
            logits=logits,
            legal_mask=None,
            action_catalog=action_catalog,
            packed_legal=(
                torch.as_tensor(packed_ids, dtype=torch.long),
                torch.as_tensor(packed_offsets, dtype=torch.long),
                torch.as_tensor(packed_meta, dtype=torch.long),
            ),
            resolve_legal_mask=lambda _source_batch, _expected_shape, _action_dim: pytest.fail(
                "packed metadata path should not reconstruct a dense mask"
            ),
        ),
        record_timing_ms=lambda _name, _duration: None,
    )

    expected = summarize_structured_policy_metrics(
        logits,
        None,
        action_catalog=action_catalog,
        packed_ids=torch.as_tensor(packed_ids, dtype=torch.long),
        packed_offsets=torch.as_tensor(packed_offsets, dtype=torch.long),
        packed_meta=torch.as_tensor(packed_meta, dtype=torch.long),
    )
    assert metrics == pytest.approx(expected)


def test_impala_learner_reports_reward_advantage_and_chosen_action_metrics() -> None:
    learner = ImpalaLearner(model=TinyPolicyValueModel(action_dim=2), pass_action_id=1)
    batch = {
        "obs": np.asarray([[[1.0, 0.0]], [[0.5, -0.5]]], dtype=np.float32),
        "actions": np.asarray([[0], [1]], dtype=np.int64),
        "legal_mask": np.ones((2, 1, 2), dtype=np.uint8),
        "vtrace_result": VTraceTargets(
            vs=np.asarray([[0.25], [-0.5]], dtype=np.float32),
            pg_advantages=np.asarray([[1.5], [-0.25]], dtype=np.float32),
            rhos=np.ones((2, 1), dtype=np.float32),
        ),
        "rewards": np.asarray([[0.0], [1.0]], dtype=np.float32),
    }

    _loss, metrics = learner._loss_and_metrics(batch)

    assert metrics["reward_mean"] == pytest.approx(0.5)
    assert metrics["reward_std"] == pytest.approx(0.5)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["reward_min"] == pytest.approx(0.0)
    assert metrics["reward_max"] == pytest.approx(1.0)
    assert metrics["reward_nonzero_fraction"] == pytest.approx(0.5)
    assert metrics["reward_positive_fraction"] == pytest.approx(0.5)
    assert metrics["reward_negative_fraction"] == pytest.approx(0.0)
    assert metrics["advantage_mean"] == pytest.approx(0.625)
    assert metrics["advantage_abs_mean"] == pytest.approx(0.875)
    assert metrics["target_mean"] == pytest.approx(-0.125)
    assert metrics["target_abs_mean"] == pytest.approx(0.375)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.5)
    assert metrics["chosen_pass_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_pass_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_nonpass_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)


def test_impala_loss_metrics_builder_preserves_training_diagnostics_contract() -> None:
    metrics = build_impala_loss_metrics(
        total_loss=torch.tensor(2.0),
        policy_loss=torch.tensor(0.5),
        value_loss=torch.tensor(1.25),
        entropy_mean=torch.tensor(0.125),
        entropy_scope="family",
        loss_mask=torch.tensor([[1.0], [0.0]]),
        value_loss_mask=torch.tensor([[1.0], [1.0]]),
        actions=torch.tensor([[0], [1]], dtype=torch.long),
        action_logp=torch.tensor([[-0.2], [-0.3]]),
        behavior_logp_for_mask=torch.tensor([[-0.5], [-0.3]]),
        rewards_for_metrics=torch.tensor([[0.0], [1.0]]),
        advantages=torch.tensor([[1.5], [-0.25]]),
        targets=torch.tensor([[0.25], [-0.5]]),
        rhos_for_metrics=torch.tensor([[2.0], [4.0]]),
        rho_bar=3.0,
        c_bar=1.5,
        action_catalog=None,
        pass_action_id=1,
        terminal_outcome_backfill_count=7,
        terminal_outcome_backfill_total_micros=11,
        terminal_outcome_trace_backfill_count=13,
        terminal_outcome_trace_backfill_total_micros=17,
        trajectory_retention_metrics={"trajectory_retention_rows": 1.0},
        policy_anchor_metrics={"policy_anchor_weighted_loss": 0.25},
        teacher_metrics={"teacher_valid_fraction": 0.5},
    )

    assert metrics["entropy_scope_family_active"] == pytest.approx(1.0)
    assert metrics["reward_abs_mean"] == pytest.approx(0.5)
    assert metrics["vtrace_rho_mean"] == pytest.approx(3.0)
    assert metrics["vtrace_train_rho_mean"] == pytest.approx(2.0)
    assert metrics["vtrace_rho_clip_rate"] == pytest.approx(0.5)
    assert metrics["vtrace_c_clip_rate"] == pytest.approx(1.0)
    assert metrics["target_behavior_logp_delta_abs_mean"] == pytest.approx(0.15)
    assert metrics["target_behavior_train_logp_delta_abs_mean"] == pytest.approx(0.3)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.0)
    assert metrics["chosen_nonpass_train_advantage_mean"] == pytest.approx(1.5)
    assert metrics["terminal_outcome_backfill_count"] == pytest.approx(7.0)
    assert metrics["terminal_outcome_trace_backfill_total_micros"] == pytest.approx(17.0)
    assert metrics["trajectory_retention_rows"] == pytest.approx(1.0)
    assert metrics["policy_anchor_weighted_loss"] == pytest.approx(0.25)
    assert metrics["teacher_valid_fraction"] == pytest.approx(0.5)


def test_impala_learner_reports_family_chosen_action_outcome_metrics() -> None:
    catalog = _mulligan_metric_catalog()

    metrics = _chosen_action_outcome_metrics(
        actions=torch.tensor([[0], [1], [3], [5], [8]], dtype=torch.long),
        loss_mask=torch.tensor([[True], [True], [False], [True], [True]]),
        rewards=torch.tensor([[0.0], [1.0], [99.0], [2.0], [3.0]], dtype=torch.float32),
        advantages=torch.tensor([[0.5], [-0.25], [99.0], [1.25], [-1.0]], dtype=torch.float32),
        action_catalog=catalog,
        pass_action_id=catalog.pass_action_id,
    )

    assert metrics["chosen_mulligan_confirm_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_confirm_train_reward_mean"] == pytest.approx(0.0)
    assert metrics["chosen_mulligan_confirm_train_advantage_mean"] == pytest.approx(0.5)
    assert metrics["chosen_mulligan_select_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_mulligan_select_train_reward_mean"] == pytest.approx(1.0)
    assert metrics["chosen_mulligan_select_train_advantage_mean"] == pytest.approx(-0.25)
    assert metrics["chosen_attack_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_attack_train_advantage_mean"] == pytest.approx(1.25)
    assert metrics["chosen_pass_train_fraction"] == pytest.approx(0.25)
    assert metrics["chosen_main_play_character_train_fraction"] == pytest.approx(0.0)
