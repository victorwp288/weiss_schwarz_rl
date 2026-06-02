from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from weiss_rl.core.action_catalog import ActionCatalog
from weiss_rl.learners.structured_auxiliary import (
    active_public_heuristic_profiles,
    dense_group_log_probs,
    mix_public_heuristic_profile_logits,
    normalize_public_heuristic_profile_mode,
    normalize_public_heuristic_profiles,
    packed_group_log_probs,
    packed_soft_target_cross_entropy,
    packed_structured_legal_view,
    resolve_public_heuristic_family_ids,
    score_public_heuristic_target_logits,
    selected_public_heuristic_profiles,
    structured_catalog_metadata,
    structured_group_lookup,
)
from weiss_rl.learners.structured_teacher.auxiliary import compute_structured_teacher_auxiliary_metrics
from weiss_rl.learners.structured_teacher.common import (
    StructuredTeacherAuxiliaryCoefficients,
    StructuredTeacherAuxiliaryLossTerms,
    exact_action_family_rows,
    finalize_structured_teacher_auxiliary_loss,
    flatten_structured_teacher_labels,
)


def _catalog() -> ActionCatalog:
    return ActionCatalog.from_spec_bundle(
        {
            "action": {
                "action_encoding_version": 1,
                "action_space_size": 20,
                "pass_action_id": 19,
                "constants": [["MAX_HAND", 2], ["MAX_STAGE", 5], ["ATTACK_SLOT_COUNT", 1]],
                "families": [
                    {"name": "main_play_character", "base": 0, "count": 10},
                    {"name": "attack", "base": 10, "count": 3},
                    {"name": "main_move", "base": 13, "count": 6},
                    {"name": "pass", "base": 19, "count": 1},
                ],
                "attack_type_encoding": [["frontal", 0], ["direct", 1], ["side", 2]],
            }
        }
    )


def test_flatten_structured_teacher_labels_normalizes_time_major_labels() -> None:
    labels = flatten_structured_teacher_labels(
        loss_mask=torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
        teacher_family=torch.tensor([[0, 1], [2, -1]], dtype=torch.int16),
        teacher_slot=torch.tensor([[3, 4], [5, -1]], dtype=torch.int16),
        teacher_move_source=torch.tensor([[1, 2], [3, -1]], dtype=torch.int16),
        teacher_attack_type=torch.tensor([[0, 1], [2, -1]], dtype=torch.int16),
        teacher_action=torch.tensor([[10, 11], [12, -1]], dtype=torch.int16),
        teacher_valid=torch.tensor([[1, 1], [0, 1]], dtype=torch.int64),
    )

    assert labels.loss_mask.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert labels.loss_mask.dtype == torch.float32
    assert labels.family.dtype == torch.long
    assert labels.slot.dtype == torch.long
    assert labels.move_source is not None
    assert labels.move_source.tolist() == [1, 2, 3, -1]
    assert labels.attack_type.tolist() == [0, 1, 2, -1]
    assert labels.action is not None
    assert labels.action.tolist() == [10, 11, 12, -1]
    assert labels.valid.dtype == torch.bool
    assert labels.valid.tolist() == [True, True, False, True]


def test_exact_action_family_rows_uses_one_shared_mask_for_teacher_branches() -> None:
    family_names = ("main_play_character", "attack", "main_move", "pass")
    flat_teacher_family = torch.tensor([0, 1, 2, 3, -1], dtype=torch.long)

    expected = torch.tensor([False, True, True, False, False])
    for _branch_name in ("dense", "packed", "factorized"):
        rows = exact_action_family_rows(
            flat_teacher_family=flat_teacher_family,
            family_names=family_names,
            exact_action_families=("attack", "main_move"),
        )
        assert rows is not None
        assert torch.equal(rows, expected)

    assert (
        exact_action_family_rows(
            flat_teacher_family=flat_teacher_family,
            family_names=family_names,
            exact_action_families=(),
        )
        is None
    )


def test_finalize_structured_teacher_auxiliary_loss_uses_all_nonzero_coefficients() -> None:
    terms = StructuredTeacherAuxiliaryLossTerms(
        family=torch.tensor(1.0),
        slot=torch.tensor(2.0),
        hand=torch.tensor(3.0),
        move_source=torch.tensor(4.0),
        attack_type=torch.tensor(5.0),
        action=torch.tensor(6.0),
        same_family_action=torch.tensor(7.0),
        action_margin=torch.tensor(8.0),
        same_family_action_margin=torch.tensor(9.0),
        public_heuristic=torch.tensor(10.0),
        public_nonpass_over_pass=torch.tensor(11.0),
    )
    coefs = StructuredTeacherAuxiliaryCoefficients(
        family=0.1,
        slot=0.2,
        hand=0.3,
        move_source=0.4,
        attack_type=0.5,
        action=0.6,
        same_family_action=0.7,
        action_margin=0.8,
        same_family_action_margin=0.9,
        public_heuristic=1.0,
        public_nonpass_over_pass=1.1,
    )
    metrics: dict[str, float] = {}
    context: dict[str, torch.Tensor] = {}

    total = finalize_structured_teacher_auxiliary_loss(
        terms=terms,
        coefs=coefs,
        metrics=metrics,
        context=context,
        value_dtype=torch.float64,
    )

    expected = sum(
        value * coef
        for value, coef in zip(range(1, 12), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1), strict=True)
    )
    assert total.dtype == torch.float64
    assert total.item() == pytest.approx(expected)
    assert metrics["teacher_aux_loss"] == pytest.approx(expected)
    assert context["teacher_aux_loss"].item() == pytest.approx(expected)


def test_structured_teacher_missing_labels_return_zero_on_logits_dtype_and_device() -> None:
    logits = torch.ones((2, 3), dtype=torch.float64)
    legal_mask = torch.ones((2, 3), dtype=torch.bool)

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=None,
        teacher_slot=None,
        teacher_attack_type=None,
        teacher_action=None,
        teacher_valid=None,
        loss_mask=torch.ones((2,), dtype=torch.float32),
        action_catalog=_catalog(),
        family_coef=1.0,
        slot_coef=1.0,
        attack_type_coef=1.0,
        action_coef=1.0,
        same_family_action_coef=1.0,
    )

    assert aux_loss.item() == pytest.approx(0.0)
    assert aux_loss.dtype == logits.dtype
    assert aux_loss.device == logits.device
    assert metrics["teacher_aux_loss"] == pytest.approx(0.0)
    assert context == {}


def test_public_heuristic_profiles_are_normalized_deduplicated_and_defaulted() -> None:
    assert normalize_public_heuristic_profiles(None) == ("base",)
    assert normalize_public_heuristic_profiles([" Base ", "aggressive", "base", ""]) == ("base", "aggressive")

    with pytest.raises(ValueError, match="unsupported profiles: unknown"):
        normalize_public_heuristic_profiles(("unknown",))


def test_public_heuristic_profile_mode_is_normalized_and_validated() -> None:
    assert normalize_public_heuristic_profile_mode(None) == "mixture"
    assert normalize_public_heuristic_profile_mode(" Cycle ") == "cycle"

    with pytest.raises(ValueError, match="teacher_public_heuristic_profile_mode"):
        normalize_public_heuristic_profile_mode("round_robin")


def test_active_public_heuristic_profiles_default_and_end_update_rules() -> None:
    assert active_public_heuristic_profiles((), update_count=10, end_updates=-1) == ("base",)
    assert active_public_heuristic_profiles(("base", "control"), update_count=10, end_updates=-1) == (
        "base",
        "control",
    )
    assert active_public_heuristic_profiles(("base", "control"), update_count=11, end_updates=10) == ("base",)


def test_selected_public_heuristic_profiles_preserves_cycle_after_end_update_fallback() -> None:
    assert selected_public_heuristic_profiles(
        ("base", "aggressive", "control"),
        profile_mode="cycle",
        update_count=4,
        end_updates=-1,
    ) == ("aggressive",)
    assert selected_public_heuristic_profiles(
        ("base", "aggressive", "control"),
        profile_mode="cycle",
        update_count=4,
        end_updates=3,
    ) == ("base",)
    assert selected_public_heuristic_profiles(
        ("base", "aggressive"),
        profile_mode="mixture",
        update_count=1,
        end_updates=-1,
    ) == ("base", "aggressive")


def test_mix_public_heuristic_profile_logits_preserves_rowwise_mixture_normalization() -> None:
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    profile_a = torch.tensor([3.0, 1.0, 0.0, 2.0, 4.0], dtype=torch.float32)
    profile_b = torch.tensor([1.0, 3.0, 4.0, 2.0, 0.0], dtype=torch.float32)

    mixed = mix_public_heuristic_profile_logits(
        [profile_a, profile_b],
        offsets=offsets,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    expected_rows: list[torch.Tensor] = []
    for start, end in ((0, 2), (2, 5)):
        log_probs = torch.stack(
            [
                profile_a[start:end] - torch.logsumexp(profile_a[start:end], dim=0),
                profile_b[start:end] - torch.logsumexp(profile_b[start:end], dim=0),
            ],
            dim=0,
        )
        expected_rows.append(torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(mixed, torch.cat(expected_rows))


def test_mix_public_heuristic_profile_logits_handles_empty_and_single_profile_cases() -> None:
    single = torch.tensor([1.0, 2.0])

    assert (
        mix_public_heuristic_profile_logits(
            [],
            offsets=torch.tensor([0], dtype=torch.long),
            temperature=1.0,
            device=torch.device("cpu"),
        ).numel()
        == 0
    )
    assert (
        mix_public_heuristic_profile_logits(
            [single],
            offsets=torch.tensor([0, 2], dtype=torch.long),
            temperature=1.0,
            device=torch.device("cpu"),
        )
        is single
    )
    torch.testing.assert_close(
        mix_public_heuristic_profile_logits(
            [single, single + 1.0],
            offsets=torch.tensor([0], dtype=torch.long),
            temperature=1.0,
            device=torch.device("cpu"),
        ),
        single,
    )


def test_score_public_heuristic_target_logits_selects_profiles_and_passes_context() -> None:
    offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    legal_actions = SimpleNamespace(offsets=offsets)
    obs_rows = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    context = {"stage": torch.ones((2, 1))}
    calls: list[tuple[str, torch.Tensor, object, object]] = []

    class _Scorer:
        def score_packed_public_heuristic_candidates(
            self,
            obs: torch.Tensor,
            legal: object,
            *,
            observation_context: object,
            scoring_profile: str,
        ) -> torch.Tensor:
            calls.append((scoring_profile, obs, legal, observation_context))
            values = {
                "base": torch.tensor([1.0, 0.0, 0.0, 1.0]),
                "aggressive": torch.tensor([0.0, 1.0, 2.0, 0.0]),
                "control": torch.tensor([2.0, 0.0, 0.0, 2.0]),
            }
            return values[scoring_profile]

    logits = score_public_heuristic_target_logits(
        forward_model=_Scorer(),
        obs_rows=obs_rows,
        legal_actions=legal_actions,
        observation_context=context,
        profiles=("base", "aggressive", "control"),
        profile_mode="cycle",
        update_count=1,
        end_updates=-1,
        temperature=1.0,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(logits, torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert len(calls) == 1
    assert calls[0][0] == "aggressive"
    assert calls[0][1] is obs_rows
    assert calls[0][2] is legal_actions
    assert calls[0][3] is context


def test_structured_catalog_metadata_records_family_slots_attacks_and_main_move_pressure() -> None:
    metadata = structured_catalog_metadata(_catalog())

    assert metadata.family_names == ("main_play_character", "attack", "main_move", "pass")
    assert metadata.attack_type_names == ("frontal", "direct", "side")
    assert metadata.family_ids[0] == 0
    assert metadata.family_ids[10] == 1
    assert metadata.family_ids[13] == 2
    assert metadata.play_slots[0] == 0
    assert metadata.play_slots[8] == 3
    assert metadata.attack_slots[10] == 0
    assert metadata.attack_types[11] == 1
    assert metadata.move_from_slots[14] == 0
    assert metadata.move_to_slots[14] == 2
    assert metadata.main_move_02_action_id == 14


def test_structured_group_lookup_builds_dense_action_tables_on_requested_device() -> None:
    catalog = _catalog()

    lookup = structured_group_lookup(catalog, device=torch.device("cpu"))

    assert lookup["family_names"] == ("main_play_character", "attack", "main_move", "pass")
    assert lookup["family_index"] == {"main_play_character": 0, "attack": 1, "main_move": 2, "pass": 3}
    assert lookup["attack_type_names"] == ("frontal", "direct", "side")
    assert lookup["family_ids"].device.type == "cpu"
    assert lookup["family_ids"].tolist()[0] == 0
    assert lookup["play_slots"].tolist()[8] == 3
    assert lookup["move_to_slots"].tolist()[14] == 2
    assert lookup["attack_types"].tolist()[11] == 1


def test_dense_group_log_probs_matches_manual_group_logsumexp() -> None:
    masked_logits = torch.tensor([[2.0, 0.0, -1.0, 4.0], [1.0, 3.0, 5.0, -2.0]], dtype=torch.float32)
    group_ids = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    log_probs = dense_group_log_probs(masked_logits=masked_logits, group_ids=group_ids, group_count=4)

    expected_rows: list[torch.Tensor] = []
    for row in masked_logits:
        row_log_z = torch.logsumexp(row, dim=0)
        expected_rows.append(
            torch.stack(
                [
                    row[0] - row_log_z,
                    torch.logsumexp(row[1:3], dim=0) - row_log_z,
                    row[3] - row_log_z,
                    torch.tensor(-1.0e9, dtype=row.dtype) - row_log_z,
                ]
            )
        )
    torch.testing.assert_close(log_probs, torch.stack(expected_rows, dim=0))


def test_resolve_public_heuristic_family_ids_preserves_order_and_reports_unknowns() -> None:
    family_names = ("main_play_character", "attack", "main_move", "pass")

    assert resolve_public_heuristic_family_ids(
        family_names=family_names,
        requested_families=(" attack ", "main_move"),
    ) == (1, 2)
    assert resolve_public_heuristic_family_ids(family_names=family_names, requested_families=("", " ")) == ()

    with pytest.raises(ValueError, match="unknown action families: climax"):
        resolve_public_heuristic_family_ids(family_names=family_names, requested_families=("climax",))


def test_packed_structured_legal_view_selects_dense_logits_and_normalizes_metadata() -> None:
    unused = torch.iinfo(torch.uint16).max
    logits = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    packed_ids = torch.tensor([0, 2, 1], dtype=torch.long)
    packed_offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 10, unused, unused],
            [2, unused, 20, unused],
            [1, 30, 31, 32],
        ],
        dtype=torch.long,
    )

    view = packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    assert view is not None
    assert view.row_count == 2
    assert view.row_indices.tolist() == [0, 0, 1]
    assert view.action_ids.tolist() == [0, 2, 1]
    torch.testing.assert_close(view.logits, torch.tensor([1.0, 3.0, 5.0]))
    torch.testing.assert_close(view.row_log_z, torch.tensor([torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0), 5.0]))
    assert view.row_has_candidates.tolist() == [True, True]
    assert view.family_ids.tolist() == [0, 2, 1]
    assert view.arg0.tolist() == [10, -1, 30]
    assert view.arg1.tolist() == [-1, 20, 31]
    assert view.arg2.tolist() == [-1, -1, 32]


def test_packed_structured_legal_view_supports_flat_logits_no_logits_and_empty_rows() -> None:
    packed_ids = torch.tensor([1], dtype=torch.long)
    packed_offsets = torch.tensor([0, 0, 1], dtype=torch.long)
    packed_meta = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    flat_view = packed_structured_legal_view(
        logits=torch.tensor([7.0]),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    assert flat_view is not None
    assert flat_view.row_indices.tolist() == [1]
    torch.testing.assert_close(flat_view.logits, torch.tensor([7.0]))
    assert flat_view.row_has_candidates.tolist() == [False, True]
    assert flat_view.row_log_z[0].item() == -torch.inf
    torch.testing.assert_close(flat_view.row_log_z[1], torch.tensor(7.0))

    zero_view = packed_structured_legal_view(
        logits=None,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )
    assert zero_view is not None
    torch.testing.assert_close(zero_view.logits, torch.tensor([0.0]))


def test_packed_structured_legal_view_validates_shapes_and_missing_inputs() -> None:
    packed_ids = torch.tensor([0, 1], dtype=torch.long)
    packed_offsets = torch.tensor([0, 2], dtype=torch.long)
    packed_meta = torch.zeros((2, 4), dtype=torch.long)

    assert (
        packed_structured_legal_view(
            logits=torch.zeros((1, 2)),
            packed_ids=None,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
        is None
    )
    with pytest.raises(ValueError, match="packed logits must align"):
        packed_structured_legal_view(
            logits=torch.tensor([1.0]),
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=packed_meta,
        )
    with pytest.raises(ValueError, match="packed legal offsets must describe 1 rows"):
        packed_structured_legal_view(
            logits=torch.zeros((1, 2)),
            packed_ids=packed_ids,
            packed_offsets=torch.tensor([0, 1, 2], dtype=torch.long),
            packed_meta=packed_meta,
        )
    with pytest.raises(ValueError, match="packed legal metadata must align"):
        packed_structured_legal_view(
            logits=torch.zeros((1, 2)),
            packed_ids=packed_ids,
            packed_offsets=packed_offsets,
            packed_meta=torch.zeros((2, 3), dtype=torch.long),
        )


def test_packed_group_log_probs_handles_groups_masks_and_empty_group_count() -> None:
    view = packed_structured_legal_view(
        logits=torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]),
        packed_ids=torch.tensor([0, 2, 1], dtype=torch.long),
        packed_offsets=torch.tensor([0, 2, 3], dtype=torch.long),
        packed_meta=torch.zeros((3, 4), dtype=torch.long),
    )
    assert view is not None
    group_ids = torch.tensor([0, 1, 0], dtype=torch.long)

    log_probs = packed_group_log_probs(view, group_ids=group_ids, group_count=2)
    row0_z = torch.logsumexp(torch.tensor([1.0, 3.0]), dim=0)
    torch.testing.assert_close(
        log_probs,
        torch.tensor(
            [
                [1.0 - row0_z, 3.0 - row0_z],
                [0.0, -torch.inf],
            ]
        ),
    )

    masked = packed_group_log_probs(
        view,
        group_ids=group_ids,
        group_count=2,
        candidate_mask=torch.tensor([True, False, True]),
    )
    torch.testing.assert_close(masked, torch.tensor([[0.0, -torch.inf], [0.0, -torch.inf]]))
    assert packed_group_log_probs(view, group_ids=group_ids, group_count=0).shape == (2, 0)


def test_packed_soft_target_cross_entropy_matches_manual_row_calculation() -> None:
    view = packed_structured_legal_view(
        logits=torch.tensor([1.0, 3.0, 5.0]),
        packed_ids=torch.tensor([0, 2, 1], dtype=torch.long),
        packed_offsets=torch.tensor([0, 2, 3], dtype=torch.long),
        packed_meta=torch.zeros((3, 4), dtype=torch.long),
    )
    assert view is not None
    target_logits = torch.tensor([2.0, 0.0, 9.0])

    cross_entropy, top_mass, target_entropy = packed_soft_target_cross_entropy(
        view,
        target_logits=target_logits,
        temperature=1.0,
    )

    row0_target_log_probs = target_logits[:2] - torch.logsumexp(target_logits[:2], dim=0)
    row0_target_probs = torch.exp(row0_target_log_probs)
    row0_student_log_probs = view.logits[:2] - torch.logsumexp(view.logits[:2], dim=0)
    row0_cross_entropy = -(row0_target_probs * row0_student_log_probs).sum()
    row0_entropy = -(row0_target_probs * row0_target_log_probs).sum()

    torch.testing.assert_close(cross_entropy, torch.tensor([row0_cross_entropy, 0.0]))
    torch.testing.assert_close(top_mass, torch.tensor([row0_target_probs[1], 1.0]))
    torch.testing.assert_close(target_entropy, torch.tensor([row0_entropy, 0.0]))

    with pytest.raises(ValueError, match="temperature must be > 0"):
        packed_soft_target_cross_entropy(view, target_logits=target_logits, temperature=0.0)
    with pytest.raises(ValueError, match="target logits must align"):
        packed_soft_target_cross_entropy(view, target_logits=target_logits[:2], temperature=1.0)


def test_packed_teacher_action_margin_loss_rewards_explicit_logit_separation() -> None:
    catalog = _catalog()
    logits = torch.tensor([1.0, 2.0, 1.7, 4.0, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([0, 10, 19, 0, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 0, -1, -1],
            [1, 0, 0, -1],
            [3, -1, -1, -1],
            [0, 0, -1, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=2.0,
        action_margin=0.5,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.1))
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.05)
    assert metrics["teacher_action_margin_mean"] == pytest.approx(2.325)
    assert metrics["teacher_action_margin_satisfied_fraction"] == pytest.approx(0.75)
    torch.testing.assert_close(context["teacher_action_margins"], torch.tensor([0.3, 3.0]))


def test_factorized_teacher_action_margin_uses_packed_student_logits() -> None:
    catalog = _catalog()
    logits = torch.tensor([1.0, 2.0, 1.7, 4.0, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([0, 10, 19, 0, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 0, -1, -1],
            [1, 0, 0, -1],
            [3, -1, -1, -1],
            [0, 0, -1, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    packed_view = packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=2.0,
        action_margin=0.5,
        packed_view=packed_view,
        factorized_family_log_probs=torch.zeros((2, 4), dtype=torch.float32),
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.1))
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.05)
    torch.testing.assert_close(context["teacher_action_margins"], torch.tensor([0.3, 3.0]))


def test_packed_teacher_same_family_action_margin_loss_targets_within_family_flatness() -> None:
    catalog = _catalog()
    logits = torch.tensor([2.0, 1.7, 0.0, 4.0, 3.8, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([10, 11, 19, 13, 14, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [1, 0, 0, -1],
            [1, 0, 1, -1],
            [3, -1, -1, -1],
            [2, 0, 1, -1],
            [2, 0, 2, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 2], dtype=torch.long),
        teacher_slot=torch.tensor([0, 1], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 13], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        same_family_action_margin_coef=2.0,
        same_family_action_margin=0.5,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.55))
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.275)
    assert metrics["teacher_same_family_action_margin_mean"] == pytest.approx(0.225)
    assert metrics["teacher_same_family_action_margin_satisfied_fraction"] == pytest.approx(0.0)
    torch.testing.assert_close(context["teacher_same_family_action_margins"], torch.tensor([0.3, 0.2]))


def test_factorized_teacher_same_family_action_margin_uses_packed_student_logits() -> None:
    catalog = _catalog()
    logits = torch.tensor([2.0, 1.7, 0.0, 4.0, 3.8, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([10, 11, 19, 13, 14, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [1, 0, 0, -1],
            [1, 0, 1, -1],
            [3, -1, -1, -1],
            [2, 0, 1, -1],
            [2, 0, 2, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    packed_view = packed_structured_legal_view(
        logits=logits,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=None,
        legal_mask=None,
        teacher_family=torch.tensor([1, 2], dtype=torch.long),
        teacher_slot=torch.tensor([0, 1], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 13], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        same_family_action_margin_coef=2.0,
        same_family_action_margin=0.5,
        packed_view=packed_view,
        factorized_family_log_probs=torch.zeros((2, 4), dtype=torch.float32),
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.55))
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.275)
    torch.testing.assert_close(context["teacher_same_family_action_margins"], torch.tensor([0.3, 0.2]))


def test_same_family_action_margin_respects_exact_action_family_filter() -> None:
    catalog = _catalog()
    logits = torch.tensor([2.0, 1.7, 0.0, 4.0, 3.8, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([10, 11, 19, 13, 14, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 6], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [1, 0, 0, -1],
            [1, 0, 1, -1],
            [3, -1, -1, -1],
            [2, 0, 1, -1],
            [2, 0, 2, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 2], dtype=torch.long),
        teacher_slot=torch.tensor([0, 1], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 13], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 100.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        same_family_action_margin_coef=2.0,
        same_family_action_margin=0.5,
        exact_action_families=("attack",),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.4))
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.2)
    torch.testing.assert_close(context["teacher_same_family_action_margins"], torch.tensor([0.3]))


def test_exact_action_family_filter_skips_public_unstable_candidate_ids() -> None:
    catalog = _catalog()
    logits = torch.tensor([1.0, 2.0, 1.7, 4.0, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([0, 10, 19, 0, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 0, -1, -1],
            [1, 0, 0, -1],
            [3, -1, -1, -1],
            [0, 0, -1, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )

    aux_loss, metrics, _context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 100.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=1.0,
        same_family_action_coef=0.0,
        exact_action_families=("attack",),
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    expected = -(logits[1] - torch.logsumexp(logits[:3], dim=0))
    torch.testing.assert_close(aux_loss, expected)
    assert metrics["teacher_action_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_accuracy"] == pytest.approx(1.0)


def test_public_nonpass_over_pass_margin_loss_targets_pass_collapse_only_when_public_teacher_prefers_nonpass() -> None:
    catalog = _catalog()
    logits = torch.tensor([1.0, 2.0, 1.7, 4.0, 1.0], dtype=torch.float32)
    packed_ids = torch.tensor([0, 10, 19, 0, 19], dtype=torch.long)
    packed_offsets = torch.tensor([0, 3, 5], dtype=torch.long)
    packed_meta = torch.tensor(
        [
            [0, 0, -1, -1],
            [1, 0, 0, -1],
            [3, -1, -1, -1],
            [0, 0, -1, -1],
            [3, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    target_logits = torch.tensor([0.0, 3.0, 1.0, 0.0, 5.0], dtype=torch.float32)

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=None,
        teacher_family=torch.tensor([1, 0], dtype=torch.long),
        teacher_slot=torch.tensor([0, 0], dtype=torch.long),
        teacher_attack_type=torch.tensor([0, -1], dtype=torch.long),
        teacher_action=torch.tensor([10, 0], dtype=torch.long),
        teacher_valid=torch.tensor([True, True]),
        loss_mask=torch.tensor([1.0, 3.0], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        public_heuristic_target_logits=target_logits,
        public_nonpass_over_pass_coef=2.0,
        public_nonpass_over_pass_margin=0.5,
        packed_ids=packed_ids,
        packed_offsets=packed_offsets,
        packed_meta=packed_meta,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.4))
    assert metrics["teacher_public_nonpass_over_pass_supported_fraction"] == pytest.approx(0.25)
    assert metrics["teacher_public_nonpass_over_pass_loss"] == pytest.approx(0.2)
    assert metrics["teacher_public_nonpass_over_pass_margin_mean"] == pytest.approx(0.3)
    assert metrics["teacher_public_nonpass_over_pass_satisfied_fraction"] == pytest.approx(0.0)
    torch.testing.assert_close(context["teacher_public_nonpass_over_pass_margins"], torch.tensor([0.3]))


def test_dense_teacher_action_margin_loss_matches_packed_branch() -> None:
    catalog = _catalog()
    logits = torch.full((1, 2, 20), -7.0, dtype=torch.float32)
    logits[0, 0, 0] = 1.0
    logits[0, 0, 10] = 2.0
    logits[0, 0, 19] = 1.7
    logits[0, 1, 0] = 4.0
    logits[0, 1, 19] = 1.0
    legal_mask = torch.zeros((1, 2, 20), dtype=torch.bool)
    legal_mask[0, 0, [0, 10, 19]] = True
    legal_mask[0, 1, [0, 19]] = True

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[1, 0]], dtype=torch.long),
        teacher_slot=torch.tensor([[0, 0]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[0, -1]], dtype=torch.long),
        teacher_action=torch.tensor([[10, 0]], dtype=torch.long),
        teacher_valid=torch.tensor([[True, True]]),
        loss_mask=torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        action_margin_coef=2.0,
        action_margin=0.5,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.1))
    assert metrics["teacher_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_action_margin_loss"] == pytest.approx(0.05)
    assert metrics["teacher_action_margin_mean"] == pytest.approx(2.325)
    assert metrics["teacher_action_margin_satisfied_fraction"] == pytest.approx(0.75)
    torch.testing.assert_close(context["teacher_action_margins"], torch.tensor([0.3, 3.0]))


def test_dense_teacher_same_family_action_margin_loss_matches_packed_branch() -> None:
    catalog = _catalog()
    logits = torch.full((1, 2, 20), -7.0, dtype=torch.float32)
    logits[0, 0, 10] = 2.0
    logits[0, 0, 11] = 1.7
    logits[0, 0, 19] = 0.0
    logits[0, 1, 13] = 4.0
    logits[0, 1, 14] = 3.8
    logits[0, 1, 19] = 1.0
    legal_mask = torch.zeros((1, 2, 20), dtype=torch.bool)
    legal_mask[0, 0, [10, 11, 19]] = True
    legal_mask[0, 1, [13, 14, 19]] = True

    aux_loss, metrics, context = compute_structured_teacher_auxiliary_metrics(
        logits=logits,
        legal_mask=legal_mask,
        teacher_family=torch.tensor([[1, 2]], dtype=torch.long),
        teacher_slot=torch.tensor([[0, 1]], dtype=torch.long),
        teacher_attack_type=torch.tensor([[0, -1]], dtype=torch.long),
        teacher_action=torch.tensor([[10, 13]], dtype=torch.long),
        teacher_valid=torch.tensor([[True, True]]),
        loss_mask=torch.tensor([[1.0, 3.0]], dtype=torch.float32),
        action_catalog=catalog,
        family_coef=0.0,
        slot_coef=0.0,
        attack_type_coef=0.0,
        action_coef=0.0,
        same_family_action_coef=0.0,
        same_family_action_margin_coef=2.0,
        same_family_action_margin=0.5,
    )

    torch.testing.assert_close(aux_loss, torch.tensor(0.55))
    assert metrics["teacher_same_family_action_margin_supported_fraction"] == pytest.approx(1.0)
    assert metrics["teacher_same_family_action_margin_loss"] == pytest.approx(0.275)
    assert metrics["teacher_same_family_action_margin_mean"] == pytest.approx(0.225)
    assert metrics["teacher_same_family_action_margin_satisfied_fraction"] == pytest.approx(0.0)
    torch.testing.assert_close(context["teacher_same_family_action_margins"], torch.tensor([0.3, 0.2]))
