from __future__ import annotations

import pytest
import torch
from weiss_rl.learners.paired_outcome_preference.edge_pairs import edge_mean_preference_pair_components
from weiss_rl.learners.paired_outcome_preference.pairs import preference_pair_components
from weiss_rl.learners.paired_outcome_preference.span_pairs import span_preference_pair_components


def test_preference_pair_components_preserves_edge_mean_alignment_contract() -> None:
    current = torch.tensor([[-2.0, -2.0], [-1.0, -3.0]], dtype=torch.float32)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[5, 5], [5, 5]], dtype=torch.long)
    roles = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    valid = torch.ones_like(current, dtype=torch.bool)

    components = preference_pair_components(
        current_action_logp=current,
        reference_action_logp=reference,
        current=current.reshape(-1),
        reference=reference.reshape(-1),
        pair_ids=pair_ids.reshape(-1),
        roles=roles.reshape(-1),
        valid=valid.reshape(-1),
        group_ids=None,
        pair_weight_rows=None,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_group_ids=None,
        preference_pair_weights=None,
        unique_pair_ids=torch.tensor([5], dtype=torch.long),
        aggregation="edge_mean",
        beta=1.0,
        dtype=current.dtype,
    )

    assert [margin.item() for margin in components.margins] == pytest.approx([0.0, 2.0])
    assert len(components.pair_losses) == 2
    assert [weight.item() for weight in components.pair_weights] == pytest.approx([1.0, 1.0])
    assert components.incomplete_pair_count == 0


def test_preference_pair_dispatcher_uses_canonical_span_and_edge_builders() -> None:
    current = torch.tensor([[-2.0, -2.0], [-1.0, -3.0]], dtype=torch.float32)
    reference = torch.zeros_like(current)
    pair_ids = torch.tensor([[5, 5], [5, 5]], dtype=torch.long)
    roles = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
    valid = torch.ones_like(current, dtype=torch.bool)
    unique_pair_ids = torch.tensor([5], dtype=torch.long)

    span = span_preference_pair_components(
        current=current.reshape(-1),
        reference=reference.reshape(-1),
        pair_ids=pair_ids.reshape(-1),
        roles=roles.reshape(-1),
        valid=valid.reshape(-1),
        group_ids=None,
        pair_weight_rows=None,
        unique_pair_ids=unique_pair_ids,
        aggregation="mean",
        beta=1.0,
        dtype=current.dtype,
    )
    span_dispatched = preference_pair_components(
        current_action_logp=current,
        reference_action_logp=reference,
        current=current.reshape(-1),
        reference=reference.reshape(-1),
        pair_ids=pair_ids.reshape(-1),
        roles=roles.reshape(-1),
        valid=valid.reshape(-1),
        group_ids=None,
        pair_weight_rows=None,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_group_ids=None,
        preference_pair_weights=None,
        unique_pair_ids=unique_pair_ids,
        aggregation="mean",
        beta=1.0,
        dtype=current.dtype,
    )
    edge = edge_mean_preference_pair_components(
        current_action_logp=current,
        reference_action_logp=reference,
        pair_ids=pair_ids,
        roles=roles,
        valid=valid,
        group_ids=None,
        pair_weight_rows=None,
        unique_pair_ids=unique_pair_ids,
        beta=1.0,
        dtype=current.dtype,
    )
    edge_dispatched = preference_pair_components(
        current_action_logp=current,
        reference_action_logp=reference,
        current=current.reshape(-1),
        reference=reference.reshape(-1),
        pair_ids=pair_ids.reshape(-1),
        roles=roles.reshape(-1),
        valid=valid.reshape(-1),
        group_ids=None,
        pair_weight_rows=None,
        preference_pair_ids=pair_ids,
        preference_role=roles,
        preference_group_ids=None,
        preference_pair_weights=None,
        unique_pair_ids=unique_pair_ids,
        aggregation="edge_mean",
        beta=1.0,
        dtype=current.dtype,
    )

    assert [margin.item() for margin in span_dispatched.margins] == pytest.approx(
        [margin.item() for margin in span.margins]
    )
    assert [margin.item() for margin in edge_dispatched.margins] == pytest.approx(
        [margin.item() for margin in edge.margins]
    )
