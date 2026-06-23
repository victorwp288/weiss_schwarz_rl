from __future__ import annotations

import pytest
import torch
from torch import nn
from weiss_rl.models.candidate_projection import project_candidate_sections, score_candidate_group


def _candidate_projection() -> nn.Sequential:
    projection = nn.Sequential(nn.Linear(5, 3), nn.Tanh())
    with torch.no_grad():
        linear = projection[0]
        assert isinstance(linear, nn.Linear)
        linear.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [-1.0, 0.5, 1.5, -2.0, 0.25],
                    [0.0, -1.0, 2.0, 1.0, -0.5],
                ],
            )
        )
        linear.bias.copy_(torch.tensor([0.5, -0.25, 1.0]))
    return projection


def test_project_candidate_sections_matches_actor_and_learner_paths() -> None:
    feature = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    numeric = torch.tensor([[0.5, 2.0], [1.5, 3.0]])
    projection = _candidate_projection()

    actor_projected = project_candidate_sections(
        candidate_projection=projection,
        numeric_feature_offset=2,
        feature_sections=((feature, (0, 2)),),
        numeric_sections=((numeric, (0, 1)),),
        constant_numeric_ones=(2,),
        scoring_mode="actor",
    )
    learner_projected = project_candidate_sections(
        candidate_projection=projection,
        numeric_feature_offset=2,
        feature_sections=((feature, (0, 2)),),
        numeric_sections=((numeric, (0, 1)),),
        constant_numeric_ones=(2,),
        scoring_mode="learner",
    )

    assert torch.allclose(actor_projected, learner_projected)


def test_project_candidate_sections_requires_linear_first_layer_and_inputs() -> None:
    with pytest.raises(RuntimeError, match="structured candidate projection must begin with nn.Linear"):
        project_candidate_sections(
            candidate_projection=nn.Sequential(nn.ReLU()),
            numeric_feature_offset=2,
            feature_sections=(),
            scoring_mode="learner",
        )
    with pytest.raises(ValueError, match="requires at least one feature section"):
        project_candidate_sections(
            candidate_projection=_candidate_projection(),
            numeric_feature_offset=2,
            feature_sections=(),
            scoring_mode="learner",
        )


def test_score_candidate_group_matches_actor_and_learner_paths() -> None:
    row_states = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    feature = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    numeric = torch.tensor([[0.5, 2.0], [1.5, 3.0]])
    joint_scorer = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
    with torch.no_grad():
        linear = joint_scorer[0]
        assert isinstance(linear, nn.Linear)
        linear.weight.copy_(torch.tensor([[1.0, -0.5, 0.25, 2.0, -1.0]]))
        linear.bias.copy_(torch.tensor([0.75]))
    projection = _candidate_projection()

    actor_scores = score_candidate_group(
        row_states,
        candidate_projection=projection,
        joint_scorer=joint_scorer,
        numeric_feature_offset=2,
        feature_sections=((feature, (0, 2)),),
        numeric_sections=((numeric, (0, 1)),),
        constant_numeric_ones=(2,),
        scoring_mode="actor",
    )
    learner_scores = score_candidate_group(
        row_states,
        candidate_projection=projection,
        joint_scorer=joint_scorer,
        numeric_feature_offset=2,
        feature_sections=((feature, (0, 2)),),
        numeric_sections=((numeric, (0, 1)),),
        constant_numeric_ones=(2,),
        scoring_mode="learner",
    )

    assert torch.allclose(actor_scores, learner_scores)
