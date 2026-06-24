# Model

This page explains the policy/value model and the structured legal-action head.
Use it when the question is "why is the model shaped this way?"

## High-Level Shape

The thesis model is an actor-critic:

- the policy side scores legal actions;
- the value side estimates the expected return from the current state;
- the recurrent trunk carries seat-aware hidden state across decisions.

The public import path is `weiss_rl.model`. Implementation helpers live under
`python/weiss_rl/models/`.
`python/weiss_rl/models/architecture_map.py` names the component reading order,
owner modules, and evidence each piece is responsible for.

## Default Thesis Model

The canonical B1 and main league configs use the medium64 structured model:

- GRU hidden size: `64`
- encoder MLP width: `64`
- typed feature width: `16`
- encoder kind: `structured_v2`
- structured policy contract: `factorized_v1`

These values keep the model small enough for many parallel simulator rows while
still giving the head enough structure to distinguish card zones, action
families, slots, and move arguments.

## Why A Structured Head

The simulator action space is not just a flat list of unrelated labels. Legal
actions carry family and argument structure: play, move, attack, activate,
target, slot, and card-index decisions.

A flat policy head can score the global action catalog, but it does not expose
that structure to the scorer. The structured head does three useful things:

- builds state and candidate representations from the observation contract;
- scores only simulator-provided legal candidates when possible;
- shares parameters across related action families and arguments.

That makes the head easier to supervise, easier to inspect, and less dependent
on memorizing one global action ID table.

## Forward Path

1. `TypedObservationEncoder` groups the simulator observation into header,
   player, card-zone, and tail features.
2. The recurrent core updates the hidden state for the acting seat.
3. The value head predicts a scalar value from the recurrent output.
4. The structured policy head builds candidate features for legal actions.
5. Dense, packed, or factorized scoring returns logits for the active decision.

`python/weiss_rl/models/backbone/trunk_contract.py` names the tuple shared
between trunk and head code: recurrent output, state representation,
observation context, value, and next seat-hidden state.

## Owner Files

| Concept | Files |
| --- | --- |
| Public facade and model factory | `python/weiss_rl/model.py` |
| Model architecture map | `python/weiss_rl/models/architecture_map.py` |
| Base recurrent model behavior | `python/weiss_rl/models/backbone/base.py`, `python/weiss_rl/models/backbone/policy_value_recurrent.py` |
| Dense sequence unroll helper | `python/weiss_rl/models/backbone/sequence_forward.py` |
| Opponent-context adapters | `python/weiss_rl/models/policy/opponent_context_mixin.py`, `python/weiss_rl/models/policy/opponent_context.py`, `python/weiss_rl/models/policy/opponent_context_packed.py` |
| Structured model facade | `python/weiss_rl/models/policy/policy_value_facade.py`, `python/weiss_rl/models/scoring/packed_policy_facade.py` |
| Trunk forwarding | `python/weiss_rl/models/backbone/policy_value_trunk.py`, `python/weiss_rl/models/backbone/trunk_contract.py` |
| Typed observation encoder | `python/weiss_rl/models/observations/typed_encoder.py` |
| Observation contract | `python/weiss_rl/models/observations/observation_contract.py` |
| Legal-action masking contract | `python/weiss_rl/core/action_ids.py`, `python/weiss_rl/core/masking.py`, `python/weiss_rl/core/packed_masking.py` |
| Structured legal-action head | `python/weiss_rl/models/heads/structured_head.py`, `python/weiss_rl/models/heads/structured_head_blueprint.py`, `python/weiss_rl/models/heads/structured_head_setup.py`, `python/weiss_rl/models/heads/structured_head_dimensions.py`, `python/weiss_rl/models/heads/structured_head_modules.py` |
| Structured head state and candidate features | `python/weiss_rl/models/heads/structured_head_context.py` |
| Legal scoring interface | `python/weiss_rl/models/scoring/structured_legal_scoring.py` |
| Dense candidate scoring | `python/weiss_rl/models/scoring/dense_scoring.py` |
| Packed candidate projection | `python/weiss_rl/models/scoring/packed_projection.py` |
| Packed candidate scoring | `python/weiss_rl/models/scoring/packed_scoring.py`, `python/weiss_rl/models/scoring/packed_scoring_support.py`, `python/weiss_rl/models/scoring/packed_legal_tensors.py` |
| Factorized distribution builder | `python/weiss_rl/models/scoring/factorized_scoring.py`, `python/weiss_rl/models/scoring/factorized_conditionals.py` |
| Factorized candidate/action scoring | `python/weiss_rl/models/scoring/factorized_candidate_scoring.py`, `python/weiss_rl/models/scoring/factorized_evaluation_parts.py`, `python/weiss_rl/models/scoring/factorized_facade.py`, `python/weiss_rl/models/scoring/packed_policy_stats.py` |
| Learner-side factorized evaluation | `python/weiss_rl/learners/factorized_evaluation.py`, `python/weiss_rl/learners/factorized_public_teacher.py`, `python/weiss_rl/learners/factorized_batch.py` |
| Learner-side action log-prob reductions | `python/weiss_rl/learners/action_logp.py`, `python/weiss_rl/learners/packed_action_logp.py` |
| Learner-side packed legal views | `python/weiss_rl/learners/structured_legal_view.py`, `python/weiss_rl/learners/structured_auxiliary.py` |
| Public heuristic teacher targets | `python/weiss_rl/public_heuristic/profiles.py`, `python/weiss_rl/learners/public_heuristic_profiles.py`, `python/weiss_rl/learners/impala/support/public_heuristic_support.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_scoring.py` |
| Factorized sampling | `python/weiss_rl/models/scoring/factorized_sampling.py` |
| Factorized tensor helpers | `python/weiss_rl/models/scoring/factorized_math.py` |
| Factorized diagnostics | `python/weiss_rl/models/scoring/factorized_diagnostics.py` |
| Public heuristic bias features | `python/weiss_rl/public_heuristic/profiles.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_scoring.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_slots.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_board_actions.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_family_actions.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_bias.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_bias_mixin.py`, `python/weiss_rl/models/public_heuristic/public_heuristic_primitives.py`, `python/weiss_rl/models/public_heuristic/public_heuristics.py` |

## Dense Versus Structured Mode

`PolicyValueModel` is the dense fallback. It encodes the observation, runs the
recurrent core, and applies a linear policy head over the full action catalog.

`StructuredLegalPolicyValueModel` is the thesis path. It keeps the same trunk
and value head, then replaces the flat policy head with `_StructuredLegalActionHead`.
The model still returns logits and values through the same public interface.

`python/weiss_rl/models/policy/policy_value_factory.py` owns the route choice:

| Route | Condition | Model | Required Inputs |
| --- | --- | --- | --- |
| `structured_v2` | `config.encoder_kind == "structured_v2"` | `StructuredLegalPolicyValueModel` | observation dimensions, model config, action dimensions, observation spec, simulator spec bundle |
| dense fallback | all other encoder kinds | `PolicyValueModel` | observation dimensions, model config, action dimensions |

## Legal Candidate Scoring

The structured model supports three scoring surfaces:

- dense logits over the full action catalog for compatibility;
- packed legal-candidate logits for runtime and learner efficiency;
- factorized family/argument scores for structured supervision and diagnostics.

The packed path is the most important runtime path because it avoids wasting
work on illegal actions.

## Head Blueprint

`heads/structured_head_blueprint.py` resolves the catalog-dependent objects the
head needs before scoring can happen: the action-family view, dense action
component tables, candidate feature dimensions and offsets, and factorized
lookup tables. `heads/structured_head.py` installs that blueprint into modules
and buffers, then the dense, packed, and factorized scoring files use the
installed state. `heads/structured_head_build_plan.py` names that install order
so the head can be reviewed as a checklist instead of a long constructor.
`heads/structured_head_scoring_surfaces.py` names the four use modes: dense
legal logits, packed candidate logits, factorized policy outputs, and public
heuristic bias.

## Opponent Context

Some runs enable opponent-context adapters. These adapters add small hidden,
recurrent, action-bias, or candidate-residual adjustments keyed by opponent
policy identity. They are optional and must not change the default no-context
behavior.

The context mixin keeps this optional path separate from the recurrent trunk:
`backbone/base.py` decides when context can affect a forward path, and
`policy/opponent_context_mixin.py` owns how those offsets, biases, and
residuals are installed and computed.

## What To Avoid Changing Casually

Treat these as behavior-sensitive:

- action ID ordering;
- legal-action masks and packed offsets;
- observation slices and card scalar handling;
- seat-aware hidden-state updates;
- value shape and bootstrap semantics;
- public facade imports from `weiss_rl.model`;
- state-dict compatibility for retained checkpoints.
