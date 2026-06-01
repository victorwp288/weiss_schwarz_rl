# Thesis Shared Configs

Internal config fragments used by the public thesis configs.

These files are not launch targets. Use the public configs in `configs/thesis/`
and `configs/thesis/ablations/` for training and evaluation commands.

## Contents

- `guided_teacher/`: internal guided B1 teacher stack used by the guided seed
  and historical guided probes.
- `guided_factorized/`: internal guided-bootstrap main-league continuation
  stack used by `configs/thesis/main_league_guided_bootstrap.yaml` and related
  characterization tests.
- `main_b1only_p2/`: internal historical p2 trust-region main-league probe
  stack used by follow-up hard-negative probe characterization.
- `hardneg_core/`: internal early hard-negative main-league probe stack used by
  follow-up multiobjective and retention investigations.
- `hardneg_retention/`: internal hard-negative retention and replay-BC probe
  stack used by selected-retention and all-outcome follow-up investigations.
- `hardneg_selected/`: internal selected-checkpoint hard-negative base stack
  used by selected all-outcome and conservative follow-up investigations.
- `hardneg_selected_alloutcome/`: internal selected-checkpoint all-outcome
  repair stack used by later stratified repair investigations.
- `hardneg_selected_stratified/`: internal selected-checkpoint stratified
  all-outcome repair stack used by overlap and grouped repair investigations.
- `hardneg_selected_paired/`: internal selected-checkpoint paired replay and
  low-pressure repair stack used by outcome-contrastive investigations.
- `hardneg_selected_outcome_contrastive/`: internal selected-checkpoint
  outcome-contrastive repair stack used by interpolation continuation probes.
- `hardneg_interp_a050/`: internal a050 interpolation continuation stack used
  by later a050/a075 hard-negative follow-up probes.
- `hardneg_a050_a075_followup/`: internal a050/a075 follow-up stack used by
  later p2-live and context hard-negative probes.
- `hardneg_a050p2_live/`: internal a050p2 live hard-negative probe stack.
- `hardneg_a075_context/`: internal a075 opponent-context hard-negative probe
  stack used by later a050 context preference probes.
- `hardneg_a050_context_width128/`: internal a050 opponent-context width128
  preference probe stack.
