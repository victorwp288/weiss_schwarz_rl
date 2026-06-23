export type ApiHealth = {
  ok: boolean;
  weiss_sim?: {
    available: boolean;
    version?: string | null;
    human_decision_view: boolean;
    file?: string | null;
  };
};

export type DeckSummary = {
  deck_id: string;
  preset_name: string;
  label: string;
  role: string;
  card_count: number;
  unique_card_count: number;
  sample_cards: string[];
  source?: string | null;
  min_rules_profile?: string | null;
};

export type RunSummary = {
  run_dir: string;
  name: string;
  label: string;
  modified_unix: number;
  policy_count: number;
  default_policy_id: string;
  has_config: boolean;
  has_registry: boolean;
  config_loadable: boolean;
  load_error?: string | null;
};

export type PolicySummary = {
  policy_id: string;
  label: string;
  kind: "alias" | "baseline" | "heuristic" | "snapshot" | string;
  update?: number | null;
  path?: string | null;
  selected_by_default: boolean;
};

export type CardRef = {
  card?: CardView;
  card_ref?: string;
  ref_id?: string;
  index?: number;
  owner_seat?: number;
  relative_owner?: string;
  visibility?: string;
  zone?: string;
};

export type LegalAction = {
  action_id: number;
  index?: number;
  label?: string;
  short_label?: string;
  description?: string;
  family?: string | null;
  params?: Record<string, unknown>;
  source_refs?: CardRef[];
  target_refs?: CardRef[];
  is_attack?: boolean;
  is_move?: boolean;
  is_pass?: boolean;
  is_play?: boolean;
};

export type PlayerZone = {
  count?: number;
  cards?: CardView[];
  hidden?: boolean;
  redacted?: boolean;
};

export type CardColor = "yellow" | "green" | "red" | "blue" | string;

export type CardView = {
  id?: string;
  card_id?: string | number;
  card_no?: string;
  name?: string;
  label?: string;
  title?: string;
  card?: CardView;
  card_ref?: string;
  hidden?: boolean;
  redacted?: boolean;
  level?: number | string;
  cost?: number | string;
  power?: number | string;
  soul?: number | string;
  color?: CardColor;
  card_type?: string;
  triggers?: string[];
  traits?: Array<number | string>;
};

export type PlayerView = {
  seat: number;
  relative?: "self" | "opponent" | string;
  counts?: Record<string, number>;
  zones?: Record<string, PlayerZone | CardView[]>;
  stage?: StageSlot[];
};

export type StageSlot = {
  seat?: number;
  owner_seat?: number;
  relative_owner?: string;
  row?: string;
  lane?: number;
  slot?: number;
  slot_ref?: string;
  card_ref?: string;
  label?: string;
  card?: CardView | null;
  cards?: CardView[];
  occupied?: boolean;
  empty?: boolean;
  rested?: boolean;
  has_attacked?: boolean;
  cannot_attack?: boolean;
  marker_count?: number;
};

export type HumanDecisionView = {
  schema_version?: string;
  decision_id?: number;
  episode_index?: number;
  episode_seed?: number;
  view_hash64?: string;
  legal_fingerprint64?: string;
  legal_action_ids: number[];
  legal_actions: LegalAction[];
  players?: PlayerView[];
  stage_layout?: { slots?: StageSlot[] } | StageSlot[];
  public_event_log?: unknown[];
  summary?: {
    actor_seat?: number;
    decision_count?: number;
    decision_kind?: string;
    phase?: string;
    terminal?: boolean;
    tick_count?: number;
    turn_count?: number;
    turn_number?: number;
    turn_player?: number;
    viewer_seat?: number;
    players?: unknown;
  };
};

/** Reference text + engine-implementation flags for one card. */
export type CardInfo = {
  card_no: string;
  name?: string | null;
  text?: string | null;
  rarity?: string | null;
  traits?: string[] | null;
  expansion?: string | null;
  approx_ok?: boolean;
  strict_ok?: boolean;
};

export type PublicHistoryEntry = {
  decision_index: number;
  actor_seat: number;
  actor_kind: "human" | "model" | string;
  label: string;
  family?: string | null;
  phase?: string | null;
  elapsed_ms?: number | null;
};

/** A card the user has "armed": the subset of legal actions it can perform. */
export type ActionFocus = {
  title: string;
  cardRef?: string | null;
  handIndex?: number | null;
  slotRef?: string | null;
  actions: LegalAction[];
};

export type RankedModelAction = {
  action_id: number;
  label: string;
  family?: string | null;
  probability?: number | null;
  logit?: number | null;
};

export type RecentModelAction = {
  decision_index: number;
  actor_seat: number;
  action_id: number;
  action_label: string;
  elapsed_ms?: number | null;
  ranked_actions: RankedModelAction[];
};

export type SessionState = {
  session_id: string;
  mode: "study" | "freeplay" | string;
  human_seat: number;
  model_seat: number;
  policy_id: string;
  human_turn: boolean;
  spectate?: boolean;
  terminal: boolean;
  view: HumanDecisionView;
  history?: PublicHistoryEntry[];
  model?: {
    recent_actions?: RecentModelAction[];
    god_search?: GodSearchDiagnostics | null;
  };
  artifacts?: {
    session_dir: string;
    manifest: string;
    decisions: string;
    postgame_report: string;
  };
  result?: Record<string, unknown>;
};

export type GodSearchDiagnostics = {
  kind: string;
  config: Record<string, unknown>;
  counters?: Record<string, number>;
  changed_fraction?: number | null;
  traces?: unknown[];
};

export type CreateSessionPayload = {
  run_dir: string;
  policy_id: string;
  human_seat: number;
  seed: number;
  human_deck: string;
  model_deck: string;
  mode: "study" | "freeplay";
  spectate?: boolean;
  model_sampling_algorithm: string;
  top_k: number;
  search_rollout_opponent_policy_id: string;
  god_search: {
    mode: "disabled" | "same_world_prefix_rollout";
    top_k?: number;
    rollouts_per_action?: number;
    max_rollout_decisions?: number;
    max_search_decisions_per_game?: number;
    rollout_policy?: "eval" | "argmax" | "sample";
  };
};
