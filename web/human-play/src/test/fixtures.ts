import type { DeckSummary, SessionState } from "../types";

export const sampleDecks: DeckSummary[] = [
  {
    deck_id: "preset:main_deck_5hy_yotsuba_v1",
    preset_name: "main_deck_5hy_yotsuba_v1",
    label: "Yotsuba thesis deck",
    role: "primary thesis deck",
    card_count: 50,
    unique_card_count: 28,
    sample_cards: ["Yotsuba Nakano", "Choice Climax"],
  },
];

export const sampleSession: SessionState = {
  session_id: "abc123",
  mode: "study",
  human_seat: 0,
  model_seat: 1,
  policy_id: "policy_000002",
  human_turn: true,
  terminal: false,
  view: {
    decision_id: 9,
    view_hash64: "viewhash",
    legal_fingerprint64: "legalhash",
    legal_action_ids: [11, 22],
    legal_actions: [
      {
        action_id: 11,
        label: "Play Yotsuba from hand to center stage",
        family: "play",
        is_play: true,
      },
      {
        action_id: 22,
        label: "Pass",
        family: "pass",
        is_pass: true,
      },
    ],
    summary: {
      actor_seat: 0,
      decision_count: 9,
      decision_kind: "main",
      phase: "Main",
      turn_player: 0,
    },
    players: [
      {
        seat: 0,
        relative: "self",
        counts: { clock: 2, stock: 5, waiting_room: 8, deck: 31, level: 1, memory: 0, hand: 2 },
        zones: {
          hand: [
            { id: "c1", name: "Yotsuba Nakano", card_no: "5HY/W90-001", level: 1, power: 4500 },
            { id: "c2", name: "Choice Climax", card_no: "5HY/W90-CC" },
          ],
        },
        stage: [{ seat: 0, lane: 2, row: "center", card: { id: "c3", name: "Front row Yotsuba" } }],
      },
      {
        seat: 1,
        relative: "opponent",
        counts: { clock: 1, stock: 4, waiting_room: 6, deck: 34, level: 0, memory: 0, hand: 7 },
        stage: [{ seat: 1, lane: 2, row: "center", card: { id: "m1", name: "Opponent public card" } }],
      },
    ],
  },
  model: {
    recent_actions: [
      {
        decision_index: 8,
        actor_seat: 1,
        action_id: 44,
        action_label: "Attack front",
        elapsed_ms: 15,
        ranked_actions: [
          { action_id: 44, label: "Attack front", family: "attack", probability: 0.72 },
          { action_id: 45, label: "Pass attack", family: "pass", probability: 0.28 },
        ],
      },
    ],
    god_search: null,
  },
  artifacts: {
    session_dir: "runs/example/human_play/abc123",
    manifest: "runs/example/human_play/abc123/manifest.json",
    decisions: "runs/example/human_play/abc123/decisions.jsonl",
    postgame_report: "runs/example/human_play/abc123/postgame_report.md",
  },
};
