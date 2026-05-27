import { Check, Swords } from "lucide-react";

import { cx, formatProbability } from "../format";
import type { LegalAction, RankedModelAction, SessionState } from "../types";

type ActionInspectorProps = {
  state: SessionState | null;
  selectedActionId: number | null;
  busy: boolean;
  onSelectAction: (action: LegalAction) => void;
};

export function ActionInspector({ state, selectedActionId, busy, onSelectAction }: ActionInspectorProps) {
  const legalActions = state?.view.legal_actions ?? [];
  const recentActions = state?.model?.recent_actions ?? [];
  const ranked = recentActions.length ? recentActions[recentActions.length - 1].ranked_actions : [];

  return (
    <aside className="action-rail" aria-label="Legal moves">
      <div className="rail-heading">
        <div>
          <h2>Legal Moves</h2>
          <p>
            {state
              ? state.terminal
                ? "Match complete."
                : state.human_turn
                  ? "Choose one simulator-legal action."
                  : "Waiting for the model."
              : "Start a session."}
          </p>
        </div>
        <span className="count-pill">{legalActions.length}</span>
      </div>

      <div className="legal-list">
        {legalActions.length === 0 ? (
          <div className="empty-state">
            {state?.terminal
              ? "No further actions. The match is complete."
              : "No legal actions are available in the current view."}
          </div>
        ) : (
          legalActions.map((action) => (
            <button
              key={`${action.action_id}-${action.index ?? "x"}`}
              className={cx("legal-action", selectedActionId === action.action_id && "is-selected")}
              type="button"
              disabled={!state?.human_turn || busy}
              onClick={() => onSelectAction(action)}
            >
              <span className="action-family">{action.family ?? actionType(action)}</span>
              <span className="action-label">{action.label ?? action.short_label ?? `Action ${action.action_id}`}</span>
              <span className="action-meta">
                #{action.action_id}
                {selectedActionId === action.action_id ? <Check size={13} aria-hidden /> : null}
              </span>
            </button>
          ))
        )}
      </div>

      <div className="model-panel">
        <div className="mini-heading">
          <Swords size={15} aria-hidden />
          <span>Last model preference</span>
        </div>
        {ranked.length ? <RankedActions ranked={ranked} /> : <p className="muted">Appears after the model acts.</p>}
      </div>
    </aside>
  );
}

function RankedActions({ ranked }: { ranked: RankedModelAction[] }) {
  return (
    <ol className="ranked-list">
      {ranked.slice(0, 5).map((action) => (
        <li key={action.action_id}>
          <span>{action.label}</span>
          <strong>{formatProbability(action.probability)}</strong>
        </li>
      ))}
    </ol>
  );
}

function actionType(action: LegalAction): string {
  if (action.is_attack) {
    return "attack";
  }
  if (action.is_play) {
    return "play";
  }
  if (action.is_move) {
    return "move";
  }
  if (action.is_pass) {
    return "pass";
  }
  return "action";
}
