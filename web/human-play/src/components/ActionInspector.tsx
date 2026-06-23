import { ArrowRight, ChevronRight, MoveRight, Play, Shuffle, SkipForward, Sparkles, Swords, X } from "lucide-react";
import type { ReactNode } from "react";
import { Fragment } from "react";

import { cx, formatProbability, friendlyActionLabel, unwrapCard } from "../format";
import type { ActionFocus, CardRef, LegalAction, RankedModelAction, SessionState } from "../types";
import { CardFace } from "./Card";

type ActionInspectorProps = {
  state: SessionState | null;
  selectedActionId: number | null;
  markedActionIds?: ReadonlySet<number>;
  busy: boolean;
  focus?: ActionFocus | null;
  onClearFocus?: () => void;
  onSelectAction: (action: LegalAction) => void;
};

const VERB: Record<string, string> = {
  play: "Play",
  attack: "Attack",
  move: "Move",
  pass: "Pass",
  mulligan_select: "Mulligan",
  mulligan_confirm: "Mulligan",
  clock_phase: "Clock",
  climax: "Climax",
};

/** Display order: aggressive options first, pass always last. */
const FAMILY_ORDER = ["attack", "play", "move", "mulligan", "action", "pass"];
const FAMILY_TITLE: Record<string, string> = {
  attack: "Attack",
  play: "Play",
  move: "Move",
  mulligan: "Mulligan",
  action: "Other",
  pass: "Pass",
};

export function ActionInspector({
  state,
  selectedActionId,
  markedActionIds = new Set(),
  busy,
  focus = null,
  onClearFocus = () => {},
  onSelectAction,
}: ActionInspectorProps) {
  const legalActions = state?.view.legal_actions ?? [];
  const recent = state?.model?.recent_actions ?? [];
  // Prefer the latest model decision that was an actual choice over forced passes.
  const rankedSource =
    [...recent].reverse().find((entry) => entry.ranked_actions.length > 1) ?? recent[recent.length - 1];
  const ranked = rankedSource?.ranked_actions ?? [];
  const phase = state?.view.summary?.phase;

  const pool = focus ? focus.actions : legalActions;
  const sorted = [...pool].sort((a, b) => FAMILY_ORDER.indexOf(familyOf(a)) - FAMILY_ORDER.indexOf(familyOf(b)));
  const showGroups = !focus && sorted.length > 4 && new Set(sorted.map(familyOf)).size > 1;

  return (
    <aside className="actions" aria-label="Legal moves">
      <div className="actions__head">
        <div>
          <h2>Your moves</h2>
          <p>
            {state
              ? state.terminal
                ? "Match complete."
                : state.spectate
                  ? "Spectating — the model plays both seats."
                  : state.human_turn
                    ? "Choose one simulator-legal action."
                    : "Waiting for the model."
              : "Start a session."}
            {phase && state && !state.terminal ? ` · ${phase}` : ""}
          </p>
        </div>
        <span className="count-bubble">{legalActions.length}</span>
      </div>

      {focus ? (
        <div className="focus-banner" role="status">
          <span className="focus-banner__title">{focus.title}</span>
          <button type="button" className="focus-banner__clear" onClick={onClearFocus} aria-label="Show all moves">
            <X size={13} aria-hidden /> All moves
          </button>
        </div>
      ) : null}

      <div className="action-list">
        {sorted.length === 0 ? (
          <div className="empty-state">
            {state?.terminal
              ? "No further actions. The match is complete."
              : "No legal actions are available in the current view."}
          </div>
        ) : (
          sorted.map((action, index) => {
            const fam = familyOf(action);
            const selected = selectedActionId === action.action_id || markedActionIds.has(action.action_id);
            const card = refCard(action);
            const newGroup = showGroups && (index === 0 || familyOf(sorted[index - 1]) !== fam);
            const label = friendlyActionLabel(action);
            return (
              <Fragment key={`${action.action_id}-${action.index ?? "x"}`}>
                {newGroup ? <span className="group-label">{FAMILY_TITLE[fam] ?? fam}</span> : null}
                <button
                  className={cx("action", fam === "pass" && "action--pass", selected && "selected")}
                  type="button"
                  disabled={!state?.human_turn || busy}
                  onClick={() => onSelectAction(action)}
                  aria-label={label}
                  aria-pressed={selected}
                  title={`action #${action.action_id}${action.family ? ` · ${action.family}` : ""}`}
                >
                  {card && unwrapCard(card)?.name ? (
                    <span className="action__thumb" aria-hidden>
                      <CardFace card={card} size="xs" />
                    </span>
                  ) : (
                    <span className="action__verb-ico">{verbIcon(fam)}</span>
                  )}
                  <span className="action__text">
                    <span className="action__name">{label}</span>
                    <span className="action__verb">{VERB[action.family ?? fam] ?? fam}</span>
                  </span>
                  {selected ? (
                    <ChevronRight size={16} className="action__go" aria-hidden />
                  ) : (
                    <ArrowRight size={14} className="action__go" aria-hidden style={{ opacity: 0.4 }} />
                  )}
                </button>
              </Fragment>
            );
          })
        )}
      </div>

      <div className="pref">
        <div className="pref__head">
          <Swords size={15} aria-hidden />
          <span>Last model preference</span>
        </div>
        {ranked.length ? <Ranked ranked={ranked} /> : <p className="muted">Appears after the model acts.</p>}
      </div>
    </aside>
  );
}

function Ranked({ ranked }: { ranked: RankedModelAction[] }) {
  return (
    <ol className="pref__list">
      {ranked.slice(0, 4).map((action) => {
        const pct = Math.max(0, Math.min(1, action.probability ?? 0)) * 100;
        return (
          <li key={action.action_id} className="pref__row">
            <span className="pref__top">
              <span>{action.label}</span>
              <strong>{formatProbability(action.probability)}</strong>
            </span>
            <span className="pref__bar">
              <i style={{ width: `${pct}%` }} />
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function refCard(action: LegalAction): CardRef | null {
  const refs = [...(action.source_refs ?? []), ...(action.target_refs ?? [])];
  return refs.find((ref) => ref.card && unwrapCard(ref.card)?.name) ?? null;
}

function verbIcon(fam: string): ReactNode {
  switch (fam) {
    case "attack":
      return <Swords size={15} aria-hidden />;
    case "play":
      return <Play size={15} aria-hidden />;
    case "move":
      return <MoveRight size={15} aria-hidden />;
    case "pass":
      return <SkipForward size={15} aria-hidden />;
    case "mulligan":
      return <Shuffle size={15} aria-hidden />;
    default:
      return <Sparkles size={15} aria-hidden />;
  }
}

function familyOf(action: LegalAction): string {
  if (action.is_attack) return "attack";
  if (action.is_play) return "play";
  if (action.is_move) return "move";
  if (action.is_pass) return "pass";
  if (typeof action.family === "string" && action.family.startsWith("mulligan")) return "mulligan";
  return "action";
}
