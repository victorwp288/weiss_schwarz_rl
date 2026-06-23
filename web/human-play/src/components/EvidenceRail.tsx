import { X } from "lucide-react";

import { compactId, cx, formatMs } from "../format";
import type { ApiHealth, SessionState } from "../types";

type EvidenceRailProps = {
  health: ApiHealth | null;
  state: SessionState | null;
  onClose: () => void;
};

export function EvidenceRail({ health, state, onClose }: EvidenceRailProps) {
  const counters = state?.model?.god_search?.counters;
  const recent = state?.model?.recent_actions ?? [];
  const lastModel = recent.length ? recent[recent.length - 1] : undefined;
  const winnerSeat = terminalNumber(state, "winner_seat");
  const terminalDecisionCount = terminalNumber(state, "decision_count");
  const terminalStatus = terminalString(state, "status");

  return (
    <>
      <div className="sheet__scrim" onClick={onClose} aria-hidden />
      <aside className="sheet" aria-label="Match log and telemetry">
        <div className="sheet__head">
          <div>
            <h2>Match log</h2>
            <p>Low-level state written as decisions are made.</p>
          </div>
          <button className="btn btn--icon" type="button" onClick={onClose} aria-label="Close log">
            <X size={16} aria-hidden />
          </button>
        </div>

        <div className="sheet__body">
          <div className="sheet__group">
            <h3>State</h3>
            <Kv label="Decision" value={String(terminalDecisionCount ?? state?.view.summary?.decision_count ?? state?.view.decision_id ?? "—")} />
            <Kv label="Phase" value={state?.view.summary?.phase ?? "—"} />
            <Kv label="Policy" value={state?.policy_id ? compactId(state.policy_id, 12) : "—"} />
            <Kv label="Legal fingerprint" value={compactId(state?.view.legal_fingerprint64, 8) || "—"} />
            <Kv label="View hash" value={compactId(state?.view.view_hash64, 8) || "—"} />
            <Kv label="Search" value={counters ? `${counters.search_decisions ?? 0} probes` : "disabled"} />
            {state?.terminal ? (
              <Kv
                label="Result"
                value={winnerSeat === null ? (terminalStatus ?? "complete") : `Winner seat ${winnerSeat}`}
              />
            ) : null}
          </div>

          <div className="sheet__group">
            <h3>Play-by-play</h3>
            {state?.history?.length ? (
              <ol className="playbyplay">
                {[...state.history].reverse().map((entry) => (
                  <li key={entry.decision_index} className={cx("ply", entry.actor_kind === "model" && "ply--model")}>
                    <span className="ply__who">{entry.actor_kind === "model" ? "Opp" : "You"}</span>
                    <span className="ply__label">{entry.label}</span>
                    {entry.phase ? <span className="ply__phase">{entry.phase}</span> : null}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="muted">Moves appear here as they happen.</p>
            )}
          </div>

          <div className="sheet__group">
            <h3>Latest model act</h3>
            {lastModel ? (
              <>
                <Kv label="Action" value={lastModel.action_label} />
                <Kv label="Took" value={formatMs(lastModel.elapsed_ms)} />
              </>
            ) : (
              <p className="muted">No model action recorded yet.</p>
            )}
          </div>

          <div className="sheet__group">
            <h3>Artifacts</h3>
            {state?.artifacts ? (
              <>
                <div className="mono">{state.artifacts.manifest}</div>
                <div className="mono">{state.artifacts.decisions}</div>
                <div className="mono">{state.artifacts.postgame_report}</div>
              </>
            ) : (
              <p className="muted">Start a match to create a transcript.</p>
            )}
          </div>

          <div className="sheet__group">
            <h3>Server</h3>
            <Kv label="API" value={health?.ok ? "ready" : "offline"} />
            {health?.weiss_sim?.version ? <Kv label="weiss-sim" value={health.weiss_sim.version} /> : null}
          </div>
        </div>
      </aside>
    </>
  );
}

function Kv({ label, value }: { label: string; value: string }) {
  return (
    <div className="kv">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function terminalNumber(state: SessionState | null, key: string): number | null {
  const value = state?.result?.[key];
  return typeof value === "number" ? value : null;
}

function terminalString(state: SessionState | null, key: string): string | null {
  const value = state?.result?.[key];
  return typeof value === "string" ? value : null;
}
