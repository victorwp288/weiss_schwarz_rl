import { Archive, BrainCircuit, Clock3, FileText, ShieldCheck, Trophy } from "lucide-react";
import type { ReactNode } from "react";

import { compactId, formatMs } from "../format";
import type { ApiHealth, SessionState } from "../types";

type EvidenceRailProps = {
  health: ApiHealth | null;
  state: SessionState | null;
  onRefresh: () => void;
  onClose: () => void;
};

export function EvidenceRail({ health, state, onRefresh, onClose }: EvidenceRailProps) {
  const counters = state?.model?.god_search?.counters;
  const recent = state?.model?.recent_actions ?? [];
  const lastModel = recent.length ? recent[recent.length - 1] : undefined;
  const winnerSeat = terminalNumberResult(state, "winner_seat");
  const terminalDecisionCount = terminalNumberResult(state, "decision_count");
  const terminalStatus = terminalStringResult(state, "status");

  return (
    <aside className="evidence-rail" aria-label="Evidence and session artifacts">
      <div className="rail-heading">
        <div>
          <h2>Evidence</h2>
          <p>Session artifacts are written as decisions are made.</p>
        </div>
        <ShieldCheck size={18} aria-hidden />
      </div>

      <div className="evidence-stack">
        <MetricLine
          icon={<Clock3 size={15} />}
          label="Decision"
          value={String(terminalDecisionCount ?? state?.view.summary?.decision_count ?? state?.view.decision_id ?? "--")}
        />
        <MetricLine
          icon={<BrainCircuit size={15} />}
          label="Policy"
          value={state?.policy_id ? compactId(state.policy_id, 12) : "--"}
        />
        {state?.terminal ? (
          <MetricLine
            icon={<Trophy size={15} />}
            label="Result"
            value={winnerSeat === null ? (terminalStatus ?? "complete") : `Winner seat ${winnerSeat}`}
          />
        ) : null}
        <MetricLine
          icon={<Archive size={15} />}
          label="Search"
          value={counters ? `${counters.search_decisions ?? 0} probes` : "disabled"}
        />
      </div>

      <div className="artifact-panel">
        <div className="mini-heading">
          <FileText size={15} aria-hidden />
          <span>Artifacts</span>
        </div>
        {state?.artifacts ? (
          <dl className="artifact-list">
            <div>
              <dt>Manifest</dt>
              <dd>{state.artifacts.manifest}</dd>
            </div>
            <div>
              <dt>Decisions</dt>
              <dd>{state.artifacts.decisions}</dd>
            </div>
            <div>
              <dt>Report</dt>
              <dd>{state.artifacts.postgame_report}</dd>
            </div>
          </dl>
        ) : (
          <p className="muted">Start a match to create a transcript.</p>
        )}
      </div>

      <div className="artifact-panel">
        <div className="mini-heading">
          <BrainCircuit size={15} aria-hidden />
          <span>Latest model act</span>
        </div>
        {lastModel ? (
          <div className="model-decision">
            <strong>{lastModel.action_label}</strong>
            <span>
              Action #{lastModel.action_id} in {formatMs(lastModel.elapsed_ms)}
            </span>
          </div>
        ) : (
          <p className="muted">No model action recorded yet.</p>
        )}
      </div>

      <div className="rail-actions">
        <button type="button" onClick={onRefresh} disabled={!state}>
          Refresh
        </button>
        <button type="button" onClick={onClose} disabled={!state}>
          Close
        </button>
      </div>

      <footer className="server-footnote">
        API {health?.ok ? "ready" : "offline"}
        {health?.weiss_sim?.file ? <span>{compactId(health.weiss_sim.file, 22)}</span> : null}
      </footer>
    </aside>
  );
}

function MetricLine({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="metric-line">
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function terminalNumberResult(state: SessionState | null, key: string): number | null {
  const value = state?.result?.[key];
  return typeof value === "number" ? value : null;
}

function terminalStringResult(state: SessionState | null, key: string): string | null {
  const value = state?.result?.[key];
  return typeof value === "string" ? value : null;
}
