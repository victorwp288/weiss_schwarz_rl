import { AlertTriangle, Gamepad2, Play, RefreshCw, Search, XCircle } from "lucide-react";

import type { ApiHealth, DeckSummary, PolicySummary, RunSummary } from "../types";

type SetupRailProps = {
  health: ApiHealth | null;
  decks: DeckSummary[];
  runs: RunSummary[];
  policies: PolicySummary[];
  runDir: string;
  policyId: string;
  humanDeck: string;
  modelDeck: string;
  humanSeat: number;
  seed: number;
  mode: "study" | "freeplay";
  searchEnabled: boolean;
  busy: boolean;
  error: string | null;
  onRunDirChange: (value: string) => void;
  onPolicyIdChange: (value: string) => void;
  onHumanDeckChange: (value: string) => void;
  onModelDeckChange: (value: string) => void;
  onHumanSeatChange: (value: number) => void;
  onSeedChange: (value: number) => void;
  onModeChange: (value: "study" | "freeplay") => void;
  onSearchEnabledChange: (value: boolean) => void;
  onCreate: () => void;
  onRefreshCatalog: () => void;
};

export function SetupRail({
  health,
  decks,
  runs,
  policies,
  runDir,
  policyId,
  humanDeck,
  modelDeck,
  humanSeat,
  seed,
  mode,
  searchEnabled,
  busy,
  error,
  onRunDirChange,
  onPolicyIdChange,
  onHumanDeckChange,
  onModelDeckChange,
  onHumanSeatChange,
  onSeedChange,
  onModeChange,
  onSearchEnabledChange,
  onCreate,
  onRefreshCatalog,
}: SetupRailProps) {
  return (
    <aside className="setup-rail" aria-label="Match setup">
      <div className="brand-lockup">
        <div className="brand-mark">
          <Gamepad2 size={18} aria-hidden />
        </div>
        <div>
          <h1>Weiss Human Play</h1>
          <p>Study match against a trained policy</p>
        </div>
      </div>

      <div className={health?.ok ? "system-chip is-ok" : "system-chip is-warn"}>
        {health?.ok ? <span className="status-dot" /> : <AlertTriangle size={14} aria-hidden />}
        <span>{health?.ok ? `Simulator ${health.weiss_sim?.version ?? "ready"}` : "Simulator API unavailable"}</span>
      </div>

      <section className="rail-section">
        <div className="section-title-row">
          <h2>Model</h2>
          <button className="icon-button" type="button" onClick={onRefreshCatalog} aria-label="Refresh catalog">
            <RefreshCw size={15} aria-hidden />
          </button>
        </div>
        <label className="field">
          <span>Run</span>
          <select value={runDir} onChange={(event) => onRunDirChange(event.target.value)}>
            <option value="">Select a run</option>
            {runs.map((run) => (
              <option key={run.run_dir} value={run.run_dir}>
                {run.label} ({run.policy_count}
                {run.config_loadable ? "" : ", config issue"})
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Run path</span>
          <input
            value={runDir}
            onChange={(event) => onRunDirChange(event.target.value)}
            placeholder="C:\\path\\to\\run"
          />
        </label>
        <label className="field">
          <span>Policy</span>
          <select value={policyId} onChange={(event) => onPolicyIdChange(event.target.value)}>
            <option value="main_league_selected">Auto-select strongest main model</option>
            {policies.map((policy) => (
              <option key={policy.policy_id} value={policy.policy_id}>
                {policy.label}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="rail-section">
        <h2>Seats and Decks</h2>
        <div className="segmented" role="group" aria-label="Human seat">
          <button
            type="button"
            className={humanSeat === 0 ? "is-selected" : ""}
            onClick={() => onHumanSeatChange(0)}
          >
            Seat 0
          </button>
          <button
            type="button"
            className={humanSeat === 1 ? "is-selected" : ""}
            onClick={() => onHumanSeatChange(1)}
          >
            Seat 1
          </button>
        </div>
        <DeckSelect
          label="Human deck"
          decks={decks}
          value={humanDeck}
          onChange={onHumanDeckChange}
        />
        <DeckSelect
          label="Model deck"
          decks={decks}
          value={modelDeck}
          onChange={onModelDeckChange}
        />
      </section>

      <section className="rail-section">
        <h2>Study Options</h2>
        <label className="field">
          <span>Seed</span>
          <input
            type="number"
            value={seed}
            onChange={(event) => onSeedChange(Number(event.target.value))}
          />
        </label>
        <div className="segmented" role="group" aria-label="Mode">
          <button type="button" className={mode === "study" ? "is-selected" : ""} onClick={() => onModeChange("study")}>
            Study
          </button>
          <button
            type="button"
            className={mode === "freeplay" ? "is-selected" : ""}
            onClick={() => onModeChange("freeplay")}
          >
            Freeplay
          </button>
        </div>
        <button
          className={searchEnabled ? "toggle-row is-selected" : "toggle-row"}
          type="button"
          onClick={() => onSearchEnabledChange(!searchEnabled)}
          aria-pressed={searchEnabled}
        >
          <Search size={16} aria-hidden />
          <span>Root search proxy</span>
          <strong>{searchEnabled ? "On" : "Off"}</strong>
        </button>
      </section>

      {error ? (
        <div className="error-banner" role="alert">
          <XCircle size={15} aria-hidden />
          <span>{error}</span>
        </div>
      ) : null}

      <button className="primary-action" type="button" onClick={onCreate} disabled={busy || !runDir}>
        <Play size={17} aria-hidden />
        {busy ? "Starting..." : "Start match"}
      </button>
    </aside>
  );
}

function DeckSelect({
  label,
  decks,
  value,
  onChange,
}: {
  label: string;
  decks: DeckSummary[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {decks.map((deck) => (
          <option key={deck.deck_id} value={deck.deck_id}>
            {deck.label} - {deck.role}
          </option>
        ))}
      </select>
    </label>
  );
}
