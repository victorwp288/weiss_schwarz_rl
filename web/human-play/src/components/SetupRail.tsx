import { Play, RefreshCw, Search } from "lucide-react";

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
  mode: "study" | "freeplay" | "spectate";
  searchEnabled: boolean;
  busy: boolean;
  error: string | null;
  onRunDirChange: (value: string) => void;
  onPolicyIdChange: (value: string) => void;
  onHumanDeckChange: (value: string) => void;
  onModelDeckChange: (value: string) => void;
  onHumanSeatChange: (value: number) => void;
  onSeedChange: (value: number) => void;
  onModeChange: (value: "study" | "freeplay" | "spectate") => void;
  onSearchEnabledChange: (value: boolean) => void;
  onCreate: () => void;
  onRefreshCatalog: () => void;
};

export function SetupRail(props: SetupRailProps) {
  const {
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
  } = props;

  const apiReady = Boolean(health?.ok);
  const displayPolicies = policies.filter((policy) => policy.policy_id !== "main_league_selected");

  return (
    <form
      className="setup"
      aria-label="Match setup"
      onSubmit={(event) => {
        event.preventDefault();
        if (!busy && runDir) {
          onCreate();
        }
      }}
    >
      <div className="setup__hero">
        <span className="stamp" aria-hidden>
          勝
        </span>
        <div>
          <h1 className="setup__title">
            Shōbu<span className="tilde">。</span>
          </h1>
          <p className="setup__sub">Sit down at the table — Weiss Schwarz against your trained policy.</p>
        </div>
      </div>

      {apiReady ? (
        <span className="status-pill ok">
          <span className="dot" /> Simulator {health?.weiss_sim?.version ?? "ready"}
        </span>
      ) : (
        <div className="alert" role="alert">
          <strong>Simulator API unavailable.</strong>
          <span>
            Start the API with the project virtualenv (so <code>weiss_sim</code> is importable):
          </span>
          <code>.venv\Scripts\python -m weiss_rl.human_play.web_server</code>
          <span>The dev server proxies <code>/api</code> to it on port 8765. Then refresh the catalog.</span>
        </div>
      )}

      <section className="setup__section">
        <div className="row-between">
          <h2>Model</h2>
          <button className="btn btn--icon" type="button" onClick={onRefreshCatalog} aria-label="Refresh catalog">
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
          <input value={runDir} onChange={(event) => onRunDirChange(event.target.value)} placeholder="C:\\path\\to\\run" />
        </label>
        <label className="field">
          <span>Policy</span>
          <select value={policyId} onChange={(event) => onPolicyIdChange(event.target.value)}>
            <option value="main_league_selected">Auto-select strongest main model</option>
            {displayPolicies.map((policy) => (
              <option key={policy.policy_id} value={policy.policy_id}>
                {policy.label}
              </option>
            ))}
          </select>
        </label>
      </section>

      <section className="setup__section">
        <h2>Seats &amp; decks</h2>
        <label className="field">
          <span>Your seat</span>
          <div className="seg" role="group" aria-label="Human seat">
            <button type="button" className={humanSeat === 0 ? "is-on" : ""} onClick={() => onHumanSeatChange(0)}>
              Seat 0
            </button>
            <button type="button" className={humanSeat === 1 ? "is-on" : ""} onClick={() => onHumanSeatChange(1)}>
              Seat 1
            </button>
          </div>
        </label>
        <DeckSelect label="Your deck" decks={decks} value={humanDeck} onChange={onHumanDeckChange} />
        <DeckSelect label="Opponent deck" decks={decks} value={modelDeck} onChange={onModelDeckChange} />
      </section>

      <section className="setup__section">
        <h2>Match options</h2>
        <div className="grid-2">
          <label className="field">
            <span>Seed</span>
            <input type="number" value={seed} onChange={(event) => onSeedChange(Number(event.target.value))} />
          </label>
          <label className="field">
            <span>Mode</span>
            <div className="seg" role="group" aria-label="Mode">
              <button type="button" className={mode === "study" ? "is-on" : ""} onClick={() => onModeChange("study")}>
                Study
              </button>
              <button type="button" className={mode === "freeplay" ? "is-on" : ""} onClick={() => onModeChange("freeplay")}>
                Freeplay
              </button>
              <button
                type="button"
                className={mode === "spectate" ? "is-on" : ""}
                onClick={() => onModeChange("spectate")}
                title="The model plays both seats; you watch."
              >
                Spectate
              </button>
            </div>
          </label>
        </div>
        <button
          className={searchEnabled ? "toggle is-on" : "toggle"}
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
        <div className="alert" role="alert">
          <strong>Could not start</strong>
          <span>{error}</span>
        </div>
      ) : null}

      <button className="start-btn" type="submit" disabled={busy || !runDir}>
        {busy ? <RefreshCw size={18} aria-hidden /> : <Play size={18} aria-hidden />}
        {busy ? "Dealing…" : "Start match"}
      </button>
    </form>
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
