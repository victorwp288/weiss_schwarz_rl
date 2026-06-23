import { useCallback, useEffect, useMemo, useState } from "react";
import { LogOut, Pause, Play, RefreshCw, ScrollText, StepForward } from "lucide-react";

import {
  closeSession,
  createSession,
  getDecks,
  getHealth,
  getPolicies,
  getRuns,
  getSession,
  stepSession,
  submitAction,
} from "./api";
import { ActionInspector } from "./components/ActionInspector";
import { CardInspector } from "./components/CardInspector";
import type { InspectTarget } from "./components/CardInspector";
import { EvidenceRail } from "./components/EvidenceRail";
import { Playfield } from "./components/Playfield";
import { SetupRail } from "./components/SetupRail";
import { cx } from "./format";
import type {
  ActionFocus,
  ApiHealth,
  CreateSessionPayload,
  DeckSummary,
  LegalAction,
  PolicySummary,
  RunSummary,
  SessionState,
} from "./types";

export type UiMode = "study" | "freeplay" | "spectate";

const DEFAULT_MAIN_DECK = "preset:main_deck_5hy_yotsuba_v1";
const RUN_DIR_STORAGE_KEY = "weiss-human-play.run-dir";
const AUTO_POLICY_ID = "main_league_selected";

export function App() {
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [policies, setPolicies] = useState<PolicySummary[]>([]);
  const [runDir, setRunDir] = useState(() => localStorage.getItem(RUN_DIR_STORAGE_KEY) ?? "");
  const [policyId, setPolicyId] = useState(AUTO_POLICY_ID);
  const [humanDeck, setHumanDeck] = useState(DEFAULT_MAIN_DECK);
  const [modelDeck, setModelDeck] = useState(DEFAULT_MAIN_DECK);
  const [humanSeat, setHumanSeat] = useState(0);
  const [seed, setSeed] = useState(20260521);
  const [mode, setMode] = useState<UiMode>("study");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [state, setState] = useState<SessionState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedActionId, setSelectedActionId] = useState<number | null>(null);
  const [focus, setFocus] = useState<ActionFocus | null>(null);
  const [inspect, setInspect] = useState<InspectTarget | null>(null);
  const [notice, setNotice] = useState<{ id: number; text: string; tone: "info" | "error" } | null>(null);
  const [autoPlay, setAutoPlay] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [markedMulliganActionIds, setMarkedMulliganActionIds] = useState<ReadonlySet<number>>(() => new Set());

  // Any new decision invalidates an armed card: the legal action set changed.
  const viewHash = state?.view.view_hash64;
  useEffect(() => {
    setFocus(null);
    setInspect(null);
  }, [viewHash]);

  const nudge = useCallback((text: string, tone: "info" | "error" = "info") => {
    setNotice({ id: Date.now(), text, tone });
  }, []);

  useEffect(() => {
    if (!notice) {
      return;
    }
    const timer = window.setTimeout(() => setNotice(null), 3600);
    return () => window.clearTimeout(timer);
  }, [notice]);

  // Mid-match API errors surface as toasts (the setup screen has its own alert box).
  useEffect(() => {
    if (error && state) {
      nudge(error, "error");
    }
  }, [error, state, nudge]);

  const selectedRun = useMemo(() => runs.find((run) => run.run_dir === runDir), [runDir, runs]);
  const setupPolicies = useMemo(
    () => policies.filter((policy) => policy.policy_id !== AUTO_POLICY_ID),
    [policies],
  );

  const refreshCatalog = useCallback(async () => {
    setError(null);
    try {
      const [healthPayload, decksPayload, runsPayload] = await Promise.all([getHealth(), getDecks(), getRuns()]);
      setHealth(healthPayload);
      setDecks(decksPayload);
      setRuns(runsPayload);
      const currentRunIsLoadable = runsPayload.some((run) => run.run_dir === runDir && run.config_loadable);
      const preferredRun = runsPayload.find((run) => run.config_loadable) ?? runsPayload[0];
      if ((!runDir || !currentRunIsLoadable) && preferredRun) {
        setRunDir(preferredRun.run_dir);
      }
      if (!decksPayload.some((deck) => deck.deck_id === humanDeck) && decksPayload[0]) {
        setHumanDeck(decksPayload[0].deck_id);
      }
      if (!decksPayload.some((deck) => deck.deck_id === modelDeck) && decksPayload[0]) {
        setModelDeck(decksPayload[0].deck_id);
      }
    } catch (catalogError) {
      setError(catalogError instanceof Error ? catalogError.message : String(catalogError));
    }
  }, [humanDeck, modelDeck, runDir]);

  useEffect(() => {
    void refreshCatalog();
  }, [refreshCatalog]);

  useEffect(() => {
    if (!runDir) {
      setPolicies([]);
      return;
    }
    localStorage.setItem(RUN_DIR_STORAGE_KEY, runDir);
    let cancelled = false;
    getPolicies(runDir)
      .then((payload) => {
        if (!cancelled) {
          setPolicies(payload);
          setPolicyId((current) => reconcilePolicyId(current, payload));
        }
      })
      .catch((policyError) => {
        if (!cancelled) {
          setError(policyError instanceof Error ? policyError.message : String(policyError));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runDir]);

  useEffect(() => {
    if (state?.session_id) {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    }
  }, [state?.session_id]);

  useEffect(() => {
    const phase = state?.view.summary?.phase?.toLowerCase();
    if (!state || state.terminal || phase !== "mulligan") {
      setMarkedMulliganActionIds(new Set());
    }
  }, [state?.session_id, state?.terminal, state?.view.summary?.phase]);

  async function startSession() {
    setBusy(true);
    setError(null);
    setSelectedActionId(null);
    setMarkedMulliganActionIds(new Set());
    try {
      const next = await createSession(
        buildSessionPayload({
          runDir,
          policyId,
          humanSeat,
          seed,
          humanDeck,
          modelDeck,
          mode,
          searchEnabled,
        }),
      );
      setState(next);
    } catch (sessionError) {
      setError(sessionError instanceof Error ? sessionError.message : String(sessionError));
    } finally {
      setBusy(false);
    }
  }

  async function chooseAction(action: LegalAction) {
    if (!state || busy) {
      return;
    }
    setSelectedActionId(action.action_id);
    setBusy(true);
    setError(null);
    try {
      const next = await submitAction(state.session_id, action.action_id, state.view.view_hash64);
      setState(next);
      if (isMulliganSelect(action)) {
        setMarkedMulliganActionIds((current) => toggleSetValue(current, action.action_id));
      } else if (isMulliganConfirm(action) || next.view.summary?.phase?.toLowerCase() !== "mulligan") {
        setMarkedMulliganActionIds(new Set());
      }
      setSelectedActionId(null);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : String(actionError));
    } finally {
      setBusy(false);
    }
  }

  const stepOnce = useCallback(async () => {
    if (!state || busy || state.terminal) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setState(await stepSession(state.session_id));
    } catch (stepError) {
      setAutoPlay(false);
      setError(stepError instanceof Error ? stepError.message : String(stepError));
    } finally {
      setBusy(false);
    }
  }, [busy, state]);

  // Spectate autoplay: one decision at a time with a small beat between moves.
  useEffect(() => {
    if (!autoPlay || !state?.spectate || state.terminal || busy) {
      return;
    }
    const timer = window.setTimeout(() => {
      void stepOnce();
    }, 850);
    return () => window.clearTimeout(timer);
  }, [autoPlay, busy, state, stepOnce]);

  useEffect(() => {
    if (state?.terminal) {
      setAutoPlay(false);
    }
  }, [state?.terminal]);

  async function refreshSession() {
    if (!state) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      setState(await getSession(state.session_id));
    } catch (sessionError) {
      setError(sessionError instanceof Error ? sessionError.message : String(sessionError));
    } finally {
      setBusy(false);
    }
  }

  async function endSession() {
    if (!state) {
      return;
    }
    const sessionId = state.session_id;
    setState(null);
    setSelectedActionId(null);
    setMarkedMulliganActionIds(new Set());
    try {
      await closeSession(sessionId);
    } catch (closeError) {
      setError(closeError instanceof Error ? closeError.message : String(closeError));
    }
  }

  if (!state) {
    return (
      <div className="app app--setup">
        <SetupRail
          health={health}
          decks={decks}
          runs={runs}
          policies={setupPolicies}
          runDir={runDir}
          policyId={policyId}
          humanDeck={humanDeck}
          modelDeck={modelDeck}
          humanSeat={humanSeat}
          seed={seed}
          mode={mode}
          searchEnabled={searchEnabled}
          busy={busy}
          error={error}
          onRunDirChange={setRunDir}
          onPolicyIdChange={setPolicyId}
          onHumanDeckChange={setHumanDeck}
          onModelDeckChange={setModelDeck}
          onHumanSeatChange={setHumanSeat}
          onSeedChange={setSeed}
          onModeChange={setMode}
          onSearchEnabledChange={setSearchEnabled}
          onCreate={startSession}
          onRefreshCatalog={refreshCatalog}
        />
      </div>
    );
  }

  return (
    <div className="app app--match">
      <section className="arena-wrap">
        <header className="topbar">
          <div className="brand-mini">
            <span className="stamp stamp--sm" aria-hidden>
              勝
            </span>
            <b>Shōbu</b>
          </div>
          <span className="session-tag" title={selectedRun ? selectedRun.label : runDir}>
            · {selectedRun ? selectedRun.label : "match"} · {state.session_id.slice(0, 8)}
          </span>
          <span className="spacer" />
          <div className="hud-actions">
            {state.spectate && !state.terminal ? (
              <>
                <button className="btn" type="button" onClick={() => void stepOnce()} disabled={busy || autoPlay}>
                  <StepForward size={15} aria-hidden />
                  Step
                </button>
                <button
                  className={cx("btn", autoPlay && "btn--ghost is-active")}
                  type="button"
                  onClick={() => setAutoPlay((value) => !value)}
                  aria-pressed={autoPlay}
                >
                  {autoPlay ? <Pause size={15} aria-hidden /> : <Play size={15} aria-hidden />}
                  {autoPlay ? "Pause" : "Auto"}
                </button>
              </>
            ) : null}
            <span className={health?.ok ? "status-pill ok" : "status-pill warn"}>
              <span className="dot" /> {health?.ok ? "API ready" : "API offline"}
            </span>
            <button
              className={cx("btn btn--ghost", showDebug && "is-active")}
              type="button"
              onClick={() => setShowDebug((value) => !value)}
              aria-pressed={showDebug}
            >
              <ScrollText size={15} aria-hidden />
              Log
            </button>
            <button className="btn btn--ghost" type="button" onClick={refreshSession} disabled={busy}>
              <RefreshCw size={15} aria-hidden />
              Refresh
            </button>
            <button className="btn btn--danger" type="button" onClick={endSession}>
              <LogOut size={15} aria-hidden />
              Exit
            </button>
          </div>
        </header>

        <Playfield
          state={state}
          loading={busy}
          selectedActionId={selectedActionId}
          markedActionIds={markedMulliganActionIds}
          focus={focus}
          onFocusChange={setFocus}
          onSelectAction={chooseAction}
          onInspect={setInspect}
          onNudge={nudge}
        />
      </section>

      <ActionInspector
        state={state}
        selectedActionId={selectedActionId}
        markedActionIds={markedMulliganActionIds}
        busy={busy}
        focus={focus}
        onClearFocus={() => setFocus(null)}
        onSelectAction={chooseAction}
      />

      <CardInspector
        target={inspect}
        humanTurn={Boolean(state.human_turn)}
        spectate={Boolean(state.spectate)}
        phase={state.view.summary?.phase ?? null}
      />

      {notice ? (
        <div key={notice.id} className={cx("toast", notice.tone === "error" && "toast--error")} role="status">
          {notice.text}
        </div>
      ) : null}

      {showDebug ? <EvidenceRail health={health} state={state} onClose={() => setShowDebug(false)} /> : null}
    </div>
  );
}

function reconcilePolicyId(current: string, policies: PolicySummary[]): string {
  if (!current || current === AUTO_POLICY_ID || current === "auto" || current === "recommended") {
    return AUTO_POLICY_ID;
  }
  if (policies.some((policy) => policy.policy_id === current)) {
    return current;
  }
  return (
    policies.find((policy) => policy.selected_by_default && policy.policy_id !== AUTO_POLICY_ID)?.policy_id ??
    policies.find((policy) => policy.policy_id !== AUTO_POLICY_ID)?.policy_id ??
    AUTO_POLICY_ID
  );
}

function buildSessionPayload({
  runDir,
  policyId,
  humanSeat,
  seed,
  humanDeck,
  modelDeck,
  mode,
  searchEnabled,
}: {
  runDir: string;
  policyId: string;
  humanSeat: number;
  seed: number;
  humanDeck: string;
  modelDeck: string;
  mode: UiMode;
  searchEnabled: boolean;
}): CreateSessionPayload {
  return {
    run_dir: runDir,
    policy_id: policyId,
    human_seat: humanSeat,
    seed,
    human_deck: humanDeck,
    model_deck: modelDeck,
    mode: mode === "spectate" ? "study" : mode,
    spectate: mode === "spectate",
    model_sampling_algorithm: "model_argmax_pinned_v1",
    top_k: 5,
    search_rollout_opponent_policy_id: "B0 RandomLegal",
    god_search: searchEnabled
      ? {
          mode: "same_world_prefix_rollout",
          top_k: 4,
          rollouts_per_action: 1,
          max_rollout_decisions: 80,
          max_search_decisions_per_game: 12,
          rollout_policy: "eval",
        }
      : { mode: "disabled" },
  };
}

function isMulliganSelect(action: LegalAction): boolean {
  return action.family === "mulligan_select";
}

function isMulliganConfirm(action: LegalAction): boolean {
  return action.family === "mulligan_confirm";
}

function toggleSetValue(values: ReadonlySet<number>, value: number): ReadonlySet<number> {
  const next = new Set(values);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}
