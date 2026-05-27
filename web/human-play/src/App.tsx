import { useCallback, useEffect, useMemo, useState } from "react";

import {
  closeSession,
  createSession,
  getDecks,
  getHealth,
  getPolicies,
  getRuns,
  getSession,
  submitAction,
} from "./api";
import { ActionInspector } from "./components/ActionInspector";
import { EvidenceRail } from "./components/EvidenceRail";
import { Playfield } from "./components/Playfield";
import { SetupRail } from "./components/SetupRail";
import type { ApiHealth, DeckSummary, LegalAction, PolicySummary, RunSummary, SessionState } from "./types";

const DEFAULT_MAIN_DECK = "preset:main_deck_5hy_yotsuba_v1";
const RUN_DIR_STORAGE_KEY = "weiss-human-play.run-dir";

export function App() {
  const [health, setHealth] = useState<ApiHealth | null>(null);
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [policies, setPolicies] = useState<PolicySummary[]>([]);
  const [runDir, setRunDir] = useState(() => localStorage.getItem(RUN_DIR_STORAGE_KEY) ?? "");
  const [policyId, setPolicyId] = useState("main_league_selected");
  const [humanDeck, setHumanDeck] = useState(DEFAULT_MAIN_DECK);
  const [modelDeck, setModelDeck] = useState(DEFAULT_MAIN_DECK);
  const [humanSeat, setHumanSeat] = useState(0);
  const [seed, setSeed] = useState(20260521);
  const [mode, setMode] = useState<"study" | "freeplay">("study");
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [state, setState] = useState<SessionState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedActionId, setSelectedActionId] = useState<number | null>(null);

  const selectedRun = useMemo(() => runs.find((run) => run.run_dir === runDir), [runDir, runs]);

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

  async function startSession() {
    setBusy(true);
    setError(null);
    setSelectedActionId(null);
    try {
      const next = await createSession({
        run_dir: runDir,
        policy_id: policyId,
        human_seat: humanSeat,
        seed,
        human_deck: humanDeck,
        model_deck: modelDeck,
        mode,
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
      });
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
      setSelectedActionId(null);
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : String(actionError));
    } finally {
      setBusy(false);
    }
  }

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
    try {
      await closeSession(sessionId);
    } catch (closeError) {
      setError(closeError instanceof Error ? closeError.message : String(closeError));
    }
  }

  return (
    <div className={state ? "app-shell has-session" : "app-shell"}>
      <SetupRail
        health={health}
        decks={decks}
        runs={runs}
        policies={policies}
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
      <div className="workspace">
        <div className="workspace-topline">
          <span>{selectedRun ? selectedRun.label : "No run selected"}</span>
          <strong>{state ? `Session ${state.session_id.slice(0, 8)}` : "No active session"}</strong>
        </div>
        <Playfield state={state} loading={busy && !state} />
      </div>
      <ActionInspector state={state} selectedActionId={selectedActionId} busy={busy} onSelectAction={chooseAction} />
      <EvidenceRail health={health} state={state} onRefresh={refreshSession} onClose={endSession} />
    </div>
  );
}
