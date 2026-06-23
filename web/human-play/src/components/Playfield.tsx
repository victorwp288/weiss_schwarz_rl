import { Boxes, Hand, Layers, Sparkles, Trash2 } from "lucide-react";
import type { CSSProperties, DragEvent, ReactNode } from "react";
import { useEffect } from "react";

import {
  actionTargetSlot,
  actionsForCard,
  actionsForStageSlot,
  cardName,
  cx,
  stageCard,
  stageOccupied,
  stageRows,
  unwrapCard,
  zoneCards,
  zoneCount,
} from "../format";
import type {
  ActionFocus,
  CardView,
  HumanDecisionView,
  LegalAction,
  PlayerView,
  PublicHistoryEntry,
  SessionState,
  StageSlot,
} from "../types";
import { CardFace } from "./Card";
import type { InspectTarget } from "./CardInspector";

type PlayfieldProps = {
  state: SessionState | null;
  loading: boolean;
  selectedActionId?: number | null;
  markedActionIds?: ReadonlySet<number>;
  focus?: ActionFocus | null;
  onFocusChange?: (focus: ActionFocus | null) => void;
  onSelectAction?: (action: LegalAction) => void;
  onInspect?: (target: InspectTarget | null) => void;
  onNudge?: (text: string) => void;
};

export function Playfield({
  state,
  loading,
  selectedActionId = null,
  markedActionIds = new Set(),
  focus = null,
  onFocusChange = () => {},
  onSelectAction = () => {},
  onInspect = () => {},
  onNudge = () => {},
}: PlayfieldProps) {
  const view = state?.view;
  const human = view ? playerForSeat(view, state.human_seat) : undefined;
  const model = view ? playerForSeat(view, state.model_seat) : undefined;
  const humanHand = zoneCards(human, "hand");
  const legalActions = view?.legal_actions ?? [];
  const spectate = Boolean(state?.spectate);
  const interactive = Boolean(state?.human_turn && !loading);
  const phase = view?.summary?.phase;

  const feedTop = spectate ? lastEntriesForSeat(state, state?.model_seat ?? 1) : modelActionsSinceLastHumanAct(state);
  const feedBottom = spectate ? lastEntriesForSeat(state, state?.human_seat ?? 0) : [];

  // Whole-board actions (pass, confirm mulligan…): no card refs involved.
  const boardActions = legalActions.filter(
    (action) => !(action.source_refs?.length || action.target_refs?.length),
  );

  const noActionReason = (card: CardView | null): string => {
    const name = card ? cardName(card) : "That card";
    if (!state || state.terminal) {
      return "The match is over.";
    }
    if (spectate) {
      return "Spectate mode — the model plays both seats.";
    }
    if (!state.human_turn) {
      return "Waiting for the model to move.";
    }
    const passOnly = legalActions.length > 0 && legalActions.every((a) => a.is_pass || a.family === "pass");
    if (passOnly) {
      return `${name} can't act right now — your only legal move is Pass.`;
    }
    return `${name} has no legal action in the ${phase ?? "current"} phase.`;
  };

  const inspectFor = (card: CardView | null | undefined, actions: LegalAction[]) => {
    if (!card || !unwrapCard(card)?.name) {
      return undefined;
    }
    return (anchor: { top: number; bottom: number; left: number; right: number } | null) =>
      onInspect(anchor ? { card, anchor, actions } : null);
  };

  const turnState = !state ? "setup" : state.terminal ? "over" : state.human_turn ? "you" : "foe";
  const turnLabel =
    turnState === "over"
      ? "Match over"
      : turnState === "you"
        ? "Your turn"
        : turnState === "foe"
          ? "Opponent’s turn"
          : "Ready";

  useEffect(() => {
    if (!focus) {
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onFocusChange(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focus, onFocusChange]);

  const playHandCard = (card: CardView, index: number) => {
    if (!interactive) {
      return;
    }
    if (focus && focus.handIndex === index) {
      onFocusChange(null);
      return;
    }
    const matches = actionsForCard(legalActions, { cardRef: card.card_ref, handIndex: index });
    if (matches.length === 1) {
      onFocusChange(null);
      onSelectAction(matches[0]);
    } else if (matches.length > 1) {
      onFocusChange({
        title: cardName(card),
        cardRef: card.card_ref ?? null,
        handIndex: index,
        actions: matches,
      });
    }
  };

  const dragHandCard = (card: CardView, index: number) => (event: DragEvent<HTMLButtonElement>) => {
    const matches = actionsForCard(legalActions, { cardRef: card.card_ref, handIndex: index });
    if (!interactive || !matches.length) {
      event.preventDefault();
      return;
    }
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", String(index));
    onFocusChange({
      title: cardName(card),
      cardRef: card.card_ref ?? null,
      handIndex: index,
      actions: matches,
    });
  };

  // A drag that ends anywhere other than a drop target just disarms the card.
  const dragEnd = () => onFocusChange(null);

  // Single non-slot action (clock, climax, mulligan toggle): dropping the card
  // anywhere on your own side of the table performs it.
  const yourSideDropAction =
    focus && focus.handIndex != null && focus.actions.length === 1 && actionTargetSlot(focus.actions[0]) == null
      ? focus.actions[0]
      : null;
  // An armed attacker with exactly one option: drop it onto the opponent's side.
  const foeSideDropAction = focus && focus.slotRef != null && focus.actions.length === 1 ? focus.actions[0] : null;

  return (
    <div className="playfield">
      <PlayerHud player={model} side="foe" seat={state?.model_seat ?? 1}>
        <HudFeed entries={feedTop} terminal={Boolean(state?.terminal)} tone="ai" label="Opponent's last actions" />
      </PlayerHud>

      <div className={cx("arena", loading && "is-loading")}>
        <p className="redacted-note">Opponent hidden zones remain redacted.</p>

        <BoardSide
          player={model}
          view={view}
          seat={state?.model_seat ?? 1}
          side="foe"
          dropAction={foeSideDropAction}
          onFocusChange={onFocusChange}
          onSelectAction={onSelectAction}
          inspectFor={inspectFor}
        />

        <div className="divider">
          <span className="divider__rule" />
          <span className={cx("turn-token", (turnState === "foe" || spectate) && "is-foe", turnState === "over" && "is-over")}>
            {turnState !== "over" ? <span className="turn-token__dot" /> : null}
            {spectate && turnState !== "over" ? "Spectating" : turnLabel}
            {phase && turnState !== "over" ? <span className="turn-token__phase">{phase}</span> : null}
          </span>
          {!spectate && interactive
            ? boardActions.slice(0, 2).map((action) => (
                <button
                  key={action.action_id}
                  type="button"
                  className={cx("board-btn", (action.is_pass || action.family === "pass") && "board-btn--pass")}
                  onClick={() => {
                    onFocusChange(null);
                    onSelectAction(action);
                  }}
                  title={action.description ?? action.label ?? undefined}
                >
                  {action.label ?? action.short_label ?? "Continue"}
                </button>
              ))
            : null}
          <span className="divider__rule" />
        </div>

        <BoardSide
          player={human}
          view={view}
          seat={state?.human_seat ?? 0}
          side="you"
          interactive={interactive}
          legalActions={legalActions}
          selectedActionId={selectedActionId}
          markedActionIds={markedActionIds}
          focus={focus}
          dropAction={yourSideDropAction}
          onFocusChange={onFocusChange}
          onSelectAction={onSelectAction}
          inspectFor={inspectFor}
          onNudgeCard={(card) => onNudge(noActionReason(card))}
        />

        {focus ? (
          <div className="focus-hint" role="status">
            <span className="stamp stamp--sm" aria-hidden>
              的
            </span>
            <span>
              <b>{focus.title}</b>
              {focus.actions.some((action) => actionTargetSlot(action) != null)
                ? " — tap a marked slot, or pick the exact move on the right."
                : " — pick the exact move on the right."}
            </span>
            <button type="button" className="focus-hint__cancel" onClick={() => onFocusChange(null)}>
              Cancel
            </button>
          </div>
        ) : null}
      </div>

      <PlayerHud player={human} side="you" seat={state?.human_seat ?? 0}>
        {spectate ? (
          <HudFeed entries={feedBottom} terminal={Boolean(state?.terminal)} tone="shu" label="Seat's last actions" />
        ) : null}
      </PlayerHud>

      <section className="hand" aria-label="Your hand">
        <div className="hand__label">
          <h3>Your hand</h3>
          <span>
            {humanHand.length || zoneCount(human, "hand")} cards
            {spectate ? " · spectating" : interactive ? " · tap a stamped card to play it" : ""}
          </span>
        </div>
        {humanHand.length ? (
          <div
            className={cx("hand__fan", focus && focus.handIndex != null && "has-focus")}
            style={{ ["--overlap" as string]: humanHand.length > 7 ? "-24px" : "-13px" } as CSSProperties}
          >
            {humanHand.map((card, index) => {
              const matches = interactive
                ? actionsForCard(legalActions, { cardRef: card.card_ref, handIndex: index })
                : [];
              const focused = focus?.handIndex === index;
              const marked = matches.some((action) => markedActionIds.has(action.action_id));
              const center = (humanHand.length - 1) / 2;
              const angle = (index - center) * 6;
              const lift = Math.abs(index - center) * 5;
              return (
                <CardFace
                  key={card.card_ref ?? unwrapCard(card)?.card_id ?? index}
                  card={card}
                  actionable={matches.length > 0}
                  selected={focused || marked || (matches.length === 1 && matches[0].action_id === selectedActionId)}
                  onClick={
                    matches.length
                      ? () => playHandCard(card, index)
                      : state && !state.terminal
                        ? () => onNudge(noActionReason(card))
                        : undefined
                  }
                  draggable={matches.length > 0}
                  onDragStart={dragHandCard(card, index)}
                  onDragEnd={dragEnd}
                  onHover={inspectFor(card, matches)}
                  title={
                    matches.length === 1
                      ? `${cardName(card)} — ${matches[0].label ?? "play"}`
                      : matches.length > 1
                        ? `${cardName(card)} — drag it out, or tap and choose`
                        : cardName(card)
                  }
                  style={{ ["--rot" as string]: `${angle}deg`, ["--lift" as string]: `${lift}px` } as CSSProperties}
                />
              );
            })}
          </div>
        ) : (
          <div className="empty-hand">
            <Sparkles size={16} aria-hidden />
            {state ? "Your hand is empty." : "Start a match to draw your hand."}
          </div>
        )}
      </section>
    </div>
  );
}

function HudFeed({
  entries,
  terminal,
  tone,
  label,
}: {
  entries: PublicHistoryEntry[];
  terminal: boolean;
  tone: "ai" | "shu";
  label: string;
}) {
  if (!entries.length || terminal) {
    return null;
  }
  return (
    <div className="foe-feed" aria-label={label}>
      {entries.map((entry) => (
        <span key={entry.decision_index} className="foe-callout" title={entry.phase ? `during ${entry.phase}` : undefined}>
          <span className={cx("stamp", "stamp--sm", tone === "ai" && "stamp--ai")} aria-hidden>
            打
          </span>
          <em>{entry.label}</em>
        </span>
      ))}
    </div>
  );
}

function lastEntriesForSeat(state: SessionState | null, seat: number): PublicHistoryEntry[] {
  return (state?.history ?? []).filter((entry) => Number(entry.actor_seat) === Number(seat)).slice(-3);
}

function modelActionsSinceLastHumanAct(state: SessionState | null): PublicHistoryEntry[] {
  const history = state?.history ?? [];
  if (history.length) {
    let lastHuman = -1;
    for (let i = history.length - 1; i >= 0; i -= 1) {
      if (history[i].actor_kind === "human") {
        lastHuman = i;
        break;
      }
    }
    return history.slice(lastHuman + 1).filter((entry) => entry.actor_kind === "model");
  }
  // Older servers: fall back to the single most recent model action.
  const recent = state?.model?.recent_actions ?? [];
  if (!recent.length) {
    return [];
  }
  const last = recent[recent.length - 1];
  return [
    {
      decision_index: last.decision_index,
      actor_seat: last.actor_seat,
      actor_kind: "model",
      label: last.action_label,
      elapsed_ms: last.elapsed_ms,
    },
  ];
}

function PlayerHud({
  player,
  side,
  seat,
  children,
}: {
  player?: PlayerView;
  side: "foe" | "you";
  seat: number;
  children?: ReactNode;
}) {
  const level = zoneCount(player, "level");
  const clock = zoneCount(player, "clock");
  return (
    <header className={cx("hud", side === "foe" ? "hud--foe" : "hud--you")}>
      <div className="nameplate">
        <span className="kanji-tab" aria-hidden>
          {side === "foe" ? "相手" : "自分"}
        </span>
        <div className="name-block">
          <span className="name">{side === "foe" ? "Opponent" : "You"}</span>
          <span className="seat">Seat {seat}</span>
        </div>
      </div>
      {children}
      <div className="vitals">
        <span className="vital vital--level">
          <span className="vital__label">Lv</span>
          <span className="pips">
            {Array.from({ length: 3 }, (_, i) => (
              <i key={i} className={cx("pip", i < level && "on")} />
            ))}
          </span>
        </span>
        <span className="vital vital--clock" title={`${clock} damage — level up at 7`}>
          <span className="vital__label">Clock</span>
          <span className="clock-track">
            {Array.from({ length: 7 }, (_, i) => (
              <i key={i} className={cx("tick", i < clock && "on")} />
            ))}
          </span>
        </span>
        <span className="vital vital--stock">
          <span className="vital__label">Stock</span>
          <span className="vital__n">{zoneCount(player, "stock")}</span>
        </span>
        <span className="hand-count" title="Cards in hand">
          <Hand size={14} aria-hidden /> {zoneCount(player, "hand")}
        </span>
      </div>
    </header>
  );
}

function BoardSide({
  player,
  view,
  seat,
  side,
  interactive,
  legalActions = [],
  selectedActionId,
  markedActionIds = new Set(),
  focus = null,
  dropAction = null,
  onFocusChange = () => {},
  onSelectAction,
  inspectFor,
  onNudgeCard,
}: {
  player?: PlayerView;
  view?: HumanDecisionView;
  seat: number;
  side: "foe" | "you";
  interactive?: boolean;
  legalActions?: LegalAction[];
  selectedActionId?: number | null;
  markedActionIds?: ReadonlySet<number>;
  focus?: ActionFocus | null;
  dropAction?: LegalAction | null;
  onFocusChange?: (focus: ActionFocus | null) => void;
  onSelectAction?: (action: LegalAction) => void;
  inspectFor?: (
    card: CardView | null | undefined,
    actions: LegalAction[],
  ) => ((anchor: { top: number; bottom: number; left: number; right: number } | null) => void) | undefined;
  onNudgeCard?: (card: CardView | null) => void;
}) {
  const { center, back } = stageRows(stageSlotsForSeat(normalizeStageSlots(view, player), seat));

  const fireDrop = (action: LegalAction) => (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    onFocusChange(null);
    onSelectAction?.(action);
  };
  const allowDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  };

  const targetBySlot = new Map<number, LegalAction>();
  if (focus && side === "you") {
    for (const action of focus.actions) {
      const target = actionTargetSlot(action);
      if (target != null && !targetBySlot.has(target)) {
        targetBySlot.set(target, action);
      }
    }
  }

  const clickStageCard = (slot: StageSlot, matches: LegalAction[]) => {
    if (matches.length === 1) {
      onFocusChange(null);
      onSelectAction?.(matches[0]);
      return;
    }
    const slotRef = slot.card_ref ?? slot.slot_ref ?? null;
    if (focus && focus.slotRef != null && focus.slotRef === slotRef) {
      onFocusChange(null);
      return;
    }
    onFocusChange({
      title: cardName(stageCard(slot)),
      slotRef,
      actions: matches,
    });
  };

  const renderSlot = (slot: StageSlot, key: number) => {
    const card = stageCard(slot);
    const occupied = stageOccupied(slot);
    const slotIndex = Number(slot.slot ?? slot.lane ?? key);
    const targetAction = targetBySlot.get(slotIndex);
    const matches = interactive && occupied ? actionsForStageSlot(legalActions, slot) : [];
    const slotRef = slot.card_ref ?? slot.slot_ref ?? null;
    const focused = Boolean(focus && focus.slotRef != null && focus.slotRef === slotRef);
    const marked = matches.some((action) => markedActionIds.has(action.action_id));
    return (
      <div
        key={slot.slot_ref ?? key}
        className={cx("slot", occupied && "filled", targetAction && "slot--target")}
        onDragOver={targetAction ? allowDrop : undefined}
        onDragEnter={targetAction ? (event) => event.currentTarget.classList.add("drag-over") : undefined}
        onDragLeave={targetAction ? (event) => event.currentTarget.classList.remove("drag-over") : undefined}
        onDrop={targetAction ? fireDrop(targetAction) : undefined}
      >
        {targetAction ? (
          <button
            type="button"
            className="slot__target-btn"
            onClick={() => {
              onFocusChange(null);
              onSelectAction?.(targetAction);
            }}
            title={targetAction.label ?? "Play here"}
          >
            的
          </button>
        ) : occupied ? (
          <CardFace
            card={card}
            size="sm"
            rested={slot.rested || slot.has_attacked}
            actionable={matches.length > 0}
            selected={focused || marked || (matches.length === 1 && matches[0].action_id === selectedActionId)}
            onClick={
              matches.length
                ? () => clickStageCard(slot, matches)
                : interactive && onNudgeCard
                  ? () => onNudgeCard(card)
                  : undefined
            }
            onHover={inspectFor?.(card, matches)}
            draggable={matches.length > 0}
            onDragStart={(event) => {
              event.dataTransfer.effectAllowed = "move";
              event.dataTransfer.setData("text/plain", slotRef ?? String(slotIndex));
              onFocusChange({ title: cardName(card), slotRef, actions: matches });
            }}
            onDragEnd={() => onFocusChange(null)}
          />
        ) : (
          <span className="slot__label">{slot.label ?? "Open"}</span>
        )}
      </div>
    );
  };

  return (
    <div className={cx("arena-side", side === "foe" ? "arena-side--foe" : "arena-side--you")}>
      <div
        className={cx("side-stage", dropAction && "drop-ready")}
        onDragOver={dropAction ? allowDrop : undefined}
        onDrop={dropAction ? fireDrop(dropAction) : undefined}
        title={dropAction ? (dropAction.label ?? undefined) : undefined}
      >
        <div className="stage-row back">{back.map(renderSlot)}</div>
        <div className="stage-row center">{center.map(renderSlot)}</div>
      </div>
      <div className="zones">
        <ZoneChip label="Deck" tone="deck" icon={<Layers size={14} />} value={zoneCount(player, "deck")} />
        <ZoneChip label="Waiting" icon={<Trash2 size={14} />} value={zoneCount(player, "waiting_room")} />
        <ZoneChip label="Climax" icon={<Sparkles size={14} />} value={zoneCount(player, "climax")} />
        <ZoneChip label="Memory" icon={<Boxes size={14} />} value={zoneCount(player, "memory")} />
      </div>
    </div>
  );
}

function ZoneChip({ label, value, icon, tone }: { label: string; value: number; icon: ReactNode; tone?: string }) {
  return (
    <div className={cx("zone-chip", tone && `zone-chip--${tone}`)}>
      <span className="zone-chip__ico">{icon}</span>
      <span className="zone-chip__meta">
        <span className="zone-chip__label">{label}</span>
        <span className="zone-chip__n">{value}</span>
      </span>
    </div>
  );
}

function playerForSeat(view: HumanDecisionView, seat: number): PlayerView | undefined {
  return view.players?.find((player) => Number(player.seat) === Number(seat));
}

function normalizeStageSlots(view: HumanDecisionView | undefined, player: PlayerView | undefined): StageSlot[] {
  if (Array.isArray(player?.stage) && player.stage.length) {
    return player.stage;
  }
  const layout = view?.stage_layout;
  if (Array.isArray(layout)) {
    return layout;
  }
  if (Array.isArray(layout?.slots)) {
    return layout.slots;
  }
  return (
    view?.players?.flatMap((p) =>
      Array.isArray(p.stage) ? p.stage.map((slot) => ({ ...slot, seat: slot.seat ?? p.seat })) : [],
    ) ?? []
  );
}

function stageSlotsForSeat(slots: StageSlot[], seat: number): StageSlot[] {
  const owned = slots.filter((slot) =>
    slot.seat == null && slot.owner_seat == null ? true : Number(slot.seat ?? slot.owner_seat) === Number(seat),
  );
  return owned.length ? owned : slots;
}
