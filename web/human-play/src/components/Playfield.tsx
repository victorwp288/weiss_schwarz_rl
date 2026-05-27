import { Crown, EyeOff, RotateCw, Sparkles } from "lucide-react";

import { cardDetails, cardName, compactId, cx, zoneCards, zoneCount } from "../format";
import type { CardView, HumanDecisionView, PlayerView, SessionState, StageSlot } from "../types";

type PlayfieldProps = {
  state: SessionState | null;
  loading: boolean;
};

export function Playfield({ state, loading }: PlayfieldProps) {
  const view = state?.view;
  const human = view ? playerForSeat(view, state.human_seat) : undefined;
  const model = view ? playerForSeat(view, state.model_seat) : undefined;
  const humanHand = zoneCards(human, "hand");

  return (
    <main className="playfield-shell" aria-label="Weiss Schwarz playfield">
      <div className="table-header">
        <div>
          <span className="eyeline">Live simulator view</span>
          <h2>{state ? (state.terminal ? "Game complete" : state.human_turn ? "Your decision" : "Model resolving") : "Ready to start"}</h2>
        </div>
        <div className={cx("turn-badge", state?.human_turn && "is-human", state?.terminal && "is-terminal")}>
          {state ? (state.terminal ? "Terminal" : state.human_turn ? "Human turn" : "Model turn") : "Setup"}
        </div>
      </div>

      <div className="table-status">
        <StatusItem label="Phase" value={view?.summary?.phase ?? "--"} />
        <StatusItem label="Decision" value={String(view?.summary?.decision_count ?? view?.decision_id ?? "--")} />
        <StatusItem label="Kind" value={view?.summary?.decision_kind ?? "--"} />
        <StatusItem label="Fingerprint" value={compactId(view?.legal_fingerprint64, 8) || "--"} />
      </div>

      <div className="felt-table">
        {loading ? <div className="loading-veil">Loading simulator state...</div> : null}
        <PlayerBand player={model} label="Model" seat={state?.model_seat ?? 1} side="opponent" />
        <StageGrid view={view} humanSeat={state?.human_seat ?? 0} modelSeat={state?.model_seat ?? 1} />
        <PlayerBand player={human} label="Human" seat={state?.human_seat ?? 0} side="human" />
      </div>

      <section className="hand-rail" aria-label="Human hand">
        <div className="hand-header">
          <div>
            <h3>Hand</h3>
            <span>{humanHand.length || zoneCount(human, "hand")} cards visible to you</span>
          </div>
          <div className="hand-tools">
            <span>Clock {zoneCount(human, "clock")}</span>
            <span>Stock {zoneCount(human, "stock")}</span>
          </div>
        </div>
        <div className="hand-cards">
          {humanHand.length ? (
            humanHand.map((card, index) => <CardFace key={card.id ?? card.card_no ?? index} card={card} />)
          ) : (
            <div className="empty-hand">
              <Sparkles size={16} aria-hidden />
              <span>Human-readable card names appear here when the simulator exposes visible hand cards.</span>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function StatusItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PlayerBand({
  player,
  label,
  seat,
  side,
}: {
  player?: PlayerView;
  label: string;
  seat: number;
  side: "human" | "opponent";
}) {
  return (
    <section className={cx("player-band", side === "opponent" && "is-opponent")}>
      <div className="player-title">
        <Crown size={15} aria-hidden />
        <strong>
          {label} <span>Seat {seat}</span>
        </strong>
      </div>
      <div className="zone-strip">
        <ZoneChip label="Level" value={zoneCount(player, "level")} />
        <ZoneChip label="Clock" value={zoneCount(player, "clock")} />
        <ZoneChip label="Stock" value={zoneCount(player, "stock")} />
        <ZoneChip label="Memory" value={zoneCount(player, "memory")} />
        <ZoneChip label="Waiting" value={zoneCount(player, "waiting_room")} />
        <ZoneChip label="Deck" value={zoneCount(player, "deck")} />
      </div>
      {side === "opponent" ? (
        <div className="hidden-note">
          <EyeOff size={14} aria-hidden />
          Opponent hidden zones remain redacted.
        </div>
      ) : null}
    </section>
  );
}

function ZoneChip({ label, value }: { label: string; value: number }) {
  return (
    <div className="zone-chip">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StageGrid({
  view,
  humanSeat,
  modelSeat,
}: {
  view?: HumanDecisionView;
  humanSeat: number;
  modelSeat: number;
}) {
  const slots = normalizeStageSlots(view);
  const modelSlots = stageSlotsForSeat(slots, modelSeat);
  const humanSlots = stageSlotsForSeat(slots, humanSeat);

  return (
    <section className="stage-grid" aria-label="Stage">
      <StageRow label="Opponent stage" slots={modelSlots} fallbackSeat={modelSeat} />
      <div className="center-line">
        <RotateCw size={15} aria-hidden />
        <span>Public stage and legal action surface</span>
      </div>
      <StageRow label="Human stage" slots={humanSlots} fallbackSeat={humanSeat} />
    </section>
  );
}

function StageRow({ label, slots, fallbackSeat }: { label: string; slots: StageSlot[]; fallbackSeat: number }) {
  const normalized: StageSlot[] = slots.length
    ? slots
    : Array.from({ length: 5 }, (_, lane) => ({ seat: fallbackSeat, lane }));
  return (
    <div className="stage-row">
      <span className="stage-row-label">{label}</span>
      <div className="stage-slots">
        {normalized.slice(0, 5).map((slot, index) => (
          <StageCell key={`${slot.seat ?? fallbackSeat}-${slot.row ?? "row"}-${slot.lane ?? index}`} slot={slot} />
        ))}
      </div>
    </div>
  );
}

function StageCell({ slot }: { slot: StageSlot }) {
  const card = slot.card ?? slot.cards?.[0] ?? null;
  const occupied = Boolean(card || slot.occupied || slot.label);
  return (
    <div className={cx("stage-cell", occupied && "is-occupied")}>
      <span>{slot.row ?? `Lane ${Number(slot.lane ?? 0) + 1}`}</span>
      <strong>{card ? cardName(card) : slot.label ?? "Open slot"}</strong>
    </div>
  );
}

function CardFace({ card }: { card: CardView }) {
  const details = cardDetails(card);
  return (
    <article className={cx("card-face", (card.hidden || card.redacted) && "is-redacted")}>
      <strong>{cardName(card)}</strong>
      <span>
        {details?.card_no ?? card.card_ref ?? details?.id ?? "card"}
        {details?.level != null ? ` - L${details.level}` : ""}
      </span>
      {details?.power != null ? <em>{details.power} power</em> : null}
    </article>
  );
}

function playerForSeat(view: HumanDecisionView, seat: number): PlayerView | undefined {
  return view.players?.find((player) => Number(player.seat) === Number(seat));
}

function normalizeStageSlots(view?: HumanDecisionView): StageSlot[] {
  const layout = view?.stage_layout;
  if (Array.isArray(layout)) {
    return layout;
  }
  if (Array.isArray(layout?.slots)) {
    return layout.slots;
  }
  const playerSlots = view?.players?.flatMap((player) =>
    Array.isArray(player.stage) ? player.stage.map((slot) => ({ ...slot, seat: slot.seat ?? player.seat })) : [],
  );
  return playerSlots ?? [];
}

function stageSlotsForSeat(slots: StageSlot[], seat: number): StageSlot[] {
  return slots
    .filter((slot) => slot.seat == null || Number(slot.seat) === Number(seat))
    .sort((left, right) => Number(left.lane ?? 0) - Number(right.lane ?? 0));
}
