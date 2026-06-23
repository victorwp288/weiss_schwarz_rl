import type { CardColor, CardView, LegalAction, PlayerView, StageSlot } from "./types";

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function compactId(value: string | number | null | undefined, size = 10): string {
  const text = value == null ? "" : String(value);
  if (text.length <= size * 2 + 1) {
    return text;
  }
  return `${text.slice(0, size)}...${text.slice(-size)}`;
}

export function formatProbability(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  return `${Math.round(value * 1000) / 10}%`;
}

export function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "--";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(1)} s`;
}

export function cardName(card: CardView | null | undefined): string {
  const resolved = card?.card ?? card;
  if (!resolved || resolved.hidden || resolved.redacted) {
    return "Redacted";
  }
  return String(resolved.name ?? resolved.label ?? resolved.title ?? resolved.card_no ?? resolved.id ?? "Unknown card");
}

export function cardDetails(card: CardView | null | undefined): CardView | null {
  return card?.card ?? card ?? null;
}

export function zoneCount(player: PlayerView | undefined, zoneName: string): number {
  if (!player) {
    return 0;
  }
  // Real views key counts as `${zone}_count`; older fixtures use the bare zone name.
  const counts = player.counts ?? {};
  const fromCounts = counts[`${zoneName}_count`] ?? counts[zoneName];
  if (typeof fromCounts === "number") {
    return fromCounts;
  }
  const zone = player.zones?.[zoneName];
  if (Array.isArray(zone)) {
    return zone.length;
  }
  if (zone && typeof zone.count === "number") {
    return zone.count;
  }
  if (zone?.cards) {
    return zone.cards.length;
  }
  return 0;
}

const NUMBER_RE = /-?\d+(\.\d+)?/;

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const match = value.match(NUMBER_RE);
    if (match) {
      return Number(match[0]);
    }
  }
  return null;
}

/** Returns the underlying card fields, unwrapping the `{ card: {...} }` envelope. */
export function unwrapCard(card: CardView | null | undefined): CardView | null {
  if (!card) {
    return null;
  }
  return card.card ?? card;
}

export function isFaceDown(card: CardView | null | undefined): boolean {
  const resolved = unwrapCard(card);
  if (!resolved) {
    return true;
  }
  return Boolean(card?.hidden || card?.redacted || resolved.hidden || resolved.redacted) || !resolved.name;
}

export function cardColor(card: CardView | null | undefined): CardColor {
  return unwrapCard(card)?.color ?? "neutral";
}

export function cardType(card: CardView | null | undefined): string | null {
  return unwrapCard(card)?.card_type ?? null;
}

export function cardLevel(card: CardView | null | undefined): number | null {
  return toNumber(unwrapCard(card)?.level);
}

export function cardCost(card: CardView | null | undefined): number | null {
  return toNumber(unwrapCard(card)?.cost);
}

export function cardPower(card: CardView | null | undefined): number | null {
  return toNumber(unwrapCard(card)?.power);
}

export function cardSoul(card: CardView | null | undefined): number | null {
  return toNumber(unwrapCard(card)?.soul);
}

export function cardTriggers(card: CardView | null | undefined): string[] {
  const triggers = unwrapCard(card)?.triggers;
  return Array.isArray(triggers) ? triggers.map(String) : [];
}

export function cardNumber(card: CardView | null | undefined): string {
  const resolved = unwrapCard(card);
  return String(resolved?.card_no ?? card?.card_ref ?? resolved?.id ?? "");
}

/** The card's stable reference (e.g. "self.hand.0") used to map cards to legal actions. */
export function cardRefOf(card: CardView | null | undefined): string | null {
  return card?.card_ref ?? null;
}

export function stageCard(slot: StageSlot | undefined): CardView | null {
  if (!slot) {
    return null;
  }
  return slot.card ?? slot.cards?.[0] ?? null;
}

export function stageOccupied(slot: StageSlot | undefined): boolean {
  if (!slot) {
    return false;
  }
  if (slot.empty === true) {
    return false;
  }
  return Boolean(stageCard(slot) || slot.occupied);
}

/** Order stage slots into the real 3-center / 2-back playmat shape. */
export function stageRows(slots: StageSlot[]): { center: StageSlot[]; back: StageSlot[] } {
  const indexOf = (slot: StageSlot, fallback: number) =>
    Number(slot.slot ?? slot.lane ?? fallback);
  const ordered = [...slots].sort((a, b) => indexOf(a, 0) - indexOf(b, 0));
  const center = ordered.filter((slot) => (slot.row ?? "").toLowerCase() !== "back");
  const back = ordered.filter((slot) => (slot.row ?? "").toLowerCase() === "back");
  return {
    center: padSlots(center, 3),
    back: padSlots(back, 2),
  };
}

function padSlots(slots: StageSlot[], size: number): StageSlot[] {
  if (slots.length >= size) {
    return slots.slice(0, size);
  }
  return [...slots, ...Array.from({ length: size - slots.length }, () => ({ empty: true }) as StageSlot)];
}

/** Every legal action that operates on a given card (hand index or card/slot ref). */
export function actionsForCard(
  actions: LegalAction[],
  options: { cardRef?: string | null; handIndex?: number | null },
): LegalAction[] {
  const { cardRef, handIndex } = options;
  return actions.filter((action) => {
    const refs = action.source_refs ?? [];
    if (cardRef && refs.some((ref) => ref.ref_id === cardRef || ref.card_ref === cardRef)) {
      return true;
    }
    if (handIndex != null && typeof action.params?.hand_index === "number") {
      return action.params.hand_index === handIndex;
    }
    return false;
  });
}

/** Find the single legal action that operates on a given card reference, if unambiguous. */
export function actionForCard(
  actions: LegalAction[],
  options: { cardRef?: string | null; handIndex?: number | null },
): LegalAction | null {
  const matches = actionsForCard(actions, options);
  return matches.length === 1 ? matches[0] : (matches[0] ?? null);
}

/** Every legal action sourced from a given stage slot (by ref or slot number). */
export function actionsForStageSlot(actions: LegalAction[], slot: StageSlot): LegalAction[] {
  const slotRef = slot.card_ref ?? slot.slot_ref ?? slot.card?.card_ref ?? null;
  const slotIndex = slot.slot ?? slot.lane ?? null;
  return actions.filter((action) => {
    const refs = action.source_refs ?? [];
    if (slotRef && refs.some((ref) => ref.ref_id === slotRef || ref.card_ref === slotRef)) {
      return true;
    }
    if (slotIndex != null) {
      const paramSlot = action.params?.slot;
      if (typeof paramSlot === "number" && refs.some((ref) => ref.zone === "stage")) {
        return paramSlot === Number(slotIndex);
      }
    }
    return false;
  });
}

/** The stage slot a play/move action targets, if any. */
export function actionTargetSlot(action: LegalAction): number | null {
  const target = action.params?.stage_slot;
  return typeof target === "number" ? target : null;
}

const HAND_CARD_LABEL_RE = /hand card \d+/i;

/** Prefer real card names over "hand card N" indices in action labels. */
export function friendlyActionLabel(action: LegalAction): string {
  const label = action.label ?? action.short_label ?? `Action ${action.action_id}`;
  const refs = [...(action.source_refs ?? []), ...(action.target_refs ?? [])];
  const named = refs.map((ref) => unwrapCard(ref.card)?.name).find((name) => name);
  if (!named) {
    return label;
  }
  const replaced = label.replace(HAND_CARD_LABEL_RE, named);
  return replaced !== label ? replaced : label;
}

export function zoneCards(player: PlayerView | undefined, zoneName: string): CardView[] {
  const zone = player?.zones?.[zoneName];
  if (Array.isArray(zone)) {
    return zone;
  }
  if (zone?.cards) {
    return zone.cards;
  }
  return [];
}
