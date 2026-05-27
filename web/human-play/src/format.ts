import type { CardView, PlayerView } from "./types";

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
  const fromCounts = player.counts?.[zoneName];
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
