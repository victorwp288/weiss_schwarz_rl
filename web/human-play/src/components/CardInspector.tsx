import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { cardArtUrl, getCardInfo } from "../api";
import {
  cardColor,
  cardCost,
  cardLevel,
  cardName,
  cardNumber,
  cardPower,
  cardSoul,
  cardTriggers,
  cardType,
  cx,
  friendlyActionLabel,
  unwrapCard,
} from "../format";
import type { CardInfo, CardView, LegalAction } from "../types";

export type InspectTarget = {
  card: CardView;
  /** Viewport-space anchor of the hovered card. */
  anchor: { top: number; bottom: number; left: number; right: number };
  /** Legal actions involving this card right now (already filtered by the caller). */
  actions: LegalAction[];
};

type CardInspectorProps = {
  target: InspectTarget | null;
  humanTurn: boolean;
  spectate?: boolean;
  phase?: string | null;
};

/** Reference meaning of trigger icons (standard Weiss Schwarz rules). */
const TRIGGER_INFO: Record<string, string> = {
  soul: "+1 soul while attacking.",
  treasure: "On trigger: add it to hand, top of deck to stock.",
  comeback: "On trigger: return a character from waiting room to hand.",
  door: "On trigger: return a character from waiting room to hand.",
  return: "On trigger: bounce an opposing character to hand.",
  wind: "On trigger: bounce an opposing character to hand.",
  draw: "On trigger: draw a card.",
  book: "On trigger: draw a card.",
  pool: "On trigger: top of deck to stock.",
  bag: "On trigger: top of deck to stock.",
  shot: "On trigger: extra burst damage this attack.",
  gate: "On trigger: add a climax from waiting room to hand.",
  choice: "On trigger: choose a reward (hand or stock).",
  standby: "On trigger: put a character from waiting room onto stage.",
};

const PANEL_WIDTH = 282;

export function CardInspector({ target, humanTurn, spectate = false, phase }: CardInspectorProps) {
  const [visible, setVisible] = useState(false);
  const [info, setInfo] = useState<CardInfo | null>(null);
  const [top, setTop] = useState<number | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);
  const card = target ? unwrapCard(target.card) : null;
  const number = target ? cardNumber(target.card) : "";

  // Clamp to the viewport using the real rendered height (art makes it tall,
  // and the height changes again once the card text arrives).
  useLayoutEffect(() => {
    const panel = panelRef.current;
    if (!target || !visible || !panel) {
      setTop(null);
      return;
    }
    const clamp = () => {
      setTop(Math.max(8, Math.min(target.anchor.top - 40, window.innerHeight - panel.offsetHeight - 8)));
    };
    clamp();
    const observer = new ResizeObserver(clamp);
    observer.observe(panel);
    return () => observer.disconnect();
  }, [target, visible, info]);

  useEffect(() => {
    setVisible(false);
    if (!target) {
      return;
    }
    const timer = window.setTimeout(() => setVisible(true), 160);
    return () => window.clearTimeout(timer);
  }, [target]);

  useEffect(() => {
    setInfo(null);
    if (!target || !number.includes("/")) {
      return;
    }
    let cancelled = false;
    getCardInfo(number, unwrapCard(target.card)?.card_id ?? null)
      .then((payload) => {
        if (!cancelled) {
          setInfo(payload);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [target, number]);

  if (!target || !card || !visible) {
    return null;
  }

  const vw = window.innerWidth;
  const left =
    target.anchor.right + 14 + PANEL_WIDTH <= vw
      ? target.anchor.right + 14
      : Math.max(8, target.anchor.left - PANEL_WIDTH - 14);

  const level = cardLevel(target.card);
  const cost = cardCost(target.card);
  const power = cardPower(target.card);
  const soul = cardSoul(target.card);
  const triggers = cardTriggers(target.card);
  const type = cardType(target.card);
  const color = cardColor(target.card);

  const engine =
    info == null || (info.approx_ok == null && info.strict_ok == null)
      ? null
      : info.strict_ok
        ? { tone: "ok", text: "Engine: fully scripted" }
        : info.approx_ok
          ? { tone: "ok", text: "Engine: approximated (approx profile)" }
          : { tone: "warn", text: "Engine: vanilla — printed abilities are not simulated" };

  return (
    <aside
      ref={panelRef}
      className="inspector"
      style={{ left, top: top ?? 8, width: PANEL_WIDTH, visibility: top == null ? "hidden" : undefined }}
      aria-hidden
    >
      <header className="inspector__head">
        <b>{cardName(target.card)}</b>
        <span>{number}</span>
      </header>

      {number.includes("/") ? (
        <div className="inspector__art">
          <img src={cardArtUrl(number)} alt="" loading="lazy" draggable={false} />
        </div>
      ) : null}

      <div className="inspector__stats">
        <span className={cx("istat", `istat--${color}`)}>{color}</span>
        {type ? <span className="istat">{type}</span> : null}
        {level != null ? <span className="istat">Lv {level}</span> : null}
        {cost != null ? <span className="istat">Cost {cost}</span> : null}
        {power != null ? <span className="istat">{power} pow</span> : null}
        {soul != null ? <span className="istat">{soul} soul</span> : null}
      </div>

      <div className="inspector__now">
        <h4>Right now</h4>
        {target.actions.length ? (
          <ul>
            {target.actions.slice(0, 4).map((action) => (
              <li key={action.action_id}>{friendlyActionLabel(action)}</li>
            ))}
            {target.actions.length > 4 ? <li>…and {target.actions.length - 4} more</li> : null}
          </ul>
        ) : (
          <p>
            {spectate
              ? "Spectating — the model decides."
              : humanTurn
                ? `No legal action in the ${phase ?? "current"} phase.`
                : "Waiting for the model."}
          </p>
        )}
      </div>

      {triggers.length ? (
        <div className="inspector__triggers">
          {triggers.map((trigger) => (
            <p key={trigger}>
              <b>{trigger}</b> {TRIGGER_INFO[trigger.toLowerCase()] ?? "Trigger icon."}
            </p>
          ))}
        </div>
      ) : null}

      {info?.text && info.text !== "-" ? (
        <div className={cx("inspector__text", engine?.tone === "warn" && "is-vanilla")}>
          {info.text.split("\n").map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>
      ) : null}

      {engine ? <div className={cx("inspector__engine", `is-${engine.tone}`)}>{engine.text}</div> : null}

      {info?.traits?.length ? <div className="inspector__traits">{info.traits.join(" · ")}</div> : null}
    </aside>
  );
}
