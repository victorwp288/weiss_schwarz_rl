import type { CSSProperties, DragEventHandler } from "react";
import { useState } from "react";

import { cardArtUrl } from "../api";
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
  isFaceDown,
} from "../format";
import type { CardView } from "../types";

type CardFaceProps = {
  card: CardView | null | undefined;
  size?: "xs" | "sm" | "md";
  selected?: boolean;
  actionable?: boolean;
  rested?: boolean;
  draggable?: boolean;
  onDragStart?: DragEventHandler<HTMLButtonElement>;
  onDragEnd?: DragEventHandler<HTMLButtonElement>;
  onClick?: () => void;
  /** Hover inspection: called with the card's viewport rect on enter, null on leave. */
  onHover?: (anchor: { top: number; bottom: number; left: number; right: number } | null) => void;
  title?: string;
  style?: CSSProperties;
};

/** A Weiss Schwarz card. Shows the official scan when the art proxy has it,
 *  otherwise a woodblock-print label face. Seigaiha back when face-down. */
export function CardFace({
  card,
  size = "md",
  selected,
  actionable,
  rested,
  draggable,
  onDragStart,
  onDragEnd,
  onClick,
  onHover,
  title,
  style,
}: CardFaceProps) {
  const hoverProps = onHover
    ? {
        onMouseEnter: (event: { currentTarget: Element }) => {
          const r = event.currentTarget.getBoundingClientRect();
          onHover({ top: r.top, bottom: r.bottom, left: r.left, right: r.right });
        },
        onMouseLeave: () => onHover(null),
      }
    : {};
  const [artOk, setArtOk] = useState(false);
  const faceDown = isFaceDown(card);
  const color = cardColor(card);
  const isClimax = cardType(card) === "climax";
  const level = cardLevel(card);
  const cost = cardCost(card);
  const power = cardPower(card);
  const soul = cardSoul(card);
  const triggers = cardTriggers(card);
  const name = cardName(card);
  const number = cardNumber(card);
  const wantsArt = !faceDown && number.includes("/");
  const emblem = isClimax ? "華" : (name.match(/[A-Za-z぀-ヿ一-龯]/)?.[0] ?? "✦");

  const className = cx(
    "card",
    `size-${size}`,
    faceDown ? "c-none" : `c-${color}`,
    isClimax && !faceDown && "is-climax",
    faceDown && "is-back",
    wantsArt && artOk && "has-art",
    selected && "selected",
    actionable && "actionable",
    rested && "rested",
    onClick && "is-button",
  );

  const inner = faceDown ? (
    <span className="card-back" aria-hidden>
      勝
    </span>
  ) : (
    <>
      <span className="card__ribbon" aria-hidden />
      <div className="card__head">
        {level != null ? (
          <span className="card__lvl" title={`Level ${level}`}>
            {Array.from({ length: Math.max(0, Math.min(3, level)) }, (_, i) => (
              <i key={i} />
            ))}
            {level > 3 ? <b>{level}</b> : level === 0 ? <b>0</b> : null}
          </span>
        ) : (
          <span />
        )}
        {cost != null ? <span className="card__cost">{cost}</span> : null}
      </div>

      <div className="card__art" aria-hidden>
        {emblem}
      </div>

      {triggers.length ? (
        <span className="card__trig" title={`Trigger: ${triggers.join(", ")}`}>
          {triggers[0]?.[0]?.toUpperCase() ?? "T"}
        </span>
      ) : null}

      <div className="card__name">{name}</div>

      <div className="card__foot">
        {isClimax ? (
          <span className="climax-tag">CLIMAX</span>
        ) : (
          <>
            <span className="card__pow">{power ?? "—"}</span>
            {soul != null ? (
              <span className="card__soul" title={`${soul} soul`}>
                {Array.from({ length: Math.max(0, Math.min(4, soul)) }, (_, i) => (
                  <i key={i} />
                ))}
              </span>
            ) : null}
          </>
        )}
      </div>

      <span className="card__no">{number}</span>

      {wantsArt ? (
        <img
          className="card__img"
          src={cardArtUrl(number)}
          alt=""
          loading="lazy"
          draggable={false}
          onLoad={() => setArtOk(true)}
          onError={() => setArtOk(false)}
        />
      ) : null}
    </>
  );

  if (onClick || draggable) {
    return (
      <button
        type="button"
        className={className}
        style={style}
        onClick={onClick}
        draggable={draggable}
        onDragStart={onDragStart}
        onDragEnd={onDragEnd}
        title={title ?? name}
        aria-pressed={selected}
        {...hoverProps}
      >
        {inner}
      </button>
    );
  }
  return (
    <div className={className} style={style} title={title ?? (faceDown ? "Face-down card" : name)} {...hoverProps}>
      {inner}
    </div>
  );
}
