import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { sampleSession } from "../test/fixtures";
import { Playfield } from "./Playfield";

describe("Playfield", () => {
  it("shows the stage, hand cards, and a turn indicator for the human", () => {
    render(<Playfield state={sampleSession} loading={false} />);

    expect(screen.getByText("Your turn")).toBeInTheDocument();
    expect(screen.getByText("Front row Yotsuba")).toBeInTheDocument();
    expect(screen.getByText("Opponent public card")).toBeInTheDocument();
    expect(screen.getByText("Yotsuba Nakano")).toBeInTheDocument();
    expect(screen.getByText("Choice Climax")).toBeInTheDocument();
    expect(screen.getByText(/Opponent hidden zones remain redacted/i)).toBeInTheDocument();
  });

  it("shows a finished indicator for terminal states", () => {
    render(<Playfield state={{ ...sampleSession, human_turn: false, terminal: true }} loading={false} />);

    expect(screen.getByText("Match over")).toBeInTheDocument();
    expect(screen.queryByText("Your turn")).not.toBeInTheDocument();
  });

  it("marks accepted mulligan toggles in the hand", () => {
    const state = {
      ...sampleSession,
      view: {
        ...sampleSession.view,
        summary: { ...sampleSession.view.summary, phase: "mulligan" },
        legal_actions: [
          {
            action_id: 11,
            label: "Toggle Yotsuba Nakano",
            family: "mulligan_select",
            source_refs: [{ card: { name: "Yotsuba Nakano", card_no: "5HY/W90-001" }, ref_id: "self.hand.0" }],
            params: { hand_index: 0 },
          },
        ],
      },
    };

    render(<Playfield state={state} loading={false} markedActionIds={new Set([11])} />);

    expect(screen.getByTitle("Yotsuba Nakano — Toggle Yotsuba Nakano")).toHaveAttribute("aria-pressed", "true");
  });
});
