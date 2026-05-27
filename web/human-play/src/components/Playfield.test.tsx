import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { sampleSession } from "../test/fixtures";
import { Playfield } from "./Playfield";

describe("Playfield", () => {
  it("shows public stage, zone counts, and human-readable hand cards", () => {
    render(<Playfield state={sampleSession} loading={false} />);

    expect(screen.getByText("Your decision")).toBeInTheDocument();
    expect(screen.getByText("Front row Yotsuba")).toBeInTheDocument();
    expect(screen.getByText("Opponent public card")).toBeInTheDocument();
    expect(screen.getByText("Yotsuba Nakano")).toBeInTheDocument();
    expect(screen.getByText("Choice Climax")).toBeInTheDocument();
    expect(screen.getByText(/Opponent hidden zones remain redacted/i)).toBeInTheDocument();
  });

  it("shows a clear completed-game heading for terminal states", () => {
    render(<Playfield state={{ ...sampleSession, human_turn: false, terminal: true }} loading={false} />);

    expect(screen.getByText("Game complete")).toBeInTheDocument();
    expect(screen.getByText("Terminal")).toBeInTheDocument();
    expect(screen.queryByText("Model resolving")).not.toBeInTheDocument();
  });
});
