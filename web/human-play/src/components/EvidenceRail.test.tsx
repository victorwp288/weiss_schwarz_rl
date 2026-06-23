import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { sampleSession } from "../test/fixtures";
import { EvidenceRail } from "./EvidenceRail";

describe("EvidenceRail", () => {
  it("summarizes terminal result evidence", () => {
    render(
      <EvidenceRail
        health={{ ok: true }}
        state={{
          ...sampleSession,
          human_turn: false,
          terminal: true,
          result: { decision_count: 122, status: "complete", winner_seat: 1 },
        }}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText("122")).toBeInTheDocument();
    expect(screen.getByText("Result")).toBeInTheDocument();
    expect(screen.getByText("Winner seat 1")).toBeInTheDocument();
  });
});
