import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { sampleSession } from "../test/fixtures";
import { ActionInspector } from "./ActionInspector";

describe("ActionInspector", () => {
  it("renders only simulator-provided legal moves and submits the selected action", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();

    render(<ActionInspector state={sampleSession} selectedActionId={null} busy={false} onSelectAction={onSelect} />);

    expect(screen.getByRole("button", { name: /Play Yotsuba/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /Pass/i })).toBeEnabled();
    expect(screen.queryByText(/Illegal/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Play Yotsuba/i }));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ action_id: 11 }));
  });

  it("disables moves while waiting for model decisions", () => {
    render(
      <ActionInspector
        state={{ ...sampleSession, human_turn: false }}
        selectedActionId={null}
        busy={false}
        onSelectAction={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: /Play Yotsuba/i })).toBeDisabled();
  });

  it("labels terminal sessions as complete instead of waiting for the model", () => {
    render(
      <ActionInspector
        state={{
          ...sampleSession,
          human_turn: false,
          terminal: true,
          view: { ...sampleSession.view, legal_actions: [] },
        }}
        selectedActionId={null}
        busy={false}
        onSelectAction={vi.fn()}
      />,
    );

    expect(screen.getByText("Match complete.")).toBeInTheDocument();
    expect(screen.getByText("No further actions. The match is complete.")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for the model.")).not.toBeInTheDocument();
  });
});
