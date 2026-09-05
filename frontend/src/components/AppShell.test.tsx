import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppShell } from "./AppShell";

describe("AppShell", () => {
  it("opens the glossary and dismisses it with Escape", () => {
    render(<AppShell activePage="overview" onNavigate={vi.fn()}><p>Page content</p></AppShell>);
    const trigger = screen.getByRole("button", { name: /Glossary/ });
    expect(trigger).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(trigger);
    expect(screen.getByRole("dialog", { name: "Research glossary" })).toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "true");

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Research glossary" })).not.toBeInTheDocument();
  });
});
