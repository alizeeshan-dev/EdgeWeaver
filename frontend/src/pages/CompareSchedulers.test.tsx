import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CompareSchedulers } from "./CompareSchedulers";

const row = (variant_id: string, metric: string, mean: number) => ({
  scenario_id: "network_slowdown",
  variant_id,
  scheduler_name: variant_id,
  metric,
  dimension: "",
  unit: "ratio",
  run_count: 5,
  mean,
  sample_std: 0,
  seed_values: [],
});

afterEach(() => vi.restoreAllMocks());

describe("CompareSchedulers", () => {
  it("starts with two distinct policies and reveals secondary metrics on demand", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify({
      rows: [
        row("fastest_device", "deadline_satisfaction_rate", 0.5),
        row("edgeweaver", "deadline_satisfaction_rate", 0.8),
        row("fastest_device", "mean_end_to_end_latency_ms", 50),
        row("edgeweaver", "mean_end_to_end_latency_ms", 40),
        row("fastest_device", "estimated_energy_per_completed_request_units", 2),
        row("edgeweaver", "estimated_energy_per_completed_request_units", 1),
        row("fastest_device", "useful_goodput_requests_per_second", 1),
        row("edgeweaver", "useful_goodput_requests_per_second", 2),
        row("fastest_device", "p95_end_to_end_latency_ms", 70),
        row("edgeweaver", "p95_end_to_end_latency_ms", 55),
        row("fastest_device", "actual_prediction_accuracy", 0.9),
        row("edgeweaver", "actual_prediction_accuracy", 0.92),
      ],
      findings_markdown: "",
      figures: [],
    }), { status: 200, headers: { "Content-Type": "application/json" } }))));

    render(<CompareSchedulers />);
    await waitFor(() => expect(screen.getByText("WHAT THE SAVED STUDY SHOWS")).toBeInTheDocument());
    const selects = screen.getAllByRole("combobox");
    expect(selects[1]).toHaveValue("fastest_device");
    expect(selects[2]).toHaveValue("edgeweaver");
    expect(screen.queryByText("Useful goodput")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show more details" }));
    expect(screen.getAllByText("Useful goodput").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Show key metrics" })).toBeInTheDocument();
  });
});
