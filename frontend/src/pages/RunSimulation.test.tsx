import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RunSimulation } from "./RunSimulation";

const summary = {
  run_id: "interactive-network_slowdown-fastest_device-seed-1",
  scenario_id: "network_slowdown",
  scheduler_name: "fastest_device",
  seed: 1,
  simulation_duration_ms: 20_000,
  total_requests: 2,
  completed_requests: 2,
  rejected_requests: 0,
  deadline_satisfaction_rate: 0.5,
  useful_goodput_requests_per_second: 0.05,
  mean_end_to_end_latency_ms: 42,
  p95_end_to_end_latency_ms: 60,
  actual_prediction_accuracy: 1,
  estimated_energy_per_completed_request_units: 1.25,
  device_utilization: [{ device_id: "edge-server", utilization: 0.1 }],
  model_selection_distribution: [{ model_id: "mlp-v1", model_role: "heavy", assigned_requests: 2, percentage: 100 }],
};

afterEach(() => vi.restoreAllMocks());

describe("RunSimulation", () => {
  it("runs the guided preset and exposes same-trace replay and event playback", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      let body: unknown;
      if (url === "/api/scenarios") body = [{ scenario_id: "network_slowdown", description: "The edge-server link degrades mid-run.", simulation_duration_ms: 20_000, arrival: { base_rate_per_second: 1, bursts: [] } }];
      else if (url === "/api/schedulers") body = [
        { id: "fastest_device", name: "Fastest Device", description: "Ignores congestion and selects isolated compute speed.", accent: "red" },
        { id: "edgeweaver", name: "EdgeWeaver", description: "Balances feasibility and normalized energy.", accent: "lime" },
      ];
      else if (url === "/api/simulations" && init?.method === "POST") body = { run_id: summary.run_id, executed: true, summary };
      else if (url.endsWith("/events")) body = { format_version: "1.0", events: [{ sequence: 1, timestamp_ms: 0, event_type: "REQUEST_ARRIVED", request_id: "request-00001", device_id: null, model_id: null, details: {} }] };
      else body = { run: { run_id: summary.run_id, scheduler_name: "fastest_device", random_seed: 1, request_count: 2, completed_requests: 2, rejected_requests: 0 }, summary, trace: { trace_id: "network_slowdown-seed-1" }, requests: [] };
      return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } }));
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<RunSimulation initialPreset={{ scenario: "network_slowdown", scheduler: "fastest_device", seed: 1, guided: true }} onInspect={vi.fn()} />);
    await waitFor(() => expect(screen.getByText(/GUIDED DEMO/)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Run simulation" }));

    await waitFor(() => expect(screen.getByText(summary.run_id)).toBeInTheDocument());
    expect(screen.getByText(/Paired trace: network_slowdown-seed-1/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Compare with EdgeWeaver" })).toBeInTheDocument();
    expect(screen.getByText("Execution playback")).toBeInTheDocument();
    expect(screen.getByText(/virtual SimPy time/)).toBeInTheDocument();

    const post = fetchMock.mock.calls.find((call) => String(call[0]) === "/api/simulations" && call[1]?.method === "POST");
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({ scenario_id: "network_slowdown", scheduler_name: "fastest_device", seed: 1 });
  });
});
