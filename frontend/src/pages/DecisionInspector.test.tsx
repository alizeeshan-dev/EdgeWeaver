import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DecisionInspector } from "./DecisionInspector";

const bundle = {
  run: { run_id: "core-bursty-edgeweaver-seed-1", scheduler_name: "edgeweaver", random_seed: 1, request_count: 1, completed_requests: 1, rejected_requests: 0 },
  summary: {},
  trace: {},
  requests: [{
    request: { request_id: "request-00001", arrival_time_ms: 12440, deadline_ms: 150, minimum_accuracy: 0.91, input_size_bytes: 2244, source_device_id: "mobile", true_label: 1, true_label_name: "Walking", feature_vector_id: 428 },
    execution: {
      request_id: "request-00001", status: "completed", arrival_time_ms: 12440, absolute_deadline_ms: 12590,
      upload_time_ms: 0, queue_wait_ms: 18, inference_time_ms: 1.14, return_time_ms: 0, scheduler_overhead_ms: 0,
      completion_time_ms: 12459.14, end_to_end_latency_ms: 19.14, deadline_met: true, actual_prediction: 1,
      actual_prediction_name: "Walking", prediction_correct: true, rejection_reason: null,
      energy: { compute_energy_units: 1.1, network_energy_units: 0, estimated_total_energy_units: 1.1 },
      assignment: {
        request_id: "request-00001", device_id: "mobile", model_id: "logistic-regression-v1",
        predicted_completion_ms: 12459, predicted_energy_units: 1.1, predicted_model_accuracy: 0.912,
        expected_to_meet_deadline: true, decision_reason: "lowest estimated energy among feasible candidates",
        candidates: [{
          device_id: "mobile", model_id: "logistic-regression-v1", model_accuracy: 0.912, compatible: true,
          accuracy_eligible: true, network_available: true, eligible: true, rejection_reason: null, queue_admissible: true,
          predicted_upload_time_ms: 0, predicted_queue_wait_ms: 18, predicted_inference_time_ms: 1.05,
          predicted_return_time_ms: 0, scheduler_overhead_ms: 0, predicted_completion_ms: 12459,
          predicted_energy_units: 1.1, expected_to_meet_deadline: true,
        }, {
          device_id: "mobile", model_id: "mlp-v1", model_accuracy: 0.94, compatible: false,
          accuracy_eligible: true, network_available: true, eligible: false, rejection_reason: "incompatible_device_model", queue_admissible: true,
          predicted_upload_time_ms: 0, predicted_queue_wait_ms: 0, predicted_inference_time_ms: 0,
          predicted_return_time_ms: 0, scheduler_overhead_ms: 0, predicted_completion_ms: 12440,
          predicted_energy_units: 0, expected_to_meet_deadline: false,
        }],
      },
    },
    profile_update: null,
  }],
};

afterEach(() => vi.restoreAllMocks());

describe("DecisionInspector", () => {
  it("renders live request, candidate, decision, and outcome data", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      const value = url === "/api/simulations"
        ? [{ run_id: bundle.run.run_id, scenario_id: "bursty", scheduler_name: "edgeweaver", seed: 1, completed_requests: 1, deadline_satisfaction_rate: 1, source: "phase7" }]
        : bundle;
      return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } }));
    }));

    render(<DecisionInspector initialRunId={bundle.run.run_id} />);
    await waitFor(() => expect(screen.getByText("Candidate assignments")).toBeInTheDocument());
    expect(screen.getAllByText("Walking").length).toBeGreaterThan(0);
    expect(screen.getByText("mobile / light")).toBeInTheDocument();
    expect(screen.getByText(/Completed in/)).toBeInTheDocument();
    expect(screen.getByText(/Deadline met with/)).toBeInTheDocument();
    expect(screen.queryByText("mobile / heavy")).not.toBeInTheDocument();
    expect(screen.getByText("1 rejected candidates hidden")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: "Show rejected candidates" }));
    expect(screen.getByText("mobile / heavy")).toBeInTheDocument();
    expect(screen.getByText("unsupported on device")).toBeInTheDocument();
  });
});
