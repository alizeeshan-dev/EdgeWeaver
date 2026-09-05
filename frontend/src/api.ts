import type {
  ExperimentResults,
  EventPayload,
  FailureCasesPayload,
  ModelProfile,
  ProjectPayload,
  RunListItem,
  RunPayload,
  Scenario,
  Scheduler,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  project: () => request<ProjectPayload>("/api/project"),
  models: () => request<ModelProfile[]>("/api/models"),
  scenarios: () => request<Scenario[]>("/api/scenarios"),
  schedulers: () => request<Scheduler[]>("/api/schedulers"),
  runs: () => request<RunListItem[]>("/api/simulations"),
  run: (runId: string) => request<RunPayload>(`/api/simulations/${encodeURIComponent(runId)}`),
  events: (runId: string) =>
    request<EventPayload>(`/api/simulations/${encodeURIComponent(runId)}/events`),
  results: () => request<ExperimentResults>("/api/experiments/results"),
  failureCases: () => request<FailureCasesPayload>("/api/experiments/failure-cases"),
  simulate: (scenarioId: Scenario["scenario_id"], schedulerName: Scheduler["id"], seed: number) =>
    request<{ run_id: string; executed: boolean; summary: RunPayload["summary"] }>(
      "/api/simulations",
      {
        method: "POST",
        body: JSON.stringify({ scenario_id: scenarioId, scheduler_name: schedulerName, seed }),
      },
    ),
};
