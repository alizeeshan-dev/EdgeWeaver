import { useEffect, useState } from "react";

import { AppShell } from "./components/AppShell";
import { CompareSchedulers } from "./pages/CompareSchedulers";
import { DecisionInspector } from "./pages/DecisionInspector";
import { FailureCases } from "./pages/FailureCases";
import { Overview } from "./pages/Overview";
import { RunSimulation } from "./pages/RunSimulation";
import type { PageId, RunPreset, Scenario, Scheduler } from "./types";

interface RouteState {
  inspectorRequestId?: string;
  inspectorRunId?: string;
  page: PageId;
  runPreset?: RunPreset;
}

const pages: PageId[] = ["overview", "run", "inspector", "compare", "failures"];
const scenarios: Scenario["scenario_id"][] = ["normal", "bursty", "network_slowdown", "device_slowdown"];
const schedulers: Scheduler["id"][] = ["round_robin", "fastest_device", "min_completion", "edgeweaver"];

export const routeFromHash = (): RouteState => {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?", 2);
  const page = pages.includes(path as PageId) ? path as PageId : "overview";
  const params = new URLSearchParams(query);
  if (page === "inspector") {
    return {
      page,
      inspectorRunId: params.get("run") || undefined,
      inspectorRequestId: params.get("request") || undefined,
    };
  }
  if (page === "run") {
    const scenario = params.get("scenario") as Scenario["scenario_id"];
    const scheduler = params.get("scheduler") as Scheduler["id"];
    const seed = Number(params.get("seed"));
    if (scenarios.includes(scenario) && schedulers.includes(scheduler) && Number.isInteger(seed) && seed >= 0) {
      return { page, runPreset: { scenario, scheduler, seed, guided: params.get("guided") === "1" } };
    }
  }
  return { page };
};

export default function App() {
  const [route, setRoute] = useState<RouteState>(routeFromHash);

  useEffect(() => {
    const onHashChange = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const setHashRoute = (hash: string, nextRoute: RouteState) => {
    window.location.hash = hash;
    setRoute(nextRoute);
  };

  const navigate = (page: PageId) => setHashRoute(`/${page}`, { page });

  const inspectRun = (runId: string, requestId?: string) => {
    const params = new URLSearchParams({ run: runId });
    if (requestId) params.set("request", requestId);
    setHashRoute(`/inspector?${params.toString()}`, {
      page: "inspector",
      inspectorRunId: runId,
      inspectorRequestId: requestId,
    });
  };

  const openRun = (preset?: RunPreset) => {
    if (!preset) {
      navigate("run");
      return;
    }
    const params = new URLSearchParams({
      scenario: preset.scenario,
      scheduler: preset.scheduler,
      seed: String(preset.seed),
    });
    if (preset.guided) params.set("guided", "1");
    setHashRoute(`/run?${params.toString()}`, { page: "run", runPreset: preset });
  };

  return (
    <AppShell activePage={route.page} onNavigate={navigate}>
      {route.page === "overview" ? (
        <Overview onNavigate={navigate} onRecommendedDemo={() => openRun({ scenario: "network_slowdown", scheduler: "fastest_device", seed: 1, guided: true })} />
      ) : null}
      {route.page === "run" ? <RunSimulation initialPreset={route.runPreset} onInspect={inspectRun} /> : null}
      {route.page === "inspector" ? (
        <DecisionInspector
          initialRequestId={route.inspectorRequestId}
          initialRunId={route.inspectorRunId}
        />
      ) : null}
      {route.page === "compare" ? <CompareSchedulers /> : null}
      {route.page === "failures" ? <FailureCases onInspect={inspectRun} onReplay={openRun} /> : null}
    </AppShell>
  );
}
