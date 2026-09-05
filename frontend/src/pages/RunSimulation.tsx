import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { ErrorState, LoadingState } from "../components/States";
import { compactRequestId, displayName, formatMs, formatPercent } from "../format";
import { useRemote } from "../hooks";
import type { RunMetrics, RunPreset, Scenario, Scheduler, SimulationEvent } from "../types";

interface CompletedRun {
  events: SimulationEvent[];
  runId: string;
  scenario: Scenario["scenario_id"];
  scheduler: Scheduler["id"];
  seed: number;
  summary: RunMetrics;
  traceId: string;
}

const progressStages = [
  "Preparing the paired request trace",
  "Scheduling and executing requests",
  "Calculating metrics and events",
];

export function RunSimulation({
  initialPreset,
  onInspect,
}: {
  initialPreset?: RunPreset;
  onInspect: (runId: string) => void;
}) {
  const scenarios = useRemote(api.scenarios);
  const schedulers = useRemote(api.schedulers);
  const [scenario, setScenario] = useState<Scenario["scenario_id"]>(initialPreset?.scenario ?? "bursty");
  const [scheduler, setScheduler] = useState<Scheduler["id"]>(initialPreset?.scheduler ?? "edgeweaver");
  const [seed, setSeed] = useState(initialPreset?.seed ?? 1);
  const [completedRun, setCompletedRun] = useState<CompletedRun | null>(null);
  const [comparisonRun, setComparisonRun] = useState<CompletedRun | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [runError, setRunError] = useState<string | null>(null);

  useEffect(() => {
    if (!initialPreset) return;
    setScenario(initialPreset.scenario);
    setScheduler(initialPreset.scheduler);
    setSeed(initialPreset.seed);
  }, [initialPreset]);

  if (scenarios.loading || schedulers.loading) return <LoadingState />;
  if (scenarios.error || schedulers.error || !scenarios.data || !schedulers.data) {
    return <ErrorState message={scenarios.error ?? schedulers.error ?? "Controls unavailable."} />;
  }

  const selectedScenario = scenarios.data.find((item) => item.scenario_id === scenario)!;
  const selectedScheduler = schedulers.data.find((item) => item.id === scheduler)!;

  const run = async (schedulerOverride?: Scheduler["id"]) => {
    const nextScheduler = schedulerOverride ?? scheduler;
    setScheduler(nextScheduler);
    setRunning(true);
    setProgress(0);
    setRunError(null);
    const timers = [
      window.setTimeout(() => setProgress(1), 350),
      window.setTimeout(() => setProgress(2), 900),
    ];
    try {
      const response = await api.simulate(scenario, nextScheduler, seed);
      const [details, eventPayload] = await Promise.all([
        api.run(response.run_id),
        api.events(response.run_id),
      ]);
      const nextRun: CompletedRun = {
        events: eventPayload.events,
        runId: response.run_id,
        scenario,
        scheduler: nextScheduler,
        seed,
        summary: response.summary,
        traceId: details.trace.trace_id ?? `${scenario}-seed-${seed}`,
      };
      setComparisonRun(
        completedRun &&
          completedRun.scenario === scenario &&
          completedRun.seed === seed &&
          completedRun.scheduler !== nextScheduler
          ? completedRun
          : null,
      );
      setCompletedRun(nextRun);
    } catch (reason) {
      setRunError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      timers.forEach(window.clearTimeout);
      setRunning(false);
    }
  };

  const replayScheduler: Scheduler["id"] =
    (completedRun?.scheduler ?? scheduler) === "edgeweaver" ? "fastest_device" : "edgeweaver";
  const replayName = schedulers.data.find((item) => item.id === replayScheduler)!.name;

  return (
    <section className="page run-page">
      <div className="page-title-row">
        <div><h1>Run One Simulation</h1><p>Replay one controlled scenario through the research core.</p></div>
        <span className="live-pill"><i /> LOCAL ENGINE</span>
      </div>

      {initialPreset?.guided ? (
        <div className="guided-banner">
          <div><p className="eyebrow">GUIDED DEMO · STEP 1 OF 2</p><strong>See what happens when “fastest” ignores a degraded network.</strong><span>Run Fastest Device first, then replay this exact trace with EdgeWeaver.</span></div>
          <span className="trace-lock">{displayName(scenario).toUpperCase()} · SEED {seed}</span>
        </div>
      ) : null}

      <div className="run-layout">
        <aside className="control-panel">
          <p className="eyebrow">RUN CONFIGURATION</p>
          <label>Scenario
            <select value={scenario} onChange={(event) => setScenario(event.target.value as Scenario["scenario_id"])}>
              {scenarios.data.map((item) => <option key={item.scenario_id} value={item.scenario_id}>{displayName(item.scenario_id)}</option>)}
            </select>
          </label>
          <p className="field-help">{selectedScenario.description}</p>
          <label>Scheduler
            <select value={scheduler} onChange={(event) => setScheduler(event.target.value as Scheduler["id"])}>
              {schedulers.data.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
            </select>
          </label>
          <p className="field-help">{selectedScheduler.description}</p>
          <label>Random seed
            <input min="0" type="number" value={seed} onChange={(event) => setSeed(Math.max(0, Number(event.target.value)))} />
          </label>
          <p className="field-help">The same scenario and seed always produce the same request trace, independent of scheduler.</p>
          <div className="duration-readout"><span>SIMULATED DURATION</span><strong>{selectedScenario.simulation_duration_ms / 1000} seconds</strong></div>
          <button className="run-button" type="button" disabled={running} onClick={() => void run()}>
            {running ? "Simulation running…" : "Run simulation"}
          </button>
          {running ? <RunProgress active={progress} /> : null}
          {runError ? <p className="inline-error">{runError}</p> : null}
        </aside>

        <div className="run-results">
          {!completedRun ? (
            <div className="empty-run-state">
              <div className="pulse-core" />
              <h2>Ready to simulate</h2>
              <p>Choose one scenario, policy, and seed. Reusing a seed keeps the workload identical for a fair scheduler comparison.</p>
            </div>
          ) : (
            <>
              <div className="result-heading">
                <div><p className="eyebrow">RUN COMPLETE</p><h2>{completedRun.runId}</h2><span className="trace-chip">⌁ Paired trace: {completedRun.traceId}</span></div>
                <button className="outline-button" type="button" onClick={() => onInspect(completedRun.runId)}>Inspect decisions</button>
              </div>
              <div className="metric-grid">
                <Metric help="Completed requests that met their relative deadline." label="Deadline satisfaction" value={formatPercent(completedRun.summary.deadline_satisfaction_rate)} accent="cyan" />
                <Metric help="Correct, on-time, accuracy-eligible completions per simulated second." label="Useful goodput" value={`${completedRun.summary.useful_goodput_requests_per_second.toFixed(2)} /s`} accent="lime" />
                <Metric help="Mean completed-request end-to-end latency." label="Mean latency" value={formatMs(completedRun.summary.mean_end_to_end_latency_ms)} accent="violet" />
                <Metric help="95% of completed requests are at or below this latency." label="P95 latency" value={formatMs(completedRun.summary.p95_end_to_end_latency_ms)} accent="orange" />
                <Metric help="Actual prediction correctness on sampled held-out UCI HAR requests." label="Actual accuracy" value={formatPercent(completedRun.summary.actual_prediction_accuracy)} accent="magenta" />
                <Metric help="Estimated normalized energy units per completed request; not joules." label="Est. energy / request" value={`${completedRun.summary.estimated_energy_per_completed_request_units.toFixed(3)} NU`} accent="red" />
              </div>

              <RunTakeaway current={completedRun} previous={comparisonRun} />

              <div className="next-actions">
                <div><strong>Keep the trace fixed</strong><span>Replay {completedRun.traceId} with {replayName} so only the policy changes.</span></div>
                <button className="primary-button" type="button" disabled={running} onClick={() => void run(replayScheduler)}>Compare with {replayName}</button>
              </div>

              <div className="result-panels">
                <article className="data-card"><h3>Device utilization</h3>{completedRun.summary.device_utilization.map((item) => <Bar key={item.device_id} label={displayName(item.device_id)} value={item.utilization * 100} />)}</article>
                <article className="data-card"><h3>Model selection</h3>{completedRun.summary.model_selection_distribution.map((item) => <Bar key={item.model_id} label={displayName(item.model_role)} value={item.percentage} />)}</article>
              </div>
              <EventPlayback events={completedRun.events} />
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function RunProgress({ active }: { active: number }) {
  return <div className="run-progress" role="status">{progressStages.map((stage, index) => <span className={index <= active ? "active" : ""} key={stage}><i>{index < active ? "✓" : index + 1}</i>{stage}</span>)}</div>;
}

function Metric({ label, value, accent, help }: { label: string; value: string; accent: string; help: string }) {
  return <article className={`metric-card ${accent}`} title={help}><span>{label} <abbr aria-label={`${label} definition`} title={help}>?</abbr></span><strong>{value}</strong></article>;
}

function RunTakeaway({ current, previous }: { current: CompletedRun; previous: CompletedRun | null }) {
  if (!previous) {
    const misses = current.summary.total_requests - Math.round(current.summary.total_requests * current.summary.deadline_satisfaction_rate);
    return <div className="takeaway-card"><span>WHAT TO NOTICE</span><p><strong>{displayName(current.scheduler)}</strong> completed this fixed trace with {misses === 0 ? "no deadline misses" : `${misses} deadline misses`} and {current.summary.estimated_energy_per_completed_request_units.toFixed(3)} normalized energy units per request.</p></div>;
  }
  const deadlineDelta = (current.summary.deadline_satisfaction_rate - previous.summary.deadline_satisfaction_rate) * 100;
  const latencyDelta = current.summary.mean_end_to_end_latency_ms - previous.summary.mean_end_to_end_latency_ms;
  return <div className="takeaway-card comparison"><span>SAME-TRACE COMPARISON</span><p><strong>{displayName(current.scheduler)}</strong> versus {displayName(previous.scheduler)}: {formatDelta(deadlineDelta, "percentage points deadline satisfaction")} and {formatDelta(latencyDelta, "ms mean latency")}. Inputs are identical; only the scheduler changed.</p></div>;
}

function formatDelta(value: number, label: string) {
  if (Math.abs(value) < 0.0005) return `no change in ${label}`;
  return `${value > 0 ? "+" : ""}${value.toFixed(2)} ${label}`;
}

function EventPlayback({ events }: { events: SimulationEvent[] }) {
  const [cursor, setCursor] = useState(Math.min(8, events.length));
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    setCursor(Math.min(8, events.length));
    setPlaying(false);
  }, [events]);
  useEffect(() => {
    if (!playing) return undefined;
    const timer = window.setInterval(() => {
      setCursor((current) => {
        if (current >= events.length) {
          setPlaying(false);
          return current;
        }
        return Math.min(current + 4, events.length);
      });
    }, 260);
    return () => window.clearInterval(timer);
  }, [events.length, playing]);
  const visible = useMemo(() => events.slice(Math.max(0, cursor - 5), cursor), [cursor, events]);
  return <article className="event-player data-card">
    <div className="event-player-heading"><div><p className="eyebrow">STRUCTURED EVENT LOG</p><h3>Execution playback</h3></div><button className="outline-button" type="button" onClick={() => { if (cursor >= events.length) setCursor(1); setPlaying((value) => !value); }}>{playing ? "Pause" : cursor >= events.length ? "Replay" : "Play"}</button></div>
    <input aria-label="Event playback position" max={Math.max(events.length, 1)} min="1" type="range" value={Math.max(cursor, 1)} onChange={(event) => { setPlaying(false); setCursor(Number(event.target.value)); }} />
    <div className="event-stream">{visible.map((event) => <div key={event.sequence}><time>{formatMs(event.timestamp_ms, 1)}</time><strong>{displayName(event.event_type)}</strong><span>{event.request_id ? compactRequestId(event.request_id) : "system"}</span><em>{event.device_id ? displayName(event.device_id) : "—"}</em></div>)}</div>
    <small>{cursor} of {events.length} events · virtual SimPy time, not wall-clock execution</small>
  </article>;
}

function Bar({ label, value }: { label: string; value: number }) {
  return <div className="bar-row"><div><span>{label}</span><strong>{value.toFixed(value < 1 ? 3 : 1)}%</strong></div><i><b style={{ width: `${Math.max(value, 0.3)}%` }} /></i></div>;
}
