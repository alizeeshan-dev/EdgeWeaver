import { useMemo, useState } from "react";

import { api } from "../api";
import { ErrorState, LoadingState } from "../components/States";
import { displayName } from "../format";
import { useRemote } from "../hooks";
import type { AggregateRow, Scheduler } from "../types";

const metrics = [
  { id: "deadline_satisfaction_rate", name: "Deadline satisfaction", multiplier: 100, unit: "%", primary: true, help: "Share of completed requests returned by their relative deadline." },
  { id: "mean_end_to_end_latency_ms", name: "Mean latency", multiplier: 1, unit: " ms", primary: true, help: "Mean upload, queue, inference, return, and scheduler-overhead latency." },
  { id: "estimated_energy_per_completed_request_units", name: "Estimated energy", multiplier: 1, unit: " NU", primary: true, help: "Normalized estimated compute and network energy per completion; not joules." },
  { id: "useful_goodput_requests_per_second", name: "Useful goodput", multiplier: 1, unit: " req/s", primary: false, help: "Correct, on-time, eligible completions per simulated second." },
  { id: "p95_end_to_end_latency_ms", name: "P95 latency", multiplier: 1, unit: " ms", primary: false, help: "95% of completed requests are at or below this latency." },
  { id: "actual_prediction_accuracy", name: "Actual accuracy", multiplier: 100, unit: "%", primary: false, help: "Correct predictions divided by completed requests." },
];

const policyOrder: Scheduler["id"][] = ["round_robin", "fastest_device", "min_completion", "edgeweaver"];
const colors: Record<Scheduler["id"], string> = {
  round_robin: "#8b5cff",
  fastest_device: "#ff5470",
  min_completion: "#00f5ff",
  edgeweaver: "#b7ff4a",
};

export function CompareSchedulers() {
  const remote = useRemote(api.results);
  const [scenario, setScenario] = useState("network_slowdown");
  const [policyA, setPolicyA] = useState<Scheduler["id"]>("fastest_device");
  const [policyB, setPolicyB] = useState<Scheduler["id"]>("edgeweaver");
  const [showDetails, setShowDetails] = useState(false);

  const rows = useMemo(() => {
    if (!remote.data) return [];
    return remote.data.rows.filter(
      (row) => row.scenario_id === scenario && policyOrder.includes(row.variant_id as Scheduler["id"]) && !row.dimension,
    );
  }, [remote.data, scenario]);

  if (remote.loading) return <LoadingState label="Loading aggregate study" />;
  if (remote.error || !remote.data) return <ErrorState message={remote.error ?? "Results unavailable."} onRetry={remote.reload} />;

  const visibleMetrics = showDetails ? metrics : metrics.filter((metric) => metric.primary);
  const selectedPolicies = policyA === policyB ? [policyA] : [policyA, policyB];

  return (
    <section className="page compare-page">
      <div className="page-title-row compare-heading">
        <div><h1>Compare Schedulers</h1><p>Choose two policies. Every value is a five-seed mean over paired request traces.</p></div>
        <label className="compact-select">Scenario<select value={scenario} onChange={(event) => setScenario(event.target.value)}><option value="normal">Normal</option><option value="bursty">Bursty</option><option value="network_slowdown">Network Slowdown</option><option value="device_slowdown">Device Slowdown</option></select></label>
      </div>

      <div className="policy-picker">
        <PolicySelect label="Policy A" value={policyA} exclude={policyB} onChange={setPolicyA} />
        <span>VERSUS</span>
        <PolicySelect label="Policy B" value={policyB} exclude={policyA} onChange={setPolicyB} />
        <button className="outline-button" type="button" onClick={() => setShowDetails((value) => !value)}>{showDetails ? "Show key metrics" : "Show more details"}</button>
      </div>

      <ComparisonTakeaway rows={rows} policyA={policyA} policyB={policyB} scenario={scenario} />

      <div className="comparison-grid primary-comparison">
        {visibleMetrics.map((metric) => (
          <MetricComparison key={metric.id} metric={metric} policies={selectedPolicies} rows={rows.filter((row) => row.metric === metric.id)} />
        ))}
      </div>

      <article className="comparison-table-card">
        <div><p className="eyebrow">DIRECT SIDE-BY-SIDE</p><h2>{displayName(scenario)}</h2></div>
        <table>
          <thead><tr><th>Policy</th>{visibleMetrics.map((metric) => <th title={metric.help} key={metric.id}>{metric.name}</th>)}</tr></thead>
          <tbody>{selectedPolicies.map((policy) => <tr key={policy}><th><i style={{ background: colors[policy] }} />{displayName(policy)}</th>{visibleMetrics.map((metric) => { const row = rows.find((item) => item.variant_id === policy && item.metric === metric.id); return <td key={metric.id}>{row ? `${(row.mean * metric.multiplier).toFixed(metric.multiplier === 100 ? 2 : 3)}${metric.unit}` : "—"}</td>; })}</tr>)}</tbody>
        </table>
      </article>

      <details className="figure-library">
        <summary>Open all eight saved research figures</summary>
        <div className="figure-strip">
          {remote.data.figures.map((figure) => (
            <a href={`/api/experiments/figures/${figure}`} target="_blank" rel="noreferrer" key={figure}>
              <img src={`/api/experiments/figures/${figure}`} alt={displayName(figure.replace(".png", ""))} />
              <span>{displayName(figure.replace(".png", ""))}</span>
            </a>
          ))}
        </div>
      </details>
    </section>
  );
}

function PolicySelect({ label, value, exclude, onChange }: { label: string; value: Scheduler["id"]; exclude: Scheduler["id"]; onChange: (value: Scheduler["id"]) => void }) {
  return <label>{label}<select value={value} onChange={(event) => onChange(event.target.value as Scheduler["id"])}>{policyOrder.map((policy) => <option disabled={policy === exclude} key={policy} value={policy}>{displayName(policy)}</option>)}</select></label>;
}

function ComparisonTakeaway({ rows, policyA, policyB, scenario }: { rows: AggregateRow[]; policyA: Scheduler["id"]; policyB: Scheduler["id"]; scenario: string }) {
  const value = (policy: string, metric: string) => rows.find((row) => row.variant_id === policy && row.metric === metric)?.mean;
  const deadlineA = value(policyA, "deadline_satisfaction_rate");
  const deadlineB = value(policyB, "deadline_satisfaction_rate");
  const latencyA = value(policyA, "mean_end_to_end_latency_ms");
  const latencyB = value(policyB, "mean_end_to_end_latency_ms");
  if ([deadlineA, deadlineB, latencyA, latencyB].some((item) => item == null)) return null;
  const deadlineGap = ((deadlineB ?? 0) - (deadlineA ?? 0)) * 100;
  const latencyGap = (latencyB ?? 0) - (latencyA ?? 0);
  return <div className="takeaway-card comparison-summary"><span>WHAT THE SAVED STUDY SHOWS</span><p>On <strong>{displayName(scenario)}</strong>, {displayName(policyB)} records {deltaWords(deadlineGap, "percentage points", "deadline satisfaction")} and {deltaWords(latencyGap, "ms", "mean latency")} compared with {displayName(policyA)}. These are paired-trace descriptive results, not a claim of statistical significance.</p></div>;
}

function deltaWords(value: number, unit: string, metric: string) {
  if (Math.abs(value) < 0.0005) return `the same ${metric}`;
  return `${Math.abs(value).toFixed(2)} ${unit} ${value > 0 ? "higher" : "lower"} ${metric}`;
}

function MetricComparison({ metric, rows, policies }: { metric: typeof metrics[number]; rows: AggregateRow[]; policies: Scheduler["id"][] }) {
  const maximum = Math.max(...rows.filter((row) => policies.includes(row.variant_id as Scheduler["id"])).map((row) => row.mean * metric.multiplier), 1e-9);
  return (
    <article className="comparison-card" title={metric.help}>
      <h3>{metric.name} <abbr title={metric.help}>?</abbr></h3>
      {policies.map((policy) => {
        const row = rows.find((item) => item.variant_id === policy);
        const value = row ? row.mean * metric.multiplier : 0;
        return <div className="comparison-bar" key={policy}><span>{displayName(policy)}</span><i><b style={{ width: `${(value / maximum) * 100}%`, background: colors[policy] }} /></i><strong>{value.toFixed(metric.multiplier === 100 ? 1 : value < 1 ? 3 : 1)}{metric.unit}</strong></div>;
      })}
    </article>
  );
}
