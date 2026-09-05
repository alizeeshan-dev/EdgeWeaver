import { useEffect, useMemo, useState } from "react";

import { api } from "../api";
import { EmptyState, ErrorState, LoadingState } from "../components/States";
import { compactRequestId, displayName, formatBytes, formatMs, formatPercent, modelRole } from "../format";
import { useRemote } from "../hooks";
import type { Candidate, RequestBundle, RunPayload } from "../types";

export function DecisionInspector({
  initialRequestId,
  initialRunId,
}: {
  initialRequestId?: string;
  initialRunId?: string;
}) {
  const runs = useRemote(api.runs);
  const [runId, setRunId] = useState<string | null>(initialRunId ?? null);

  useEffect(() => {
    if (initialRunId) setRunId(initialRunId);
  }, [initialRunId]);

  useEffect(() => {
    if (runId || !runs.data?.length) return;
    const preferred = runs.data.find(
      (run) => run.run_id === "core-bursty-edgeweaver-seed-1",
    );
    setRunId((preferred ?? runs.data[0]).run_id);
  }, [runId, runs.data]);

  if (runs.loading || !runId) return <LoadingState label="Loading saved decisions" />;
  if (runs.error) return <ErrorState message={runs.error} onRetry={runs.reload} />;
  return (
    <InspectorRun
      initialRequestId={initialRequestId}
      key={`${runId}/${initialRequestId ?? "default"}`}
      runId={runId}
    />
  );
}

function InspectorRun({
  initialRequestId,
  runId,
}: {
  initialRequestId?: string;
  runId: string;
}) {
  const remote = useRemote(() => api.run(runId), [runId]);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [showRejected, setShowRejected] = useState(false);

  useEffect(() => {
    if (!remote.data?.requests.length) return;
    const requested = remote.data.requests.find(
      (item) => item.request.request_id === initialRequestId,
    );
    const failure = remote.data.requests.find((item) => !item.execution.deadline_met);
    setRequestId((requested ?? failure ?? remote.data.requests[0]).request.request_id);
  }, [initialRequestId, remote.data]);

  if (remote.loading) return <LoadingState label="Loading request candidates" />;
  if (remote.error || !remote.data) return <ErrorState message={remote.error ?? "Run unavailable."} onRetry={remote.reload} />;
  if (!remote.data.requests.length) return <EmptyState message="This run has no request decisions to inspect." />;
  const selected = remote.data.requests.find((item) => item.request.request_id === requestId) ?? remote.data.requests[0];

  return (
    <section className="inspector-page" data-node-id="4:19">
      <header className="inspector-header">
        <div>
          <h1 data-node-id="4:20">Decision Inspector</h1>
          <p data-node-id="4:21">Inspect every candidate the scheduler considered for one inference request.</p>
          <span className="run-context">{remote.data.run.run_id}</span>
        </div>
        <label className="request-picker" data-node-id="4:22">
          <span>Request</span>
          <select value={selected.request.request_id} onChange={(event) => setRequestId(event.target.value)}>
            {remote.data.requests.map((item) => (
              <option key={item.request.request_id} value={item.request.request_id}>
                {compactRequestId(item.request.request_id)}
              </option>
            ))}
          </select>
        </label>
      </header>

      <RequestDetails item={selected} />
      <div className="inspector-section-heading">
        <div><h2 className="inspector-section-title" data-node-id="4:37">Candidate assignments</h2><p>Eligible alternatives are shown first; rejected pairs remain available for audit.</p></div>
        <label className="rejected-toggle"><input checked={showRejected} type="checkbox" onChange={(event) => setShowRejected(event.target.checked)} /><span>Show rejected candidates</span></label>
      </div>
      <CandidateMatrix item={selected} showRejected={showRejected} />
      <DecisionCards item={selected} run={remote.data} />
      <h2 className="inspector-section-title timeline-title" data-node-id="4:104">Execution timeline</h2>
      <ExecutionTimeline item={selected} />
    </section>
  );
}

function RequestDetails({ item }: { item: RequestBundle }) {
  const fields = [
    ["ARRIVAL", `${(item.request.arrival_time_ms / 1000).toFixed(3)} s`, ""],
    ["DEADLINE", formatMs(item.request.deadline_ms, 0), "orange"],
    ["MIN ACCURACY", formatPercent(item.request.minimum_accuracy), "cyan"],
    ["INPUT SIZE", formatBytes(item.request.input_size_bytes), ""],
    ["TRUE LABEL", item.request.true_label_name, ""],
    ["FEATURE ID", String(item.request.feature_vector_id), ""],
  ];
  return (
    <div className="request-details" data-node-id="4:24">
      {fields.map(([label, value, color]) => (
        <div key={label}><span>{label}</span><strong className={color}>{value}</strong></div>
      ))}
    </div>
  );
}

function CandidateMatrix({ item, showRejected }: { item: RequestBundle; showRejected: boolean }) {
  const selected = item.execution.assignment;
  const visibleCandidates = selected.candidates.filter(
    (candidate) =>
      showRejected ||
      (candidate.eligible && candidate.queue_admissible) ||
      (candidate.device_id === selected.device_id && candidate.model_id === selected.model_id),
  );
  const hiddenCount = selected.candidates.length - visibleCandidates.length;
  return (
    <div className="candidate-matrix" data-node-id="4:38">
      <div className="candidate-row candidate-head">
        <span>PAIR</span><abbr title="Measured model test accuracy used for request eligibility.">ACC</abbr><abbr title="Predicted wait for active and queued work on this device.">QUEUE</abbr><abbr title="Predicted upload plus return delay; local mobile execution is zero.">NETWORK</abbr>
        <abbr title="Predicted device/model service duration.">INFERENCE</abbr><abbr title="Predicted end-to-end latency from request arrival.">COMPLETE</abbr><abbr title="Estimated normalized energy units; not joules.">ENERGY</abbr><abbr title="Eligible and predicted to finish before the relative deadline.">FEASIBLE</abbr>
      </div>
      <div className="candidate-scroll">
        {visibleCandidates.map((candidate) => (
          <CandidateRow
            candidate={candidate}
            arrival={item.request.arrival_time_ms}
            chosen={candidate.device_id === selected.device_id && candidate.model_id === selected.model_id}
            key={`${candidate.device_id}/${candidate.model_id}`}
          />
        ))}
      </div>
      {!showRejected && hiddenCount > 0 ? <div className="hidden-candidate-note">{hiddenCount} rejected candidates hidden</div> : null}
    </div>
  );
}

function CandidateRow({ candidate, arrival, chosen }: { candidate: Candidate; arrival: number; chosen: boolean }) {
  const feasible = candidate.eligible && candidate.queue_admissible && candidate.expected_to_meet_deadline;
  const completion = candidate.predicted_completion_ms - arrival;
  return (
    <div className={`candidate-row ${chosen ? "chosen" : ""} ${candidate.eligible ? "" : "excluded"}`} title={candidate.rejection_reason ?? undefined}>
      <strong className={modelRole(candidate.model_id)}><span>{candidate.device_id} / {modelRole(candidate.model_id)}</span>{candidate.rejection_reason ? <small>{reasonLabel(candidate.rejection_reason)}</small> : chosen ? <small>chosen assignment</small> : null}</strong>
      <span>{formatPercent(candidate.model_accuracy)}</span>
      <span>{formatMs(candidate.predicted_queue_wait_ms)}</span>
      <span>{formatMs(candidate.predicted_upload_time_ms + candidate.predicted_return_time_ms)}</span>
      <span>{formatMs(candidate.predicted_inference_time_ms)}</span>
      <span>{formatMs(completion)}</span>
      <span>{candidate.predicted_energy_units.toFixed(3)} NU</span>
      <b className={feasible ? "yes" : "no"}>{feasible ? "YES" : "NO"}</b>
    </div>
  );
}

function reasonLabel(reason: string) {
  const labels: Record<string, string> = {
    incompatible_device_model: "unsupported on device",
    model_accuracy_below_request_minimum: "below minimum accuracy",
    network_link_unavailable: "network unavailable",
    queue_capacity_unavailable: "queue cannot admit request",
  };
  return labels[reason] ?? displayName(reason);
}

function DecisionCards({ item, run }: { item: RequestBundle; run: RunPayload }) {
  const { assignment } = item.execution;
  const predictedLatency = assignment.predicted_completion_ms - item.request.arrival_time_ms;
  const slack = item.execution.completion_time_ms == null ? null : item.execution.absolute_deadline_ms - item.execution.completion_time_ms;
  const update = item.profile_update?.details;
  return (
    <div className="decision-grid">
      <article className="chosen-decision" data-node-id="4:92">
        <p className="card-kicker cyan">CHOSEN ASSIGNMENT</p>
        <h3>{assignment.device_id} <i>/</i> {modelRole(assignment.model_id)} model</h3>
        <p>{assignment.decision_reason}.</p>
        <strong>Expected: {formatMs(predictedLatency)} <i>•</i> {assignment.predicted_energy_units.toFixed(3)} NU <i>•</i> {assignment.expected_to_meet_deadline ? "feasible" : "late"}</strong>
      </article>
      <article className="actual-outcome" data-node-id="4:98">
        <p className="card-kicker">ACTUAL EXECUTION</p>
        <h3 className={item.execution.deadline_met ? "success" : "failure"}>
          {item.execution.status === "completed" ? `Completed in ${formatMs(item.execution.end_to_end_latency_ms)}` : "Request rejected"}
        </h3>
        <p><b>Prediction:</b> {item.execution.actual_prediction_name} {item.execution.prediction_correct ? "✓ correct" : "✕ incorrect"}</p>
        <strong className={item.execution.deadline_met ? "cyan" : "failure"}>
          {item.execution.deadline_met ? `Deadline met with ${formatMs(slack)} slack` : `Deadline missed by ${formatMs(slack == null ? null : Math.abs(slack))}`}
        </strong>
        <small className="magenta">
          {update?.new_estimate_ms != null
            ? `Observed inference ${formatMs(update.observed_inference_time_ms)} → EWMA ${formatMs(update.new_estimate_ms)}`
            : run.run.scheduler_name === "edgeweaver"
              ? `Observed inference ${formatMs(item.execution.inference_time_ms)} · estimate unchanged`
              : "Static baseline profile · no online update"}
        </small>
      </article>
    </div>
  );
}

function ExecutionTimeline({ item }: { item: RequestBundle }) {
  const execution = item.execution;
  const components = useMemo(
    () => [
      { label: "UPLOAD", value: execution.upload_time_ms, className: "upload" },
      { label: "QUEUE", value: execution.queue_wait_ms, className: "queue" },
      { label: "INFER", value: execution.inference_time_ms, className: "infer" },
      { label: "RETURN", value: execution.return_time_ms, className: "return" },
    ],
    [execution],
  );
  const total = execution.end_to_end_latency_ms ?? components.reduce((sum, part) => sum + part.value, 0);
  const predicted = execution.assignment.predicted_completion_ms - item.request.arrival_time_ms;
  const scale = Math.max(item.request.deadline_ms, total, predicted, 1);
  return (
    <div className="execution-timeline" data-node-id="4:105">
      <div className="timeline-track">
        <span className="arrival-segment" />
        {components.map((part) => (
          <span
            className={`timeline-segment ${part.className}`}
            key={part.label}
            style={{ width: `${Math.max((part.value / scale) * 100, part.value > 0 ? 0.6 : 0)}%` }}
          />
        ))}
        <span className="done-marker" style={{ left: `${Math.min((total / scale) * 100, 99)}%` }} />
      </div>
      <div className="timeline-labels">
        <span className="cyan">ARRIVE</span>
        {components.map((part) => <span className={part.className} key={part.label}>{part.label}</span>)}
        <span className="lime">DONE</span>
      </div>
      <div className="timeline-legend">
        <span className="cyan">Predicted {formatMs(predicted)}</span>
        <span className="lime">Actual {formatMs(total)}</span>
        <span className="orange">Deadline {formatMs(item.request.deadline_ms)}</span>
      </div>
    </div>
  );
}
