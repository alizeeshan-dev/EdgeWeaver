import { api } from "../api";
import { ErrorState, LoadingState } from "../components/States";
import { displayName, formatMs, formatPercent } from "../format";
import { useRemote } from "../hooks";
import type { FailureCase, RunPreset } from "../types";

type Evidence = Record<string, unknown> & {
  absolute_deadline_ms?: number;
  actual_completion_ms?: number;
  actual_inference_time_ms?: number;
  actual_network_time_ms?: number;
  actual_queue_wait_ms?: number;
  candidates?: Array<Record<string, unknown>>;
  decision_reason?: string;
  minimum_accuracy?: number;
  request_deadline_ms?: number;
  request_id?: string;
  run_id?: string;
  scenario_id?: RunPreset["scenario"];
  scheduler_name?: RunPreset["scheduler"];
  seed?: number;
  selected_device_id?: string;
  selected_model_id?: string;
};

export function FailureCases({
  onInspect,
  onReplay,
}: {
  onInspect: (runId: string, requestId?: string) => void;
  onReplay: (preset: RunPreset) => void;
}) {
  const remote = useRemote(api.failureCases);
  if (remote.loading) return <LoadingState label="Loading extracted evidence" />;
  if (remote.error || !remote.data) return <ErrorState message={remote.error ?? "Failure cases unavailable."} onRetry={remote.reload} />;

  const observed = remote.data.cases.filter((item) => item.observed);
  const absent = remote.data.cases.filter((item) => !item.observed);
  return (
    <section className="page failure-page">
      <div className="page-title-row">
        <div><h1>Failure Cases</h1><p>Real extracted evidence first. Cases that did not occur remain explicit—not manufactured.</p></div>
        <span className="evidence-count"><b>{observed.length}</b> / {remote.data.cases.length} observed</span>
      </div>

      <div className="evidence-intro"><p className="eyebrow">WHY THIS PAGE EXISTS</p><strong>A scheduler can make a locally sensible decision that fails end to end.</strong><span>Follow each case from policy decision to dominant cause, actual outcome, and an available alternative.</span></div>

      <div className="failure-list observed-list">
        {observed.map((item, index) => <ObservedCase item={item} index={index} key={item.case_id} onInspect={onInspect} onReplay={onReplay} />)}
      </div>

      <details className="absent-cases">
        <summary>{absent.length} planned case types were not produced by this experiment</summary>
        <p>Absence is a valid research result. Open these entries to see the extraction criteria that were not met.</p>
        <div className="failure-list">{absent.map((item, index) => <AbsentCase item={item} index={index + observed.length} key={item.case_id} />)}</div>
      </details>
    </section>
  );
}

function ObservedCase({ item, index, onInspect, onReplay }: { item: FailureCase; index: number; onInspect: (runId: string, requestId?: string) => void; onReplay: (preset: RunPreset) => void }) {
  const evidence = item.evidence as Evidence;
  const lateBy = Math.max((evidence.actual_completion_ms ?? 0) - (evidence.absolute_deadline_ms ?? 0), 0);
  const alternatives = (evidence.candidates ?? []).filter((candidate) => candidate.eligible && candidate.expected_to_meet_deadline);
  const alternative = alternatives.sort((a, b) => Number(a.predicted_completion_ms) - Number(b.predicted_completion_ms))[0];
  return <article className="failure-card observed story-card">
    <div className="case-number">0{index + 1}</div>
    <div className="case-content">
      <p className="eyebrow">OBSERVED EVIDENCE</p>
      <h2>{item.title}</h2>
      <div className="failure-story">
        <StoryStep index="1" label="Decision">{displayName(String(evidence.scheduler_name))} chose {displayName(String(evidence.selected_device_id))} / {displayName(String(evidence.selected_model_id))}: {String(evidence.decision_reason)}.</StoryStep>
        <StoryStep index="2" label="Dominant cause">Network transfer consumed {formatMs(evidence.actual_network_time_ms)} versus {formatMs(evidence.actual_inference_time_ms)} inference.</StoryStep>
        <StoryStep index="3" label="Outcome">The request missed its {formatMs(evidence.request_deadline_ms)} deadline by {formatMs(lateBy)}.</StoryStep>
        <StoryStep index="4" label="Available alternative">{alternative ? `${displayName(String(alternative.device_id))} / ${displayName(String(alternative.model_id))} was predicted deadline-feasible.` : "No deadline-feasible alternative was recorded."}</StoryStep>
      </div>
      <div className="evidence-grid">
        <span><small>RUN</small><strong>{String(evidence.run_id)}</strong></span>
        <span><small>REQUEST</small><strong>{String(evidence.request_id)}</strong></span>
        <span><small>MIN ACCURACY</small><strong>{formatPercent(evidence.minimum_accuracy ?? null)}</strong></span>
        <span><small>QUEUE</small><strong>{formatMs(evidence.actual_queue_wait_ms)}</strong></span>
      </div>
    </div>
    <div className="case-actions">
      <button className="primary-button" type="button" onClick={() => onReplay({ scenario: evidence.scenario_id!, scheduler: evidence.scheduler_name!, seed: evidence.seed!, guided: true })}>Replay exact trace</button>
      <button className="outline-button" type="button" onClick={() => onInspect(evidence.run_id!, evidence.request_id)}>Inspect request</button>
    </div>
  </article>;
}

function StoryStep({ index, label, children }: { index: string; label: string; children: React.ReactNode }) {
  return <div><i>{index}</i><p><strong>{label}</strong><span>{children}</span></p></div>;
}

function AbsentCase({ item, index }: { item: FailureCase; index: number }) {
  return <article className="failure-card absent"><div className="case-number">0{index + 1}</div><div className="case-content"><p className="eyebrow">NOT OBSERVED</p><h2>{item.title}</h2><p>{item.note}</p></div><span className="absent-mark">—</span></article>;
}
