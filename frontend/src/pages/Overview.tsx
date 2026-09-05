import { api } from "../api";
import { ErrorState, LoadingState } from "../components/States";
import { displayName, formatMs, formatPercent } from "../format";
import { useRemote } from "../hooks";
import type { PageId } from "../types";

export function Overview({
  onNavigate,
  onRecommendedDemo,
}: {
  onNavigate: (page: PageId) => void;
  onRecommendedDemo: () => void;
}) {
  const project = useRemote(api.project);
  const models = useRemote(api.models);
  const schedulers = useRemote(api.schedulers);

  if (project.loading || models.loading || schedulers.loading) return <LoadingState />;
  const error = project.error ?? models.error ?? schedulers.error;
  if (error || !project.data || !models.data || !schedulers.data) {
    return <ErrorState message={error ?? "Project metadata is unavailable."} />;
  }
  const projectData = project.data;
  const modelData = models.data;
  const schedulerData = schedulers.data;

  return (
    <section className="page overview-page">
      <header className="hero-panel">
        <div>
          <p className="eyebrow">LOCAL RESEARCH PROTOTYPE</p>
          <h1>{projectData.name}</h1>
          <p className="hero-subtitle">{projectData.subtitle}</p>
          <p className="research-question">{projectData.research_question}</p>
          <div className="hero-actions">
            <button className="primary-button" type="button" onClick={onRecommendedDemo}>
              Start guided demo
            </button>
            <button className="outline-button" type="button" onClick={() => onNavigate("run")}>Configure a run</button>
            <button className="outline-button" type="button" onClick={() => onNavigate("compare")}> 
              View saved study
            </button>
          </div>
          <p className="demo-hint">Recommended: replay the same Network Slowdown trace with Fastest Device, then EdgeWeaver.</p>
        </div>
        <div className="study-orbit" aria-label="Study summary">
          <span>80</span>
          <strong>core runs</strong>
          <small>+ 10 ablations</small>
        </div>
      </header>

      <div className="section-heading">
        <div>
          <p className="eyebrow">MEASURED ON ONE LOCAL COMPUTER</p>
          <h2>Model profiles</h2>
        </div>
        <p>Accuracy and latency are measured; device performance and energy remain simulated.</p>
      </div>
      <div className="three-column-grid">
        {modelData.map((model) => (
          <article className={`data-card model-card ${model.computational_role}`} key={model.model_id}>
            <div className="card-kicker">{model.computational_role.toUpperCase()} ROLE</div>
            <h3>{model.display_name}</h3>
            <p>{displayName(model.model_id)}</p>
            <dl className="metric-pairs">
              <div><dt>Accuracy</dt><dd>{formatPercent(model.accuracy, 2)}</dd></div>
              <div><dt>Macro F1</dt><dd>{formatPercent(model.macro_f1, 2)}</dd></div>
              <div><dt>Mean latency</dt><dd>{formatMs(model.local_latency_ms_mean, 3)}</dd></div>
              <div><dt>P95 latency</dt><dd>{formatMs(model.local_latency_ms_p95, 3)}</dd></div>
            </dl>
          </article>
        ))}
      </div>

      <div className="section-heading compact"><h2>Simulated edge system</h2></div>
      <div className="device-flow">
        {projectData.devices.map((device, index) => (
          <div className="flow-group" key={device.id}>
            <article className="data-card device-card">
              <div className="device-index">0{index + 1}</div>
              <h3>{displayName(device.id)}</h3>
              <p>{device.supported_models.join(" · ")}</p>
              <strong>{device.speed_multiplier}× speed factor</strong>
              <small>{device.queue_capacity} waiting slots · {device.active_power_units} power units</small>
            </article>
            {index < projectData.devices.length - 1 ? <span className="flow-line" /> : null}
          </div>
        ))}
      </div>

      <div className="section-heading compact"><h2>Scheduling policies</h2></div>
      <div className="scheduler-grid">
        {schedulerData.map((scheduler) => (
          <article className="scheduler-card" key={scheduler.id} style={{ "--accent": scheduler.accent } as React.CSSProperties}>
            <span>{scheduler.name}</span>
            <p>{scheduler.description}</p>
          </article>
        ))}
      </div>

      <div className="pipeline-card">
        <p className="eyebrow">REPRODUCIBLE PIPELINE</p>
        <div className="pipeline">
          {projectData.pipeline.map((stage, index) => (
            <div className="pipeline-stage" key={stage}>
              <span>{stage}</span>
              {index < projectData.pipeline.length - 1 ? <i>→</i> : null}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
