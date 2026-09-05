import { useEffect, useState } from "react";

import type { PageId } from "../types";

const navigation: Array<{ id: PageId; label: string; index: string }> = [
  { id: "overview", label: "Project Overview", index: "01" },
  { id: "run", label: "Run Simulation", index: "02" },
  { id: "inspector", label: "Decision Inspector", index: "03" },
  { id: "compare", label: "Compare Schedulers", index: "04" },
  { id: "failures", label: "Failure Cases", index: "05" },
];

interface AppShellProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  children: React.ReactNode;
}

export function AppShell({ activePage, onNavigate, children }: AppShellProps) {
  const [glossaryOpen, setGlossaryOpen] = useState(false);

  useEffect(() => {
    if (!glossaryOpen) return undefined;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setGlossaryOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [glossaryOpen]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" type="button" onClick={() => onNavigate("overview")}>
          <span className="brand-mark">EW</span>
          <span>
            <strong>EdgeWeaver</strong>
            <small>Research console</small>
          </span>
        </button>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              className={activePage === item.id ? "nav-item active" : "nav-item"}
              key={item.id}
              onClick={() => onNavigate(item.id)}
              type="button"
            >
              <span>{item.index}</span>
              {item.label}
            </button>
          ))}
        </nav>
        <button aria-controls="research-glossary" aria-expanded={glossaryOpen} className="glossary-button" type="button" onClick={() => setGlossaryOpen(true)}>
          <span>?</span> Glossary
        </button>
        <div className="sidebar-status">
          <span className="status-dot" />
          <span>
            <strong>Local research core</strong>
            <small>90 saved runs available</small>
          </span>
        </div>
      </aside>
      <main className="app-main">{children}</main>
      {glossaryOpen ? (
        <div className="glossary-backdrop" role="presentation" onMouseDown={() => setGlossaryOpen(false)}>
          <aside className="glossary-drawer" id="research-glossary" aria-label="Research glossary" aria-modal="true" role="dialog" onMouseDown={(event) => event.stopPropagation()}>
            <div className="drawer-heading"><div><p className="eyebrow">QUICK REFERENCE</p><h2>Research glossary</h2></div><button aria-label="Close glossary" autoFocus type="button" onClick={() => setGlossaryOpen(false)}>×</button></div>
            <GlossaryTerm term="MCT">Minimum Completion Time: selects the candidate predicted to finish first.</GlossaryTerm>
            <GlossaryTerm term="EWMA">A causal moving average EdgeWeaver uses to update device/model latency estimates.</GlossaryTerm>
            <GlossaryTerm term="P95 latency">A latency that 95% of completed requests are at or below.</GlossaryTerm>
            <GlossaryTerm term="Minimum accuracy">The request's eligibility threshold, compared with measured model test accuracy.</GlossaryTerm>
            <GlossaryTerm term="Useful goodput">Correct, on-time, accuracy-eligible completions per simulated second.</GlossaryTerm>
            <GlossaryTerm term="Normalized energy (NU)">A simulated comparison unit—not joules or measured electricity.</GlossaryTerm>
            <div className="measurement-note"><strong>Measured</strong><span>Model accuracy, F1, artifact size, and local latency.</span><strong>Simulated</strong><span>Devices, network, queues, slowdowns, and normalized energy.</span></div>
          </aside>
        </div>
      ) : null}
    </div>
  );
}

function GlossaryTerm({ term, children }: { term: string; children: React.ReactNode }) {
  return <div className="glossary-term"><strong>{term}</strong><p>{children}</p></div>;
}
