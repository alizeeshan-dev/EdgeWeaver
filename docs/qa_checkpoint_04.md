# QA Checkpoint 04 — Phases 7–8

## Intended scope and current state

Phase 7 provides the paired, resumable experiment/ablation runner, saved-result aggregation, eight
research figure groups, evidence extraction, and cautious H1–H5 findings. Phase 8 completes the
non-frontend audit, strengthens run provenance, adds one small end-to-end core integration test, and
provides the README, methodology, report, architecture diagram, and manual demo script.

The expected saved study is exactly:

- 80 core runs = four policies × four scenarios × seeds `[1,2,3,4,5]`;
- 5 `edgeweaver_no_model_switching` runs on `bursty` only;
- 5 `edgeweaver_no_online_update` runs on `device_slowdown` only;
- 20 trace groups, one immutable trace per scenario/seed shared across core policies and applicable
  ablations.

Run manifests are committed last and hash detailed run, event, summary, trace, configurations,
profiles, models, UCI dataset tree, executable Python source, and runtime versions. Resume must skip
only an exact valid identity; corrupt/stale artifacts must rerun. Saved-result analysis is separate
from execution and must not load/reprofile models or rerun simulation.

## Major files and outputs

- Execution/provenance: `edgeweaver/experiments.py`, `configs/experiment.yaml`,
  `scripts/run_experiments.py`
- Aggregation/reporting: `edgeweaver/analysis.py`, `edgeweaver/reporting.py`,
  `edgeweaver/research.py`, `scripts/generate_research_outputs.py`
- Ablations: `edgeweaver/schedulers/edgeweaver.py`
- Integration fixture: `tests/test_core_integration.py`
- Documentation: `README.md`, `docs/methodology.md`, `docs/report.md`,
  `docs/demo_script.md`, `docs/implementation_status.md`
- Detailed outputs: `experiments/phase7/raw/{runs,events}/`
- Completion/per-run outputs: `experiments/phase7/results/{run_records,per_run,per_run_tidy}/`
- Eight tidy tables: `experiments/phase7/results/tables/`
- Eight PNG chart groups: `experiments/phase7/figures/`
- Evidence/findings: `experiments/phase7/results/failure_cases.json` and
  `experiments/phase7/results/findings_summary.md`

Required figures are deadline satisfaction, useful goodput, mean/P95 latency, actual accuracy,
estimated normalized energy/completion, device-utilization heatmap, model-selection distribution, and
EdgeWeaver ablations. Energy must never be labeled as joules or physical measurement.

## Expected research results

Network Slowdown is the only material deadline separation: five-seed means are 100% for MCT and
EdgeWeaver, 93.66% for Round Robin, and 79.90% for Fastest Device. All policies average 100% in the
other scenarios. Every assignment uses logistic regression because it is the measured most-accurate
and lowest-compute practical model.

- H1: not supported; Fastest Device has only 0.110 ms maximum burst queue wait and no deadline loss.
- H2: partially supported; MCT exceeds Round Robin by 6.34 percentage points only during Network
  Slowdown.
- H3/H4: inconclusive; full/no-switch variants realize identical model choices.
- H5: inconclusive; EdgeWeaver does not select the slowed edge server, so adaptation is unexercised.
- Only the remote network-slowdown miss is observed among the five requested representative case
  types. Four cases are explicitly absent; they must not be manufactured.

## Useful QA commands

```bash
python -m pip install -e ".[dev]"
python scripts/verify_models.py
python scripts/run_simulation.py --scenario bursty --scheduler edgeweaver --seed 1
python scripts/run_experiments.py --dry-run
python scripts/run_experiments.py --validate-only
python scripts/run_experiments.py
python scripts/generate_research_outputs.py
pytest
ruff check edgeweaver scripts tests
ruff format --check edgeweaver scripts tests
mypy edgeweaver scripts/run_simulation.py scripts/run_experiments.py \
  scripts/generate_research_outputs.py
```

The default experiment command should resume all 90 valid runs and regenerate downstream outputs.
The analysis-only command should work from saved traces/raw runs/manifests/summaries without the raw
UCI dataset or `joblib` models. Ordinary pytest must not download, retrain, reprofile, or execute the
80-run matrix.

## Core completeness checklist

- Official UCI HAR split validation, three trained/reloadable models, fitted preprocessors,
  accuracy/F1, artifact sizes, 500-prediction local profiles, and environment metadata.
- Three YAML devices/two links, queues/capacities, local/remote timing, real prediction integration,
  normalized energy, full events, replayable held-out-sample traces.
- Common scheduler interface; Round Robin, Fastest Device, MCT, EdgeWeaver; causal/resettable EWMA;
  baselines non-adaptive.
- Four deterministic scenarios, exact slowdown events/current-state visibility/reset, required
  metrics and miss causes.
- Exact paired matrix/ablations, atomic resume, aggregate mean/sample-SD/seed values, figures/cases/
  findings, and CLI-only reproducibility.
- README/methodology/report/demo/status distinguish measured and simulated quantities and state all
  required limitations.

## High-risk QA targets

- Missing/duplicate logical runs or mismatched trace IDs/hashes within scenario/seed pairs.
- Dataset/source/runtime changes failing to invalidate resume, or stale aggregate results after a
  rerun.
- Ablations altering behavior beyond their isolated switch, or adaptive state leaking across runs.
- Figures disagreeing with current tables, obscuring units, or treating normalized energy as physical.
- Hypothesis labels drifting from actual effect sizes; especially overstating tiny burst queues or
  claiming a deadline-driven model switch that did not occur.
- Accuracy profile eligibility confused with actual request correctness.
- Physical local profiling confused with simulated devices/network/slowdowns.
- Missing limitations: one physical computer, scaled devices, normalized energy, overall-accuracy
  approximation, simplified network, one ML domain, omitted OS/hardware effects, no production claim.
- Nondeterministic event order/replay, future slowdown leakage, mutable static profiles, or runtime
  conditions not resetting.
- Absolute/machine-specific paths used as operational inputs. The training manifest may retain its
  original measurement path as provenance, but runtime loaders must rely on configured paths.
- README commands failing outside the repository working directory after editable installation.

## Assumptions, caveats, and deferred work

All 90 saved result records have a null Git commit because they were generated before this workspace
had a resolvable committed HEAD; source SHA-256 is therefore their executable-code provenance
fallback. Stored scikit-learn artifacts should be retrained when using an incompatible library
environment; runtime versions are part of experiment identity. Five seeds and one configured
workload family limit inference.

Phase 9 remains intentionally absent: no FastAPI, React, Vite, TypeScript, browser UI, CORS, auth,
database, deployment, or production infrastructure. This checkpoint is a QA handoff, not a claim
that deep QA has already been completed.
