# Implementation Status

## Phase 1 — complete

Validated Pydantic configuration/domain models, three editable device profiles, two network links,
an explicitly synthetic fixture profile, and the deterministic one-resource/five-request SimPy
slice remain working. It emits 20 ordered events and meets 3/5 fixture deadlines.

## Phase 2 — complete

Implemented official UCI HAR download/extraction (including the current nested ZIP), strict loading
of the preserved `(7352, 561)` train and `(2947, 561)` test splits, deterministic preprocessing,
three seeded scikit-learn models, accuracy/macro-F1 evaluation, joblib artifacts, uniform reload and
prediction, and local single-request profiling. Each model used 20 warm-ups and 500 timed
transform-plus-predict calls selected with seed 2027 and timed by `perf_counter_ns`.

| Role / model | Accuracy | Macro F1 | Mean ms | Median ms | Std ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| light / logistic-regression-v1 | 0.954869 | 0.954767 | 0.246006 | 0.236300 | 0.031251 | 0.336440 |
| balanced / random-forest-v1 | 0.928402 | 0.926404 | 12.660245 | 12.253900 | 1.224675 | 14.896040 |
| heavy / mlp-v1 | 0.945707 | 0.945768 | 0.336276 | 0.328050 | 0.034855 | 0.372120 |

Generated data/artifacts: `data/raw/UCI HAR Dataset/`, `artifacts/models/`, and measured profile,
raw timing, manifest, and local-only environment JSON files in `artifacts/profiles/`. Model/preprocessor
joblib files and raw downloaded data are intentionally git-ignored; metadata and profiles are small.

```bash
python scripts/download_uci_har.py
python scripts/train_models.py --seed 2027
python scripts/verify_models.py --samples 32
python scripts/profile_models.py --predictions 500 --warmups 20 --seed 2027
python -m edgeweaver
pytest
ruff check edgeweaver scripts tests
mypy edgeweaver
```

Measurements are from this local Windows computer only and are not simulated device or energy
measurements. Wall-clock latency will vary across runs.

## Phase 3 — complete

Implemented a scheduler-agnostic SimPy engine for exactly `mobile`, `gateway`, and `edge-server`.
It consumes the stored measured-mean profiles, applies YAML device speed multipliers, models bounded
waiting queues and configurable processing capacity, performs symmetric upload/return timing over
the two simple links, reuses loaded Phase 2 models for actual held-out predictions, and records full
timing, deadline, normalized energy, rejection, assignment, and lifecycle-event data. Queue capacity
means waiting slots and excludes active inference. Optional jitter is stable per request/link/direction.

Versioned JSON workload traces validate UCI test indices/labels and replay independently of the
assignment provider. The representative fixed assignment file is explicitly a verification hook,
not a scheduler algorithm. Its six-request run completes all requests across all three devices,
emits 37 events, records one deadline miss, and replays byte-identically.

```bash
python scripts/create_phase3_fixture.py
python scripts/run_simulation.py --scheduler fixed
pytest
ruff check edgeweaver scripts tests
mypy edgeweaver
```

Key outputs are `artifacts/workload_traces/phase3-representative.json`,
`configs/phase3_assignments.json`, and `experiments/raw/phase3-representative-run.json`.
Remote links are treated as symmetric; response payload is the configurable 64-byte default.
Scenarios, aggregate metrics, experiments, ablations, API, and frontend remain deferred.

## Phase 4 — complete

Added one common scheduler protocol, structured candidate estimates, shared compatibility/timing/
queue/energy calculations, deterministic tie-breaking, a fresh-instance registry, and the three
baseline policies: `round_robin`, `fastest_device`, and `min_completion`. Queue estimates use active
remaining work plus reserved/in-transit and queued requests' assigned inference durations; network
upload time is accounted for while existing work drains. MCT excludes queues predicted to be unable
to admit a request. Candidate completion values are absolute simulation times.

```bash
python scripts/run_simulation.py --scheduler round_robin
python scripts/run_simulation.py --scheduler fastest_device
python scripts/run_simulation.py --scheduler min_completion
pytest
ruff check edgeweaver scripts tests
mypy edgeweaver scripts/run_simulation.py
```

Baseline outputs are `experiments/raw/phase4-<scheduler>-run.json`. On the six-request saved trace,
Round Robin rotated mobile/gateway/edge-server and met 6/6 deadlines; Fastest Device selected the
isolated-fast edge-server with the actually highest-accuracy qualifying model and met 3/6; MCT
selected earliest predicted completions and met 6/6. All completed 6/6 requests. Scheduler decisions
retain compact reasons plus structured estimates for every candidate. The focused/full suite passed
45 tests; Ruff lint/format and strict mypy checks passed. EdgeWeaver, online updates,
slowdowns, scenarios, aggregate metrics, experiment matrices, and ablations remain Phase 5+ work.

## Phase 5 — complete

Implemented `edgeweaver` through the common scheduler interface. It reuses shared candidates and
measured accuracy, chooses lowest normalized energy among deadline-feasible assignments, resolves
ties by completion then stable IDs, and falls back to earliest completion when no valid assignment
can meet the deadline. No-valid-candidate failure remains explicit.

EdgeWeaver keeps separate per-device/model latency estimates initialized from immutable measured-
mean profiles plus device multipliers. After each completed inference, a generic optional observation
hook applies configurable EWMA (`edgeweaver_ewma_alpha`, default `0.2`) to inference time only and
emits `PROFILE_UPDATED`; baselines have no observation hook and remain static. Scheduler-visible
reservations retain predicted durations, so controlled actual-duration changes do not leak early.

```bash
python scripts/run_simulation.py --scheduler edgeweaver
pytest tests/test_edgeweaver.py tests/test_schedulers.py tests/test_engine.py
pytest
ruff check edgeweaver scripts tests
mypy edgeweaver scripts/run_simulation.py
```

The representative output is `experiments/raw/phase5-edgeweaver-run.json`: 6/6 requests completed
and 6/6 deadlines were met. Stable observations do not emit no-op profile updates; controlled
latency changes emit causal updates. Focused fixtures cover all selection rules, EWMA behavior, pair
isolation, controlled slower execution, reset/replay, profile
immutability, and baseline non-adaptation. All 61 tests, Ruff lint/format, strict mypy, the real CLI
run, and deterministic output replay passed.

## Phase 6 — complete

Added strict YAML scenarios for `normal`, `bursty`, `network_slowdown`, and `device_slowdown`;
seeded scheduler-independent UCI HAR workload generation; versioned trace provenance; and half-open,
time-bounded runtime conditions. Network state is visible only at the current decision/transfer time.
Device slowdown changes actual service only; EdgeWeaver learns only from completed inference
observations. Base configuration and measured profiles are never mutated, and runtime state resets.

The dedicated metrics layer exports per-run JSON and tidy CSV for completion/rejection counts,
deadline satisfaction, useful goodput (correct, constraint-valid, on-time requests/second), mean/P95
end-to-end latency, actual prediction accuracy, normalized estimated energy, compute-only per-device
utilization, assigned-request model distribution, and deterministic deadline-miss causes. Raw run,
event log, and summary files remain separate.

```bash
python scripts/run_simulation.py --scenario bursty --scheduler edgeweaver --seed 42
python scripts/run_simulation.py --scenario normal --scheduler min_completion \
  --trace artifacts/workload_traces/normal-seed-42.json
pytest tests/test_scenarios.py tests/test_runtime_conditions.py tests/test_metrics.py
pytest
ruff check edgeweaver scripts tests
ruff format --check edgeweaver scripts tests
mypy edgeweaver scripts/run_simulation.py
```

Outputs use `artifacts/workload_traces/`, `experiments/raw/`, `experiments/events/`, and
`experiments/summaries/`. One-seed smoke runs succeeded for all 16 scheduler/scenario combinations;
91 tests, Ruff lint/format, strict mypy, and byte-identical replay passed. QA additionally verified
trace identity, measured-profile immutability, exact slowdown factors/events, and corrected no-op
`PROFILE_UPDATED` emission. Scenario parameters are
editable research inputs and were not tuned to enforce a preferred policy outcome. Phase 7's 80-run
matrix, ablations, aggregation/statistics/charts, and all API/frontend/deployment work remain deferred.

## Phase 7 — complete

Added the fixed, resumable experiment runner for four schedulers × four scenarios × seeds
`[1, 2, 3, 4, 5]`, plus the two five-seed EdgeWeaver ablations. One trace is generated per
scenario/seed and its ID/hash is validated across paired policies. Completion manifests are written
last, include input/output hashes and provenance, and permit corrupt or interrupted runs to be rerun
without overwriting valid runs by default. The default command runs/resumes all 90 simulations and
then regenerates tables, eight chart groups, structured failure cases, and cautious H1–H5 findings.

```bash
python scripts/run_experiments.py
python scripts/run_experiments.py --validate-only
python scripts/generate_research_outputs.py
pytest
ruff check edgeweaver scripts tests
ruff format --check edgeweaver scripts tests
mypy edgeweaver scripts/run_experiments.py scripts/generate_research_outputs.py
```

The real study completed 80/80 core and 10/10 ablation runs using 20 paired traces. Outputs are under
`experiments/phase7/`: detailed runs/events in `raw/`, per-run and aggregate CSVs in
`results/tables/`, figures in `figures/`, and the evidence/findings outputs at
`results/failure_cases.json` and `results/findings_summary.md`. Network slowdown was the only core
condition separating deadline rates materially: MCT and EdgeWeaver averaged 100%, Round Robin
93.66%, and Fastest Device 79.90%. All policies averaged 100% in the other scenarios. Both scoped
ablations matched full EdgeWeaver on the tested metrics because the measured logistic-regression
profile is simultaneously the most accurate, fastest practical local choice, and lowest-energy
choice under these inputs. H1 was not supported; H2 was partially supported; H3–H5 were
inconclusive because their intended contrasts were not exercised. One of five requested
representative case types was observed; missing cases are explicitly recorded rather than
manufactured. The Phase 7 implementation checks passed before final Phase 8 review.

No parameters, seeds, profiles, or valid results were altered or discarded to favor a hypothesis.

## Phase 8 — non-frontend core complete

Completed the core capability/reproducibility audit and added the portfolio README, research
methodology, concise report, Mermaid architecture diagram, honest manual demo script, and Phases 7–8
QA handoff. Run identity now includes UCI dataset bytes, executable Python source, and relevant
runtime versions in addition to the existing input/artifact hashes; all 90 affected runs were rerun
and downstream outputs regenerated. Saved-result analysis is decoupled from dataset/model loading.
The package now declares pandas and includes an optional exact direct-version constraints file.

The small integration test loads profiles/configuration and one fixed trace, runs all four policies,
exports detailed/events/JSON/CSV outputs, and validates required metrics without training or running
the full matrix. Final checks passed: 120 tests, Ruff lint/format, strict mypy, artifact reload,
representative CLI simulation, 80+10/20-pair validation, resume skipping 90/90, byte-identical forced
replay, dependency check, and saved-results-only regeneration.

```bash
python -m pip install -c constraints.txt -e ".[dev]"
python scripts/download_uci_har.py
python scripts/train_models.py
python scripts/verify_models.py
python scripts/profile_models.py --predictions 500
python scripts/run_simulation.py --scenario bursty --scheduler edgeweaver --seed 1
python scripts/run_experiments.py
python scripts/generate_research_outputs.py
pytest
```

Documentation is in `README.md`, `docs/methodology.md`, `docs/report.md`, and
`docs/demo_script.md`; the QA handoff is `docs/qa_checkpoint_04.md`. No unresolved core blocker is
known. Phase 9's FastAPI/React presentation layer remains explicitly deferred and no API/frontend
code exists.
