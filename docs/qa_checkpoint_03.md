# QA Checkpoint 03 — Phases 5–6

## Intended scope

Phase 5 adds EdgeWeaver to the shared scheduler interface: measured-accuracy filtering, deadline
feasibility, lowest normalized energy among feasible candidates, completion/ID tie-breaking,
earliest-completion fallback, and per-device/model EWMA latency adaptation (`alpha=0.2` default).
Phase 6 adds four controlled scenarios, seeded UCI HAR workload generation, runtime slowdowns,
per-run research metrics, and separate trace/raw-event/run-summary exports.

Main files are `edgeweaver/schedulers/edgeweaver.py`, `edgeweaver/scenarios.py`,
`edgeweaver/runtime_conditions.py`, `edgeweaver/metrics.py`, `edgeweaver/events.py`,
`edgeweaver/engine.py`, `edgeweaver/workloads.py`, `configs/scenarios/*.yaml`, and
`scripts/run_simulation.py`. Focused fixtures are in `tests/test_edgeweaver.py`,
`tests/test_scenarios.py`, `tests/test_runtime_conditions.py`, and `tests/test_metrics.py`.

## Required invariants and metric definitions

- Static measured profiles plus device multipliers initialize adaptive estimates and remain immutable.
- EWMA uses inference/service duration only and updates the selected pair after completion. State is
  reset before every run; baselines have no observation hook. An unchanged estimate does not emit a
  no-op `PROFILE_UPDATED` event.
- Slowdown windows are half-open `[start, end)`. A network condition is sampled independently when
  upload and return begin; a device multiplier is sampled when inference begins. In-flight work keeps
  the sampled duration. Transition events occur at exact boundaries before same-time arrivals.
- Schedulers see current network links, never the future condition schedule. Device slowdown changes
  actual service without directly altering EdgeWeaver's estimate; predicted reservations deliberately
  remain prediction-based until a completed observation updates EdgeWeaver.
- Trace generation accepts no scheduler. Trace metadata records scenario, seed, generator version,
  scenario-config hash, duration, and count; saved traces must match config/hash when replayed.
- Deadline satisfaction is on-time completions / completed requests; a zero denominator yields 0.
- Useful goodput is actually correct, on-time, profile-accuracy-eligible completions / configured
  duration in seconds. Actual accuracy uses held-out predictions, not profile accuracy.
- Latency is end-to-end. Utilization is the union of actual inference intervals clipped to configured
  duration / duration; network and queue time are excluded. Energy is normalized estimated units.
- Model selection uses assigned requests as its explicit denominator, including assignments later
  rejected by execution; rejection counts remain separate.
- Miss classification first uses `incorrect_static_estimate` only when predicted success plus an
  inference underestimate alone explains the miss. Otherwise the largest actual network, queue, or
  inference component wins; exact ties resolve network, then queue, then inference.

## Useful QA commands

```bash
python scripts/run_simulation.py --scenario normal --scheduler round_robin --seed 7
python scripts/run_simulation.py --scenario bursty --scheduler fastest_device --seed 7
python scripts/run_simulation.py --scenario network_slowdown --scheduler min_completion --seed 7
python scripts/run_simulation.py --scenario device_slowdown --scheduler edgeweaver --seed 7
python scripts/run_simulation.py --scenario normal --scheduler edgeweaver \
  --trace artifacts/workload_traces/normal-seed-7.json
pytest tests/test_edgeweaver.py tests/test_scenarios.py tests/test_runtime_conditions.py \
  tests/test_metrics.py
pytest
ruff check edgeweaver scripts tests
ruff format --check edgeweaver scripts tests
mypy edgeweaver scripts/run_simulation.py
```

Manually inspect exact start/end events, a transfer or inference crossing a boundary, the controlled
causal EWMA fixture, same-trace runs under all four schedulers, deterministic replay hashes, zero-
completion metrics, all four miss causes, and prediction-versus-execution timing fields. A one-seed
4×4 smoke grid should succeed; it is intentionally not automated as a Phase 7 experiment runner.

## Risks, assumptions, and deferred work

High-risk targets are accidental static-profile mutation, adaptive-state leakage, future slowdown
leakage, scheduler-dependent traces, predicted/actual queue mismatch during unexpected slowdown,
latency unit/denominator errors, profile versus actual accuracy confusion, and energy mislabeled as
physical energy. Remote links remain symmetric. Utilization uses the configured scenario horizon,
even if a late request finishes afterward. The current measured profiles make local execution the
lowest-energy feasible EdgeWeaver choice in the default device-slowdown scenario; the controlled
fixture proves causal adaptation when the slowed edge pair is selected, without tuning results.

Phase 7 remains absent: no 80-run orchestration, resume logic, ablations, cross-seed aggregation,
confidence intervals, statistical analysis, chart generation, representative failure selection, or
findings report. API, frontend, databases, auth, deployment, and production infrastructure are also
absent.
