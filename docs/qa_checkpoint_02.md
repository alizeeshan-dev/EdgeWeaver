# QA Checkpoint 02 — Phases 3–4

## Intended scope

Phase 3 provides the deterministic, scheduler-agnostic three-device SimPy engine, replayable UCI HAR
test-sample traces, real model prediction integration, bounded queues, local/remote timing, normalized
energy, lifecycle events, and structured request/run results. Phase 4 adds one scheduler protocol,
shared candidate estimates, and exactly three baselines: Round Robin, Fastest Device, and Minimum
Completion Time. The same saved trace runs under every policy.

Main components are `edgeweaver/engine.py`, `edgeweaver/domain.py`, `edgeweaver/network.py`,
`edgeweaver/energy.py`, `edgeweaver/workloads.py`, `edgeweaver/scheduler.py`, and
`edgeweaver/schedulers/`. The CLI is `scripts/run_simulation.py`; focused policy fixtures are in
`tests/test_schedulers.py`, with engine regression coverage in `tests/test_engine.py`.

## Required behavior and invariants

- Device/model compatibility and measured profile accuracy gate every assignment; no silent fallback.
- `predicted_completion_ms` is absolute simulation time and is compared with arrival plus deadline.
- Shared estimates use scheduler overhead, symmetric transfer legs, actual remaining/queued service
  work (including assigned work still in transit), measured-mean latency scaled by device speed, and
  the same normalized energy formulas as execution. Existing work may drain during overhead/upload.
- Round Robin rotates configured devices, skips unsatisfiable devices, selects that device's fastest
  qualifying model, and ignores queue/network values when selecting.
- Fastest Device selects each device's highest-accuracy qualifying model, then the lowest isolated
  scaled inference time; it intentionally ignores congestion and network delay.
- MCT selects the earliest estimated completion across qualifying pairs even when every pair misses.
- MCT excludes a device when its bounded queue is predicted to be unable to admit the request;
  Round Robin and Fastest Device intentionally remain queue-unaware and may be rejected at execution.
- Tie-breaks use stable IDs. A new registry-created Round Robin instance resets its cursor.
- Candidate estimates and concise reasons are retained in each structured assignment decision.
- The simulator calls only the common scheduler interface and contains no policy-name branches.

## Useful QA commands

```bash
python scripts/run_simulation.py --scheduler fixed
python scripts/run_simulation.py --scheduler round_robin
python scripts/run_simulation.py --scheduler fastest_device
python scripts/run_simulation.py --scheduler min_completion
pytest tests/test_schedulers.py tests/test_engine.py
pytest
ruff check edgeweaver scripts tests
ruff format --check edgeweaver scripts tests
mypy edgeweaver scripts/run_simulation.py
```

Inspect the small fixtures for congested isolated-fast devices, poor remote links, accuracy exclusion,
impossible deadlines, device incompatibility, actual heterogeneous queued work, absolute deadlines,
stable ties, fresh Round Robin state, same-trace replay, and idle prediction-versus-execution equality.
Repeated runs with the same trace/config/seed and a fresh scheduler should be structurally identical.

## Risks, assumptions, and deferred work

Assigned work is reserved immediately and carries a deterministic predicted queue-ready timestamp,
so later same-time decisions can see local, overhead-delayed, and remote in-transit work. Queue
capacity continues to mean waiting slots, excluding active service. Remote links are symmetric, and
stored measured mean latency is the fixed simulation base.

High-risk areas are same-timestamp SimPy ordering, multi-capacity FIFO estimation, queue admission
near capacity, jitter lead-time accounting, floating-point equality at deadlines, and accidental
reuse of a stateful Round Robin instance. Phase 5+ intentionally remains absent: EdgeWeaver policy,
feasibility-first energy selection, online/EWMA updates, slowdown scenarios, metrics/analytics,
experiments, ablations, API, frontend, and deployment. EdgeWeaver is not implemented.
