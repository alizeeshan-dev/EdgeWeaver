# EdgeWeaver Research Methodology

## Research question and hypotheses

EdgeWeaver asks how interactive ML inference requests should be assigned to simulated edge devices
when model accuracy, deadlines, queueing, network delay, and estimated energy conflict. The study
tests five preregistered engineering hypotheses:

- **H1:** Fastest Device performs well at low load but develops long queues during bursts.
- **H2:** Minimum Completion Time (MCT) meets more deadlines than Round Robin.
- **H3:** EdgeWeaver improves burst deadline performance by selecting smaller models when needed.
- **H4:** EdgeWeaver uses less estimated energy than accuracy-first selection while retaining
  acceptable prediction quality.
- **H5:** Online latency updates help EdgeWeaver respond to device slowdowns.

These are evaluated against the configured simulation, not treated as assumptions. Unsupported or
unexercised hypotheses remain valid research outcomes.

## Dataset and model training

The only dataset is the UCI Human Activity Recognition Using Smartphones dataset. Its 561 engineered
accelerometer/gyroscope features describe six activities. The official split is preserved: 7,352
training examples and 2,947 held-out test examples. Loading validates required files, feature count,
label/subject alignment, feature names, and activity labels. Downloading is separate from loading so
an existing official dataset can be reused.

Three scikit-learn estimators are trained with seed 2027:

| Role | Stable ID | Estimator and preprocessing | Key configuration |
|---|---|---|---|
| Light | `logistic-regression-v1` | Logistic regression + `StandardScaler` | `C=1`, `lbfgs`, 2,000 maximum iterations |
| Balanced | `random-forest-v1` | Random forest + identity transform | 200 trees, square-root features, one worker |
| Heavy | `mlp-v1` | MLP + `StandardScaler` | hidden layers 128/64, Adam, early stopping |

Accuracy and macro F1 are calculated on the same official test split. Fitted estimators and
preprocessors are saved separately with `joblib`; metadata records hyperparameters, sample counts,
class labels, library version, artifact sizes, and evaluation results. Reloaded artifacts must
reproduce held-out predictions. Role names describe intended compute roles and do not impose an
accuracy ordering.

## Physical profiling and model profiles

Profiling loads all artifacts and data before timing, performs 20 untimed warm-up predictions, then
uses a seeded selection of 500 held-out examples per model. Each observation transforms and predicts
exactly one sample and is timed with `time.perf_counter_ns()`. Raw nanosecond observations are saved;
mean, median, population standard deviation, and linear P95 are derived from them. Timer overhead is
not subtracted.

The resulting values are physically measured only on the computer described in
`artifacts/profiles/profiling_environment.json`. Measured quantities are test accuracy, macro F1,
model file size, and local single-request latency. The stored measured means used by simulation are:

| Model | Accuracy | Macro F1 | Size | Mean / median / P95 latency |
|---|---:|---:|---:|---:|
| Logistic regression | 0.95487 | 0.95477 | 25,661 B | 0.246 / 0.236 / 0.336 ms |
| Random forest | 0.92840 | 0.92640 | 2,568,840 B | 12.660 / 12.254 / 14.896 ms |
| MLP | 0.94571 | 0.94577 | 1,858,312 B | 0.336 / 0.328 / 0.372 ms |

Profiling is not repeated during simulation. Wall-clock measurements naturally vary if profiling is
rerun on another computer; stored profiles are fixed inputs to deterministic simulation.

## Simulated system model

Exactly three single-server device resources are configured in YAML:

| Device | Speed multiplier | Active power units | Waiting capacity | Supported roles |
|---|---:|---:|---:|---|
| Mobile | 2.5 | 1.0 | 20 | light, balanced |
| Gateway | 1.3 | 2.0 | 40 | light, balanced, heavy |
| Edge server | 0.6 | 4.0 | 80 | light, balanced, heavy |

The measured model mean is converted to simulated service time as

```text
simulated inference time = measured local mean latency × device speed multiplier
```

These device speeds are assumptions, not hardware measurements. SimPy resources model active work
and bounded waiting queues. A full queue causes an explicit rejected request and event; no request is
silently discarded.

Mobile execution has zero network time. Gateway and edge-server links are symmetric abstractions
configured with 12 ms / 20 Mbps and 35 ms / 10 Mbps one-way base latency/bandwidth respectively.
For each upload or return:

```text
transfer time = base one-way latency + input bits / bandwidth bits per millisecond
```

Optional jitter is seeded; the core configuration disables it. No packets, transport protocol,
routing, retransmission, or physical network is modeled.

End-to-end latency is

```text
upload + queue wait + inference + return + scheduler overhead
```

Deadlines are relative to arrival. A completion succeeds when

```text
completion timestamp <= arrival timestamp + relative deadline
```

The simulator obtains the selected model's real prediction for the referenced held-out feature
vector, while advancing only SimPy time. Saved model accuracy is an eligibility approximation; it
does not guarantee an individual prediction is correct.

## Normalized energy model

Energy is deliberately estimated in normalized units:

```text
compute energy = active power units × simulated inference time in ms
network energy = transferred kilobytes × configured network-energy units per kilobyte
estimated total energy = compute energy + network energy
```

The default transfer coefficient is 0.02 normalized units/kB. These values are not joules and were
not physically measured.

## Scheduling policies

All policies consume the same immutable request and read-only simulation state and return the same
structured assignment decision schema.

- **Round Robin:** continue through the device rotation until a compatible device has an
  accuracy-eligible model; select its lowest simulated inference-time model. Queue and network state
  do not influence device choice.
- **Fastest Device:** select the device with the lowest isolated inference duration, ignoring queues;
  on that device use the highest measured-accuracy compatible model.
- **MCT:** enumerate all compatible, accuracy-eligible and queue-admissible pairs and choose the
  earliest absolute predicted completion. Predictions include current upload, existing reserved and
  queued work, inference, return, and scheduler overhead. If all candidates miss the deadline, the
  earliest is still selected.
- **EdgeWeaver:** enumerate the same candidates; among deadline-feasible candidates choose minimum
  estimated energy, then earliest completion, then stable device/model IDs. If no candidate is
  feasible, choose earliest completion. If no valid candidate exists, fail explicitly without
  weakening accuracy.

EdgeWeaver initializes separate runtime latency estimates from each measured-mean/device-multiplier
pair. After a selected inference completes, only that pair is updated:

```text
new estimate = alpha × observed inference duration + (1 - alpha) × old estimate
```

The default `alpha` is 0.2. Queue and network time are excluded. Updates are causal, produce a
`PROFILE_UPDATED` event only when the estimate changes, never mutate measured profile JSON, and are
reset for each independent run. Baseline schedulers remain non-adaptive.

## Workloads and scenarios

Each request stores arrival time, relative deadline, minimum profile accuracy, input size, source,
true label, and held-out feature-vector ID. A dedicated seeded generator samples UCI HAR test indices
and scenario distributions before any scheduler runs. A versioned trace records the scenario hash,
seed, duration, request count, and dataset split reference. One saved trace is reused for every policy
in a scenario/seed group.

All scenarios have a 20-second configured horizon:

- **Normal:** stable 4 requests/s; deadlines 100/180/300 ms.
- **Bursty:** background 1.5 requests/s plus three 1.5-second periods at 16 requests/s; deadlines
  80/140/240 ms.
- **Network slowdown:** normal arrivals; the mobile-to-edge-server link has triple base latency and
  one-quarter bandwidth during `[7000, 14000)` ms.
- **Device slowdown:** 5 requests/s; edge-server service is tripled during `[7000, 14000)` ms.

Minimum-accuracy values 0.90/0.93/0.95 and request sizes are sampled from each scenario's declared
weights and dataset-row representation. Runtime transition events occur at the exact boundaries.
Network state is sampled when each transfer starts; device state is sampled when inference starts.
Schedulers see current state but never the future slowdown schedule. Runtime conditions reset after
every run.

## Experiment and ablations

The core paired design is four schedulers × four scenarios × seeds `[1,2,3,4,5]`, totaling 80 runs.
For a scenario/seed group, trace ID and SHA-256 must match across all policies. Run identities also
include scenario, device, network, simulation, model-profile, model-artifact, dataset, source-code,
runtime-version, and trace digests. Detailed runs/events, summaries, and completion manifests are
written separately and atomically; valid identities resume by default.

Two five-seed ablations reuse the matching core traces:

- **No model switching**, Bursty only: constrain EdgeWeaver to the most accurate qualifying model;
  keep energy/deadline logic and online updates unchanged.
- **No online update**, Device Slowdown only: keep normal selection/model switching but suppress EWMA
  updates.

They are analysis variants, not additional core policies.

## Per-run metrics

- **Deadline satisfaction:** on-time completed requests / completed requests; zero completions yields
  zero.
- **Useful goodput:** correct, on-time, profile-accuracy-eligible completed predictions / configured
  duration in seconds.
- **Latency:** mean and linear P95 of completed end-to-end latency.
- **Actual accuracy:** correct completed predictions / completed predictions.
- **Estimated energy per completion:** total normalized estimated energy / completed requests.
- **Device utilization:** union of actual inference busy intervals inside the configured horizon /
  horizon. Queue and network time are excluded.
- **Model selection:** assigned-request count and percentage by stable model ID/role. Rejections are
  reported separately.
- **Miss causes:** count/rate for network delay, queue delay, inference time, and incorrect static
  estimate.

For miss classification, `incorrect_static_estimate` is used conservatively only when the scheduler
predicted success, actual inference exceeded the selected estimate, and removing that underestimate
alone would make the deadline. Otherwise, the largest actual component among aggregate network,
queue, and inference time wins; exact ties resolve network, then queue, then inference.

## Aggregation and figures

Scalar and nested metrics are retained per seed. Each scheduler/scenario aggregate reports the
arithmetic mean, sample standard deviation (`n-1`), run count, and ordered individual seed values.
No significance tests, p-values, confidence intervals, or selective seed removal are used.

Eight headless Matplotlib outputs are regenerated solely from validated saved results: deadline
satisfaction, useful goodput, mean/P95 latency, actual accuracy, estimated normalized energy,
device-utilization heatmap, model-selection distribution, and the two-part EdgeWeaver ablation
comparison. Failure-case extraction applies deterministic evidence rules; a desired case that did not
occur is recorded as unobserved rather than manufactured.

## Experimental boundary

Only accuracy/F1, artifact size, and local inference latency were physically measured. Device speed,
remote execution, queues, network behavior, slowdowns, and normalized energy are simulated. The
study uses one dataset/domain, one physical profiling environment, simplified system abstractions,
and five seeds. It does not establish production edge-system performance.
