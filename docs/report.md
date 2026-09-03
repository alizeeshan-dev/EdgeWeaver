# EdgeWeaver: Deadline-, Accuracy-, and Energy-Aware ML Inference Scheduling in a Simulated Edge System

## 1. Abstract

EdgeWeaver is a local research prototype for comparing machine-learning inference scheduling
policies across three simulated heterogeneous edge devices. Three scikit-learn classifiers were
trained on the official UCI Human Activity Recognition Using Smartphones split, evaluated for test
accuracy and macro F1, and physically profiled for single-request inference latency on one computer.
Those measurements were then used as immutable inputs to a deterministic SimPy model of a mobile
device, gateway, edge server, bounded queues, and simplified network links. Four policies—Round
Robin, Fastest Device, Minimum Completion Time (MCT), and EdgeWeaver—were evaluated over normal,
bursty, network-slowdown, and device-slowdown scenarios using five paired seeds (80 core runs), plus
10 scoped EdgeWeaver ablations.

The main observed separation occurred under network slowdown: MCT and EdgeWeaver averaged 100%
deadline satisfaction, Round Robin 93.66%, and Fastest Device 79.90%. All policies averaged 100% in
the other scenarios. The measured logistic-regression model was both the most accurate and the
lowest-compute practical choice, so every policy selected it for every request. The model-switching
and online-update ablations therefore did not activate their intended causal contrast. These
non-results expose an important experimental limitation rather than evidence to be hidden or tuned
away. EdgeWeaver is presented as a reproducible scheduling study, not as proof of production edge
performance.

## 2. Introduction

Interactive edge inference creates a coupled assignment problem. Local execution avoids network
delay but may be slower; remote execution can provide faster compute while adding transfer latency
and congestion; different models may offer different accuracy, latency, memory, and energy profiles.
A scheduler must make a decision using estimates before the actual outcome is known.

This project asks: **How should an inference request be assigned to a device and model when the
scheduler must consider a relative deadline, minimum model accuracy, current queues, network delay,
and estimated energy?** EdgeWeaver implements one transparent answer: filter incompatible or
insufficiently accurate candidates, predict completion, use the lowest estimated-energy candidate
among deadline-feasible choices, and adapt execution estimates from completed observations.

The contribution is the full research workflow around that policy: reproducible dataset handling,
real model training and local profiling, a scheduler-agnostic simulator, common candidate estimates,
controlled paired scenarios, inspectable decisions/events, deterministic metrics, resumable
experiments, ablations, figures, and evidence-based findings.

## 3. Motivation and related work

Edge scheduling commonly separates simple baselines from state-aware heuristics. Round-robin
dispatch offers predictability but ignores heterogeneity. Fastest-resource selection exploits
isolated service speed but can ignore transfer and queue costs. Completion-time heuristics incorporate
observable waiting and execution estimates, while deadline-aware policies introduce feasibility
constraints. Adaptive policies update estimates when observed performance departs from a static
profile.

EdgeWeaver deliberately stays within this interpretable design space. It is not a learned scheduler,
reinforcement-learning agent, optimizer, or production orchestrator. Its policy can be reconstructed
from each structured decision record, which is useful for a small empirical study where identifying
why a request was assigned matters as much as aggregate performance. UCI HAR was selected because it
represents sensor inference, supplies a fixed train/test split, and remains inexpensive enough for a
reproducible local workflow. The dataset source is the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones).

## 4. System model

The simulator contains exactly three configured single-capacity compute resources. Mobile has a 2.5
speed multiplier, 1.0 active-power unit, 20 waiting slots, and supports light/balanced models.
Gateway uses 1.3, 2.0, 40 slots, and all model roles. Edge server uses 0.6, 4.0, 80 slots, and all
roles. These values are editable simulation assumptions.

For a model with measured local mean latency `L`, service duration is `L × device speed multiplier`.
Remote transfers use `base latency + input bits / bandwidth`. Mobile-to-gateway is configured at 12
ms and 20 Mbps; mobile-to-edge-server at 35 ms and 10 Mbps. Upload and return are modeled separately;
local mobile execution has zero network delay. End-to-end latency is upload, queue wait, inference,
return, and scheduler overhead. A request meets its relative deadline when completion is no later than
arrival plus deadline.

Compute energy is `active power units × inference milliseconds`. Network energy is transferred
kilobytes times a configurable coefficient. Their sum is **estimated energy in normalized units**,
not joules. SimPy advances virtual time only; no wall-clock sleeping or real networking occurs.

Requests reference real held-out UCI HAR feature-vector IDs. The selected saved estimator generates
the actual class prediction, which is compared with the trace's true label. Structured events cover
arrival, decision, queueing, inference, return, deadline misses, slowdown transitions, rejection, and
adaptive profile updates. Per-request results retain all actual and predicted timing components.

## 5. Scheduling policies

All four policies implement one common selection interface and consume identical read-only state.

**Round Robin** rotates through compatible devices. On the selected device it uses the fastest
accuracy-eligible model. It intentionally ignores congestion and network conditions when choosing
the device.

**Fastest Device** chooses the device with the lowest isolated inference duration, then the
highest-measured-accuracy compatible model on that device. It intentionally ignores current queue
length and end-to-end network cost.

**Minimum Completion Time** evaluates every compatible, accuracy-eligible, queue-admissible pair.
It predicts upload, existing active/reserved/queued work, inference, return, and overhead, then selects
the earliest absolute completion. If no candidate is predicted to meet the deadline, it still chooses
the earliest.

**EdgeWeaver** uses the same candidates and timing model. If one or more candidates are predicted to
meet the deadline, it selects minimum estimated energy, breaking ties by earlier completion and stable
IDs. Otherwise it falls back to earliest completion. After actual inference completes, it updates only
the selected device/model estimate using an EWMA with `alpha=0.2`. The observation is service time,
not total latency. Updates occur after completion, do not mutate measured profiles, and cannot affect
the decision that produced the observation.

## 6. Dataset and model profiles

The official UCI HAR split contains 7,352 training and 2,947 test samples with 561 features and six
activities. Logistic regression and MLP use a fitted `StandardScaler`; random forest uses an identity
transform. Every preprocessor and model is saved and reloaded for prediction verification.

| Role / estimator | Accuracy | Macro F1 | Model size | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|
| Light / logistic regression | 95.49% | 95.48% | 25,661 B | 0.246 ms | 0.336 ms |
| Balanced / random forest | 92.84% | 92.64% | 2,568,840 B | 12.660 ms | 14.896 ms |
| Heavy / MLP | 94.57% | 94.58% | 1,858,312 B | 0.336 ms | 0.372 ms |

Each latency profile comprises 500 single-sample predictions following 20 warm-ups and includes
preprocessing plus prediction. The timer was `perf_counter_ns`; raw observations and environment
metadata are retained. Accuracy, F1, size, and these local latencies were physically measured. All
three device performances derived from them are simulated. The unexpected accuracy ordering—light,
heavy, then balanced—was retained without manipulation.

## 7. Experimental methodology

Four 20-second scenarios were declared in YAML. Normal uses stable 4 requests/s. Bursty uses 1.5
requests/s background traffic with three 1.5-second intervals at 16 requests/s. Network Slowdown uses
normal traffic while the edge-server link has triple base latency and quarter bandwidth from 7–14
seconds. Device Slowdown uses 5 requests/s while edge-server service time is tripled over the same
half-open window. Slowdown state is visible only when it becomes current; no scheduler sees the
future schedule.

Each request samples a held-out example, relative deadline, minimum accuracy, and input size with a
local seeded RNG. The trace is generated before scheduler selection. For each scenario and seed, all
four policies consume the exact same trace ID and SHA-256. Seeds 1–5 give a 4 × 4 × 5 core matrix of
80 runs.

Per-run metrics are deadline satisfaction, useful goodput, mean/P95 end-to-end latency, actual
request accuracy, estimated normalized energy per completion, per-device compute utilization,
assigned-request model distribution, and deterministic deadline-miss causes. Goodput counts correct,
on-time, accuracy-eligible completions per simulated second. Misses are attributed to an inference
underestimate only when that underestimate alone changes the deadline outcome; otherwise the largest
network, queue, or inference component wins.

Two five-seed ablations reuse core traces. Bursty “no model switching” forces the most accurate
qualifying model but preserves the rest of EdgeWeaver. Device Slowdown “no online update” disables
EWMA while preserving normal selection. Results are aggregated using arithmetic means and sample
standard deviations; individual seed values remain in tidy tables. No significance tests or
post-hoc seed selection were performed.

## 8. Results

All 80 core runs completed, covering 7,823 requests across core and ablation runs with no execution
rejections. The table reports core means across five seeds. Energy is estimated normalized units per
completed request.

| Scenario | Policy | Deadline | Goodput (/s) | Mean / P95 latency (ms) | Est. energy |
|---|---|---:|---:|---:|---:|
| Normal | Round Robin | 100.00% | 3.610 | 33.240 / 73.789 | 0.674 |
| Normal | Fastest Device | 100.00% | 3.610 | 73.789 / 73.789 | 0.679 |
| Normal | MCT | 100.00% | 3.610 | 0.616 / 0.615 | 0.615 |
| Normal | EdgeWeaver | 100.00% | 3.610 | 0.616 / 0.615 | 0.615 |
| Bursty | Round Robin | 100.00% | 4.570 | 33.263 / 73.789 | 0.674 |
| Bursty | Fastest Device | 100.00% | 4.570 | 73.790 / 73.789 | 0.679 |
| Bursty | MCT | 100.00% | 4.570 | 0.619 / 0.615 | 0.615 |
| Bursty | EdgeWeaver | 100.00% | 4.570 | 0.619 / 0.615 | 0.615 |
| Network slowdown | Round Robin | 93.66% | 3.390 | 50.257 / 224.714 | 0.674 |
| Network slowdown | Fastest Device | 79.90% | 2.890 | 124.231 / 224.714 | 0.679 |
| Network slowdown | MCT | 100.00% | 3.610 | 0.616 / 0.615 | 0.615 |
| Network slowdown | EdgeWeaver | 100.00% | 3.610 | 0.616 / 0.615 | 0.615 |
| Device slowdown | Round Robin | 100.00% | 4.710 | 33.393 / 74.084 | 0.804 |
| Device slowdown | Fastest Device | 100.00% | 4.710 | 73.886 / 74.084 | 1.068 |
| Device slowdown | MCT | 100.00% | 4.710 | 0.617 / 0.615 | 0.615 |
| Device slowdown | EdgeWeaver | 100.00% | 4.710 | 0.617 / 0.615 | 0.615 |

Actual accuracy was policy-independent because all assignments used logistic regression: 95.99% in
Normal and Network Slowdown, 96.42% in Bursty, and 96.30% in Device Slowdown. Model-selection output
was 100% logistic regression for every policy/scenario. MCT and EdgeWeaver consequently made the same
local mobile assignment throughout this experiment.

Compute utilization was very low because measured inference durations were sub-millisecond for the
selected model and arrival rates were modest relative to capacity. The highest five-seed mean cell
was about 0.301% for mobile under MCT/EdgeWeaver Device Slowdown. This scale is central to
interpreting the burst and slowdown results: the configured system was not compute-saturated.

The figures report mean ± sample standard deviation and retain the full scenario/policy comparison:

- [Deadline satisfaction](../experiments/phase7/figures/deadline_satisfaction.png)
- [Useful goodput](../experiments/phase7/figures/useful_goodput.png)
- [Mean and P95 latency](../experiments/phase7/figures/latency.png)
- [Actual prediction accuracy](../experiments/phase7/figures/actual_prediction_accuracy.png)
- [Estimated normalized energy](../experiments/phase7/figures/estimated_energy.png)
- [Device utilization](../experiments/phase7/figures/device_utilization_heatmap.png)
- [Model selection](../experiments/phase7/figures/model_selection_distribution.png)
- [EdgeWeaver ablations](../experiments/phase7/figures/edgeweaver_ablations.png)

**Hypothesis assessment.** H1 was not supported: Fastest Device remained at 100% deadline
satisfaction, burst mean latency rose only 0.00043 ms, and maximum observed queue wait was 0.110 ms.
H2 was partially supported: MCT exceeded Round Robin by 6.34 percentage points only under Network
Slowdown and tied elsewhere. H3 was inconclusive because the no-switch variant and full EdgeWeaver
realized identical model choices. H4 was also inconclusive against the planned accuracy-first
ablation because both variants used identical models, energy, and accuracy. H5 was inconclusive
because EdgeWeaver selected mobile rather than the slowed edge server, so its adaptive treatment was
not exercised. These labels describe effect evidence, not statistical significance.

## 9. EdgeWeaver ablations

Under Bursty Load, full and no-model-switching EdgeWeaver each achieved 100% deadline satisfaction,
4.570 requests/s useful goodput, 0.615 estimated normalized energy units per completion, and 96.42%
actual accuracy. Both selected logistic regression for 100% of assignments. Since logistic
regression was already the most accurate model, the ablation did not change treatment; the study
cannot estimate a model-switching contribution from this configuration.

Under Device Slowdown, full and no-online-update EdgeWeaver each achieved 100% deadline satisfaction,
0.617 ms mean latency, and 4.710 requests/s useful goodput. Both executed locally on mobile while the
scenario slowed the edge server. The EWMA mechanism is verified by controlled unit/integration
fixtures, but this ablation supplies no empirical estimate of its benefit in the core workload.

## 10. Failure analysis

The real extracted network case is `core-network_slowdown-fastest_device-seed-1`, request
`request-00028`. Fastest Device selected edge-server/logistic regression. Its 224.566 ms network time
dominated 0.148 ms inference, producing completion at 7,637.043 ms against a 7,592.329 ms absolute
deadline. Mobile and gateway candidates were predicted feasible at decision time; the policy ignored
that end-to-end advantage by design.

The requested Fastest Device overload case did not occur materially. The largest burst queue wait
was only 0.110 ms (`core-bursty-fastest_device-seed-5`, `request-00080`), 0.14% of its 80 ms relative
deadline, and did not cause a miss. No miss attributable to a stale execution estimate occurred. No
request lacked a deadline-feasible candidate. A prior candidate match for “lighter model selected to
satisfy a deadline” was rejected during final review because slower models remained feasible on other
devices; no genuine deadline-driven model switch occurred. The machine-readable case file marks each
absent case explicitly.

## 11. Threats to validity

**Construct validity.** Overall test accuracy is used as the eligibility score for individual
requests even though per-request correctness is unknown until prediction. Normalized energy combines
simple power multipliers and data transfer, so it measures the configured cost model rather than
physical electrical consumption. Deadline-miss cause uses a deterministic primary-cause rule that
reduces interacting components to one label.

**Internal validity.** Paired traces, hashes, seeded randomness, immutable profiles, fresh scheduler
state, and deterministic SimPy execution control many confounders. Nevertheless, the measured model
ordering collapsed the intended model-choice contrast, and EdgeWeaver avoided the device targeted by
the slowdown. H3–H5 therefore lack treatment variation. Five seeds limit precision, and no inferential
statistics were performed.

**External validity.** UCI HAR is one tabular sensor domain. A single Windows computer supplied all
physical latency measurements. Device multipliers, network conditions, request rates, deadlines, and
power units are design assumptions. Results may change with a domain where accuracy/compute trade-offs
are stronger or with workloads that saturate compute.

**Implementation validity.** The simulator has deterministic tests for timing, queueing, energy,
compatibility, schedulers, adaptation, slowdowns, metrics, and replay, plus an end-to-end fixture.
Passing tests does not reproduce operating-system scheduling, caches, thermal throttling, device
parallelism, transport behavior, or all hardware effects.

## 12. Limitations

1. Only one physical computer was profiled.
2. Mobile, gateway, and edge-server speeds are simulated using scaling factors.
3. Energy values are estimated normalized units, not measured electricity or joules.
4. Overall model accuracy is an approximation for individual-request scheduling eligibility.
5. Network behavior is simplified and symmetric, without packet or transport effects.
6. UCI HAR represents only one ML domain.
7. The simulator does not reproduce all operating-system, concurrency, thermal, or hardware effects.
8. Five seeds provide a limited sample for variability estimates.
9. The results do not prove that EdgeWeaver works on production edge infrastructure.

## 13. Conclusion

EdgeWeaver demonstrates a complete, inspectable workflow for evaluating deadline-, accuracy-, and
estimated-energy-aware inference scheduling. The experiment strongly exposed the danger of ignoring
network conditions: under edge-link slowdown, Fastest Device lost 20.10 percentage points of deadline
satisfaction while MCT and EdgeWeaver retained 100%. It did not demonstrate the planned benefits of
model switching or online adaptation because the measured logistic model dominated choices and the
adaptive policy avoided the slowed device.

That outcome is scientifically useful. It shows that scheduler claims depend on realized model and
workload trade-offs, not policy intent alone. The repository preserves the measurements, paired
traces, all seeds, structured decisions, raw results, aggregates, and absent-case evidence needed to
inspect that conclusion. A future presentation layer can expose these artifacts, but the non-frontend
study is already reproducible from the command line.
