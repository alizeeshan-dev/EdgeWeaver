# Phase 7 Findings Summary

Evidence labels describe these simulated runs only; no significance tests were performed.

## H1 — not supported

Fastest Device deadline satisfaction was 100.00% under normal load and 100.00% under bursty load; mean latency was 73.7892 ms and 73.7896 ms.
Edge-server compute utilization was 0.06% under normal load and 0.07% under bursty load. A representative burst queue case was not observed.

## H2 — partially supported

MCT minus Round Robin: Normal +0.00 percentage points, Bursty +0.00 percentage points, Network slowdown +6.34 percentage points, Device slowdown +0.00 percentage points.

## H3 — inconclusive (ablation did not change realized model choices)

Bursty deadline satisfaction: full EdgeWeaver 100.00%; no-model-switching 100.00%. Useful goodput was 4.57 and 4.57 requests/s, respectively.

## H4 — inconclusive (accuracy-first ablation selected the same models)

Bursty estimated energy (normalized units/request): full EdgeWeaver 0.6150155; accuracy-first ablation 0.6150155. Actual accuracy was 96.42% and 96.42%. No separate aggregate acceptability threshold or significance test was imposed for H4.

## H5 — inconclusive (EdgeWeaver did not use the slowed device)

Device-slowdown deadline satisfaction: full EdgeWeaver 100.00%; no-online-update 100.00%. Mean latency was 0.6172 ms versus 0.6172 ms; useful goodput was 4.7100 versus 4.7100 requests/s.

## Representative cases

1/5 requested case types were observed in the saved structured outputs. Unobserved types remain explicitly marked in `failure_cases.json`.
