"""Research figures, representative cases, and cautious findings from saved results.

This module is deliberately downstream of simulation and aggregation: every output is
derived from immutable per-run metrics and structured run results.  It never reruns a
simulation or fills in an unobserved result.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

# Headless analysis should not depend on a writable user-profile directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "edgeweaver-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.figure import Figure
from pydantic import Field

from edgeweaver.domain import (
    CandidateEstimate,
    DomainModel,
    EventType,
    RequestExecutionResult,
    RequestStatus,
    SimulationRunResult,
)
from edgeweaver.metrics import DeadlineMissCause, PerRunMetrics, classify_deadline_miss
from edgeweaver.workloads import WorkloadTrace

CORE_SCHEDULERS = ("round_robin", "fastest_device", "min_completion", "edgeweaver")
CORE_SCENARIOS = ("normal", "bursty", "network_slowdown", "device_slowdown")
ABLATION_VARIANTS = ("edgeweaver_no_model_switching", "edgeweaver_no_online_update")

_SCHEDULER_LABELS = {
    "round_robin": "Round Robin",
    "fastest_device": "Fastest Device",
    "min_completion": "Minimum Completion Time",
    "edgeweaver": "EdgeWeaver",
    "edgeweaver_no_model_switching": "EdgeWeaver: no model switching",
    "edgeweaver_no_online_update": "EdgeWeaver: no online update",
}
_SCENARIO_LABELS = {
    "normal": "Normal",
    "bursty": "Bursty",
    "network_slowdown": "Network slowdown",
    "device_slowdown": "Device slowdown",
}
_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


class ReportingRun(DomainModel):
    """A structured run joined to the scheduler-independent trace it consumed."""

    scenario_id: str = Field(min_length=1)
    run: SimulationRunResult
    trace: WorkloadTrace


class FailureCaseEventReference(DomainModel):
    sequence: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0.0)
    event_type: EventType


class FailureCaseEvidence(DomainModel):
    run_id: str
    scenario_id: str
    scheduler_name: str
    seed: int
    request_id: str
    request_deadline_ms: float
    absolute_deadline_ms: float
    minimum_accuracy: float
    selected_device_id: str
    selected_model_id: str
    predicted_completion_ms: float
    actual_completion_ms: float | None
    deadline_met: bool
    decision_reason: str
    actual_queue_wait_ms: float
    actual_network_time_ms: float
    actual_inference_time_ms: float
    candidates: tuple[CandidateEstimate, ...]
    event_references: tuple[FailureCaseEventReference, ...]


class RepresentativeCase(DomainModel):
    case_id: str
    title: str
    observed: bool
    evidence: FailureCaseEvidence | None = None
    note: str


class FailureCaseReport(DomainModel):
    format_version: Literal["edgeweaver-failure-cases-v1"] = "edgeweaver-failure-cases-v1"
    cases: list[RepresentativeCase]


def load_run_metrics(paths: Iterable[Path]) -> list[PerRunMetrics]:
    """Load Phase 6 per-run JSON summaries in stable path order."""

    return [
        PerRunMetrics.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
        for path in sorted(paths)
    ]


def load_reporting_run(*, run_path: Path, trace_path: Path, scenario_id: str) -> ReportingRun:
    """Load a raw run and its trace for evidence extraction."""

    return ReportingRun(
        scenario_id=scenario_id,
        run=SimulationRunResult.model_validate_json(
            run_path.read_text(encoding="utf-8"), strict=True
        ),
        trace=WorkloadTrace.model_validate_json(
            trace_path.read_text(encoding="utf-8"), strict=True
        ),
    )


def load_reporting_runs_from_experiment(
    *, project_root: Path, run_records_directory: Path
) -> list[ReportingRun]:
    """Join raw runs to traces using completed-run records from the Phase 7 runner."""

    resolved_root = project_root.resolve()
    reporting_runs: list[ReportingRun] = []
    for record_path in sorted(run_records_directory.glob("*.json")):
        payload: object = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"completion record must be an object: {record_path}")
        scenario_id = payload.get("scenario_id")
        trace_reference = payload.get("trace_reference")
        artifact_references = payload.get("artifact_references")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"completion record has no scenario_id: {record_path}")
        if not isinstance(trace_reference, str) or not trace_reference:
            raise ValueError(f"completion record has no trace_reference: {record_path}")
        if not isinstance(artifact_references, dict):
            raise ValueError(f"completion record has no artifact_references: {record_path}")
        raw_reference = artifact_references.get("raw_run")
        if not isinstance(raw_reference, str) or not raw_reference:
            raise ValueError(f"completion record has no raw_run reference: {record_path}")

        def resolve(reference: str) -> Path:
            path = Path(reference)
            return path if path.is_absolute() else resolved_root / path

        reporting_runs.append(
            load_reporting_run(
                run_path=resolve(raw_reference),
                trace_path=resolve(trace_reference),
                scenario_id=scenario_id,
            )
        )
    return reporting_runs


def _ordered_present(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    present = set(values)
    return [value for value in preferred if value in present] + sorted(present - set(preferred))


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _values(
    summaries: Sequence[PerRunMetrics],
    scenario: str,
    scheduler: str,
    getter: Callable[[PerRunMetrics], float],
) -> list[float]:
    return [
        getter(summary)
        for summary in summaries
        if summary.scenario_id == scenario and summary.scheduler_name == scheduler
    ]


def _save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", metadata={"Software": "EdgeWeaver"})
    plt.close(figure)


def _grouped_metric_figure(
    summaries: Sequence[PerRunMetrics],
    *,
    getter: Callable[[PerRunMetrics], float],
    scale: float,
    title: str,
    y_label: str,
    path: Path,
) -> None:
    scenarios = _ordered_present((item.scenario_id for item in summaries), CORE_SCENARIOS)
    schedulers = _ordered_present((item.scheduler_name for item in summaries), CORE_SCHEDULERS)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    x_positions = np.arange(len(scenarios), dtype=float)
    width = 0.8 / max(len(schedulers), 1)
    for index, scheduler in enumerate(schedulers):
        means: list[float] = []
        errors: list[float] = []
        for scenario in scenarios:
            mean, std = _mean_std(_values(summaries, scenario, scheduler, getter))
            means.append(mean * scale)
            errors.append(std * scale)
        offset = (index - (len(schedulers) - 1) / 2.0) * width
        axis.bar(
            x_positions + offset,
            means,
            width,
            yerr=errors,
            capsize=3,
            label=_SCHEDULER_LABELS.get(scheduler, scheduler),
            color=_COLORS[index % len(_COLORS)],
        )
    seed_count = len({summary.seed for summary in summaries})
    axis.set_title(f"{title} (mean ± SD; {seed_count} seeds)")
    axis.set_ylabel(y_label)
    axis.set_xticks(x_positions, [_SCENARIO_LABELS.get(item, item) for item in scenarios])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    _save_figure(figure, path)


def _latency_figure(summaries: Sequence[PerRunMetrics], path: Path) -> None:
    scenarios = _ordered_present((item.scenario_id for item in summaries), CORE_SCENARIOS)
    schedulers = _ordered_present((item.scheduler_name for item in summaries), CORE_SCHEDULERS)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
    metrics = (
        ("Mean end-to-end latency", lambda item: item.mean_end_to_end_latency_ms),
        ("P95 end-to-end latency", lambda item: item.p95_end_to_end_latency_ms),
    )
    positions = np.arange(len(scenarios), dtype=float)
    width = 0.8 / max(len(schedulers), 1)
    for axis, (metric_name, getter) in zip(axes, metrics, strict=True):
        for index, scheduler in enumerate(schedulers):
            stats = [
                _mean_std(_values(summaries, scenario, scheduler, getter)) for scenario in scenarios
            ]
            offset = (index - (len(schedulers) - 1) / 2.0) * width
            axis.bar(
                positions + offset,
                [item[0] for item in stats],
                width,
                yerr=[item[1] for item in stats],
                capsize=2,
                color=_COLORS[index % len(_COLORS)],
                label=_SCHEDULER_LABELS.get(scheduler, scheduler),
            )
        axis.set_title(metric_name)
        axis.set_yscale("log")
        axis.set_ylabel("End-to-end latency (ms, log scale)")
        axis.set_xticks(positions, [_SCENARIO_LABELS.get(item, item) for item in scenarios])
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    seed_count = len({summary.seed for summary in summaries})
    figure.suptitle(f"Simulation latency (mean ± SD across {seed_count} seeds)")
    axes[0].legend(fontsize=8, ncols=2)
    _save_figure(figure, path)


def _device_utilization_figure(summaries: Sequence[PerRunMetrics], path: Path) -> None:
    scenarios = _ordered_present((item.scenario_id for item in summaries), CORE_SCENARIOS)
    schedulers = _ordered_present((item.scheduler_name for item in summaries), CORE_SCHEDULERS)
    devices = sorted(
        {metric.device_id for summary in summaries for metric in summary.device_utilization}
    )
    rows = [(scenario, scheduler) for scenario in scenarios for scheduler in schedulers]
    matrix = np.full((len(rows), len(devices)), np.nan)
    for row_index, (scenario, scheduler) in enumerate(rows):
        matching = [
            summary
            for summary in summaries
            if summary.scenario_id == scenario and summary.scheduler_name == scheduler
        ]
        for column_index, device in enumerate(devices):
            values = [
                metric.utilization * 100.0
                for summary in matching
                for metric in summary.device_utilization
                if metric.device_id == device
            ]
            if values:
                matrix[row_index, column_index] = statistics.fmean(values)
    figure, axis = plt.subplots(figsize=(8.5, max(5.5, len(rows) * 0.36)))
    observed_max = float(np.nanmax(matrix))
    color_max = max(observed_max, 1e-9)
    image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=color_max, cmap="YlGnBu")
    axis.set_xticks(range(len(devices)), devices)
    axis.set_yticks(
        range(len(rows)),
        [
            f"{_SCENARIO_LABELS.get(scenario, scenario)} — "
            f"{_SCHEDULER_LABELS.get(scheduler, scheduler)}"
            for scenario, scheduler in rows
        ],
        fontsize=8,
    )
    seed_count = len({summary.seed for summary in summaries})
    axis.set_title(f"Device compute utilization (mean across {seed_count} seeds)")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Compute busy time / simulation duration (%)")
    for row_index in range(len(rows)):
        for column_index in range(len(devices)):
            value = matrix[row_index, column_index]
            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > color_max * 0.55 else "black",
                )
    _save_figure(figure, path)


def _model_selection_figure(summaries: Sequence[PerRunMetrics], path: Path) -> None:
    scenarios = _ordered_present((item.scenario_id for item in summaries), CORE_SCENARIOS)
    schedulers = _ordered_present((item.scheduler_name for item in summaries), CORE_SCHEDULERS)
    rows = [(scenario, scheduler) for scenario in scenarios for scheduler in schedulers]
    roles = ("light", "balanced", "heavy")
    role_colors = {"light": "#72B7B2", "balanced": "#F2CF5B", "heavy": "#B279A2"}
    figure, axis = plt.subplots(figsize=(11, max(5.5, len(rows) * 0.36)))
    left = np.zeros(len(rows), dtype=float)
    for role in roles:
        percentages: list[float] = []
        for scenario, scheduler in rows:
            matching = [
                summary
                for summary in summaries
                if summary.scenario_id == scenario and summary.scheduler_name == scheduler
            ]
            values = [
                metric.percentage
                for summary in matching
                for metric in summary.model_selection_distribution
                if metric.model_role is not None and metric.model_role.value == role
            ]
            percentages.append(statistics.fmean(values) if values else 0.0)
        axis.barh(
            range(len(rows)),
            percentages,
            left=left,
            label=role.title(),
            color=role_colors[role],
        )
        left += np.asarray(percentages)
    axis.set_yticks(
        range(len(rows)),
        [
            f"{_SCENARIO_LABELS.get(scenario, scenario)} — "
            f"{_SCHEDULER_LABELS.get(scheduler, scheduler)}"
            for scenario, scheduler in rows
        ],
        fontsize=8,
    )
    axis.set_xlim(0.0, 100.0)
    axis.set_xlabel("Assigned requests (%)")
    seed_count = len({summary.seed for summary in summaries})
    axis.set_title(f"Model-selection distribution (mean across {seed_count} seeds)")
    axis.legend(ncols=3)
    axis.grid(axis="x", alpha=0.2)
    _save_figure(figure, path)


def _ablation_figure(summaries: Sequence[PerRunMetrics], path: Path) -> None:
    comparisons: tuple[
        tuple[str, str, tuple[tuple[str, str, Callable[[PerRunMetrics], float], float], ...]], ...
    ] = (
        (
            "bursty",
            "edgeweaver_no_model_switching",
            (
                ("Deadline satisfaction", "%", lambda item: item.deadline_satisfaction_rate, 100.0),
                (
                    "Useful goodput",
                    "requests/s",
                    lambda item: item.useful_goodput_requests_per_second,
                    1.0,
                ),
                (
                    "Estimated energy",
                    "normalized units/request",
                    lambda item: item.estimated_energy_per_completed_request_units,
                    1.0,
                ),
            ),
        ),
        (
            "device_slowdown",
            "edgeweaver_no_online_update",
            (
                ("Deadline satisfaction", "%", lambda item: item.deadline_satisfaction_rate, 100.0),
                (
                    "Mean end-to-end latency",
                    "ms",
                    lambda item: item.mean_end_to_end_latency_ms,
                    1.0,
                ),
                (
                    "Useful goodput",
                    "requests/s",
                    lambda item: item.useful_goodput_requests_per_second,
                    1.0,
                ),
            ),
        ),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    for row, (scenario, variant, metrics) in enumerate(comparisons):
        for column, (title, unit, getter, scale) in enumerate(metrics):
            axis = axes[row, column]
            labels = ("edgeweaver", variant)
            stats = [
                _mean_std(_values(summaries, scenario, scheduler, getter)) for scheduler in labels
            ]
            if any(math.isnan(mean) for mean, _ in stats):
                axis.text(
                    0.5,
                    0.5,
                    "Required ablation result not available",
                    ha="center",
                    va="center",
                )
                axis.set_axis_off()
                continue
            axis.bar(
                range(2),
                [mean * scale for mean, _ in stats],
                yerr=[std * scale for _, std in stats],
                capsize=4,
                color=("#E45756", "#9D755D"),
            )
            axis.set_xticks(
                range(2),
                ["Full EdgeWeaver", _SCHEDULER_LABELS[variant].replace("EdgeWeaver: ", "")],
                rotation=8,
            )
            axis.tick_params(axis="x", labelsize=8)
            axis.set_title(f"{_SCENARIO_LABELS[scenario]}: {title}")
            axis.set_ylabel(unit)
            axis.grid(axis="y", alpha=0.25)
    seed_count = len({summary.seed for summary in summaries})
    figure.suptitle(f"EdgeWeaver ablations (mean ± SD; {seed_count} seeds)")
    _save_figure(figure, path)


def generate_research_figures(
    summaries: Sequence[PerRunMetrics],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate the eight required chart groups solely from saved per-run metrics."""

    core = [summary for summary in summaries if summary.scheduler_name in CORE_SCHEDULERS]
    if not core:
        raise ValueError("at least one core scheduler summary is required")
    paths = {
        "deadline_satisfaction": output_dir / "deadline_satisfaction.png",
        "useful_goodput": output_dir / "useful_goodput.png",
        "latency": output_dir / "latency.png",
        "actual_prediction_accuracy": output_dir / "actual_prediction_accuracy.png",
        "estimated_energy": output_dir / "estimated_energy.png",
        "device_utilization": output_dir / "device_utilization_heatmap.png",
        "model_selection": output_dir / "model_selection_distribution.png",
        "edgeweaver_ablations": output_dir / "edgeweaver_ablations.png",
    }
    _grouped_metric_figure(
        core,
        getter=lambda item: item.deadline_satisfaction_rate,
        scale=100.0,
        title="Deadline satisfaction by scheduler and scenario",
        y_label="Completed requests meeting deadline (%)",
        path=paths["deadline_satisfaction"],
    )
    _grouped_metric_figure(
        core,
        getter=lambda item: item.useful_goodput_requests_per_second,
        scale=1.0,
        title="Useful goodput by scheduler and scenario",
        y_label="Valid, correct, on-time predictions (requests/s)",
        path=paths["useful_goodput"],
    )
    _latency_figure(core, paths["latency"])
    _grouped_metric_figure(
        core,
        getter=lambda item: item.actual_prediction_accuracy,
        scale=100.0,
        title="Actual prediction accuracy on held-out requests",
        y_label="Correct completed predictions (%)",
        path=paths["actual_prediction_accuracy"],
    )
    _grouped_metric_figure(
        core,
        getter=lambda item: item.estimated_energy_per_completed_request_units,
        scale=1.0,
        title="Estimated energy per completed request",
        y_label="Estimated energy (normalized units/request)",
        path=paths["estimated_energy"],
    )
    _device_utilization_figure(core, paths["device_utilization"])
    _model_selection_figure(core, paths["model_selection"])
    _ablation_figure(summaries, paths["edgeweaver_ablations"])
    return paths


def _evidence(reporting_run: ReportingRun, result: RequestExecutionResult) -> FailureCaseEvidence:
    request = next(
        item for item in reporting_run.trace.requests if item.request_id == result.request_id
    )
    references = tuple(
        FailureCaseEventReference(
            sequence=event.sequence,
            timestamp_ms=event.timestamp_ms,
            event_type=event.event_type,
        )
        for event in reporting_run.run.events
        if event.request_id == result.request_id
        or (
            reporting_run.scenario_id == "network_slowdown"
            and event.event_type
            in {EventType.NETWORK_SLOWDOWN_STARTED, EventType.NETWORK_SLOWDOWN_ENDED}
        )
        or (
            reporting_run.scenario_id == "device_slowdown"
            and event.event_type
            in {EventType.DEVICE_SLOWDOWN_STARTED, EventType.DEVICE_SLOWDOWN_ENDED}
        )
    )
    return FailureCaseEvidence(
        run_id=reporting_run.run.run_id,
        scenario_id=reporting_run.scenario_id,
        scheduler_name=reporting_run.run.scheduler_name,
        seed=reporting_run.run.random_seed,
        request_id=result.request_id,
        request_deadline_ms=request.deadline_ms,
        absolute_deadline_ms=result.absolute_deadline_ms,
        minimum_accuracy=request.minimum_accuracy,
        selected_device_id=result.assignment.device_id,
        selected_model_id=result.assignment.model_id,
        predicted_completion_ms=result.assignment.predicted_completion_ms,
        actual_completion_ms=result.completion_time_ms,
        deadline_met=result.deadline_met,
        decision_reason=result.assignment.decision_reason,
        actual_queue_wait_ms=result.queue_wait_ms,
        actual_network_time_ms=result.upload_time_ms + result.return_time_ms,
        actual_inference_time_ms=result.inference_time_ms,
        candidates=result.assignment.candidates,
        event_references=references,
    )


def _case(
    case_id: str,
    title: str,
    match: tuple[ReportingRun, RequestExecutionResult] | None,
    missing_note: str,
) -> RepresentativeCase:
    return RepresentativeCase(
        case_id=case_id,
        title=title,
        observed=match is not None,
        evidence=None if match is None else _evidence(*match),
        note="Observed in saved structured run output." if match is not None else missing_note,
    )


def extract_failure_cases(runs: Sequence[ReportingRun]) -> FailureCaseReport:
    """Select deterministic representative evidence without manufacturing absent cases."""

    ordered: list[tuple[ReportingRun, RequestExecutionResult]] = sorted(
        (
            (reporting_run, result)
            for reporting_run in runs
            for result in reporting_run.run.requests
        ),
        key=lambda item: (
            item[0].scenario_id,
            item[0].run.scheduler_name,
            item[0].run.random_seed,
            item[1].request_id,
        ),
    )

    burst_candidates = [
        item
        for item in ordered
        if item[0].scenario_id == "bursty"
        and item[0].run.scheduler_name == "fastest_device"
        and item[1].assignment.device_id == "edge-server"
        and item[1].status is RequestStatus.COMPLETED
        and item[1].queue_wait_ms
        >= next(
            request.deadline_ms * 0.05
            for request in item[0].trace.requests
            if request.request_id == item[1].request_id
        )
    ]
    burst = max(burst_candidates, key=lambda item: item[1].queue_wait_ms, default=None)

    network_candidates = [
        item
        for item in ordered
        if item[0].scenario_id == "network_slowdown"
        and item[1].status is RequestStatus.COMPLETED
        and not item[1].deadline_met
        and item[1].assignment.device_id
        != next(
            request.source_device_id
            for request in item[0].trace.requests
            if request.request_id == item[1].request_id
        )
        and classify_deadline_miss(item[1]) is DeadlineMissCause.NETWORK_DELAY
    ]
    network = max(
        network_candidates,
        key=lambda item: item[1].upload_time_ms + item[1].return_time_ms,
        default=None,
    )

    stale = next(
        (
            item
            for item in ordered
            if item[0].scenario_id == "device_slowdown"
            and item[1].status is RequestStatus.COMPLETED
            and not item[1].deadline_met
            and classify_deadline_miss(item[1]) is DeadlineMissCause.INCORRECT_STATIC_ESTIMATE
        ),
        None,
    )

    switched: tuple[ReportingRun, RequestExecutionResult] | None = None
    for item in ordered:
        result = item[1]
        if (
            item[0].run.scheduler_name != "edgeweaver"
            or not result.assignment.expected_to_meet_deadline
        ):
            continue
        selected = next(
            (
                candidate
                for candidate in result.assignment.candidates
                if candidate.device_id == result.assignment.device_id
                and candidate.model_id == result.assignment.model_id
            ),
            None,
        )
        if selected is None:
            continue
        higher_compute_models = {
            candidate.model_id
            for candidate in result.assignment.candidates
            if candidate.eligible
            and candidate.model_id != selected.model_id
            and candidate.predicted_inference_time_ms > selected.predicted_inference_time_ms
        }
        slower_infeasible = any(
            not any(
                candidate.eligible
                and candidate.model_id == model_id
                and candidate.expected_to_meet_deadline
                for candidate in result.assignment.candidates
            )
            for model_id in higher_compute_models
        )
        if slower_infeasible:
            switched = item
            break

    impossible = next(
        (
            item
            for item in ordered
            if not item[1].assignment.expected_to_meet_deadline
            and any(candidate.eligible for candidate in item[1].assignment.candidates)
            and not any(
                candidate.eligible and candidate.expected_to_meet_deadline
                for candidate in item[1].assignment.candidates
            )
        ),
        None,
    )

    return FailureCaseReport(
        cases=[
            _case(
                "fastest_device_burst_queue",
                "Fastest Device develops material queueing during a burst",
                burst,
                "No Fastest Device burst queue wait reached 5% of its request deadline.",
            ),
            _case(
                "network_slowdown_remote_miss",
                "Remote execution misses a deadline with network delay as the primary cause",
                network,
                "No remote network-slowdown miss classified as network delay was observed.",
            ),
            _case(
                "device_slowdown_stale_estimate",
                "A stale execution estimate contributes to a device-slowdown miss",
                stale,
                "No device-slowdown miss classified as incorrect static estimate was observed.",
            ),
            _case(
                "edgeweaver_smaller_model_for_deadline",
                "EdgeWeaver selects a lower-compute model while a slower model is infeasible",
                switched,
                "No qualifying EdgeWeaver model-switching case was observed.",
            ),
            _case(
                "no_assignment_meets_deadline",
                "No valid assignment is predicted to meet the deadline",
                impossible,
                "No request with valid candidates but no deadline-feasible assignment "
                "was observed.",
            ),
        ]
    )


def save_failure_cases(path: Path, report: FailureCaseReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_mean(
    summaries: Sequence[PerRunMetrics],
    scenario: str,
    scheduler: str,
    getter: Callable[[PerRunMetrics], float],
) -> float | None:
    values = _values(summaries, scenario, scheduler, getter)
    return statistics.fmean(values) if values else None


def _format_percent(value: float | None) -> str:
    return "not available" if value is None else f"{value * 100.0:.2f}%"


def _format_number(value: float | None) -> str:
    return "not available" if value is None else f"{value:.4f}"


def _device_utilization_mean(
    summaries: Sequence[PerRunMetrics], scenario: str, scheduler: str, device_id: str
) -> float | None:
    values = [
        metric.utilization
        for summary in summaries
        if summary.scenario_id == scenario and summary.scheduler_name == scheduler
        for metric in summary.device_utilization
        if metric.device_id == device_id
    ]
    return statistics.fmean(values) if values else None


def _model_selection_means(
    summaries: Sequence[PerRunMetrics], scenario: str, scheduler: str
) -> dict[str, float]:
    values_by_model: dict[str, list[float]] = {}
    for summary in summaries:
        if summary.scenario_id != scenario or summary.scheduler_name != scheduler:
            continue
        for metric in summary.model_selection_distribution:
            values_by_model.setdefault(metric.model_id, []).append(metric.percentage)
    return {
        model_id: round(statistics.fmean(values), 12)
        for model_id, values in sorted(values_by_model.items())
    }


def _case_observed(failure_cases: FailureCaseReport | None, case_id: str) -> bool | None:
    if failure_cases is None:
        return None
    match = next((case for case in failure_cases.cases if case.case_id == case_id), None)
    return None if match is None else match.observed


def generate_findings_summary(
    summaries: Sequence[PerRunMetrics],
    *,
    failure_cases: FailureCaseReport | None = None,
) -> str:
    """Generate a cautious H1–H5 summary from observed values only."""

    lines = [
        "# Phase 7 Findings Summary",
        "",
        "Evidence labels describe these simulated runs only; no significance tests were performed.",
        "",
    ]

    fd_normal_deadline = _metric_mean(
        summaries, "normal", "fastest_device", lambda item: item.deadline_satisfaction_rate
    )
    fd_burst_deadline = _metric_mean(
        summaries, "bursty", "fastest_device", lambda item: item.deadline_satisfaction_rate
    )
    fd_normal_latency = _metric_mean(
        summaries, "normal", "fastest_device", lambda item: item.mean_end_to_end_latency_ms
    )
    fd_burst_latency = _metric_mean(
        summaries, "bursty", "fastest_device", lambda item: item.mean_end_to_end_latency_ms
    )
    fd_normal_edge_util = _device_utilization_mean(
        summaries, "normal", "fastest_device", "edge-server"
    )
    fd_burst_edge_util = _device_utilization_mean(
        summaries, "bursty", "fastest_device", "edge-server"
    )
    queue_case = _case_observed(failure_cases, "fastest_device_burst_queue")
    queue_case_text = (
        "not evaluated" if queue_case is None else ("observed" if queue_case else "not observed")
    )
    if None in (fd_normal_deadline, fd_burst_deadline, fd_normal_latency, fd_burst_latency):
        h1 = "inconclusive"
    else:
        assert fd_normal_deadline is not None and fd_burst_deadline is not None
        assert fd_normal_latency is not None and fd_burst_latency is not None
        deadline_deterioration = fd_burst_deadline < fd_normal_deadline - 0.01
        latency_deterioration = fd_burst_latency > fd_normal_latency * 1.10
        material_queue_observed = queue_case is True
        effects = (deadline_deterioration, latency_deterioration, material_queue_observed)
        h1 = (
            "supported by observed results"
            if all(effects)
            else ("partially supported" if any(effects) else "not supported")
        )
    lines.extend(
        (
            f"## H1 — {h1}",
            "",
            f"Fastest Device deadline satisfaction was {_format_percent(fd_normal_deadline)} "
            f"under normal load and {_format_percent(fd_burst_deadline)} under bursty load; mean "
            "latency was "
            f"{_format_number(fd_normal_latency)} ms and "
            f"{_format_number(fd_burst_latency)} ms.",
            "Edge-server compute utilization was "
            f"{_format_percent(fd_normal_edge_util)} under normal load and "
            f"{_format_percent(fd_burst_edge_util)} under bursty load. A representative "
            f"burst queue case was {queue_case_text}.",
            "",
        )
    )

    h2_deltas: list[tuple[str, float]] = []
    for scenario in CORE_SCENARIOS:
        mct = _metric_mean(
            summaries, scenario, "min_completion", lambda item: item.deadline_satisfaction_rate
        )
        rr = _metric_mean(
            summaries, scenario, "round_robin", lambda item: item.deadline_satisfaction_rate
        )
        if mct is not None and rr is not None:
            h2_deltas.append((scenario, mct - rr))
    positive = sum(delta > 1e-12 for _, delta in h2_deltas)
    if not h2_deltas:
        h2 = "inconclusive"
    elif positive == len(h2_deltas):
        h2 = "supported by observed results"
    elif positive:
        h2 = "partially supported"
    else:
        h2 = "not supported"
    delta_text = (
        ", ".join(
            f"{_SCENARIO_LABELS.get(scenario, scenario)} {delta * 100.0:+.2f} percentage points"
            for scenario, delta in h2_deltas
        )
        or "comparisons not available"
    )
    lines.extend((f"## H2 — {h2}", "", f"MCT minus Round Robin: {delta_text}.", ""))

    full_burst = _metric_mean(
        summaries, "bursty", "edgeweaver", lambda item: item.deadline_satisfaction_rate
    )
    no_switch = _metric_mean(
        summaries,
        "bursty",
        "edgeweaver_no_model_switching",
        lambda item: item.deadline_satisfaction_rate,
    )
    full_burst_goodput = _metric_mean(
        summaries,
        "bursty",
        "edgeweaver",
        lambda item: item.useful_goodput_requests_per_second,
    )
    no_switch_goodput = _metric_mean(
        summaries,
        "bursty",
        "edgeweaver_no_model_switching",
        lambda item: item.useful_goodput_requests_per_second,
    )
    full_burst_distribution = _model_selection_means(summaries, "bursty", "edgeweaver")
    no_switch_distribution = _model_selection_means(
        summaries, "bursty", "edgeweaver_no_model_switching"
    )
    model_choice_changed = full_burst_distribution != no_switch_distribution
    if full_burst is None or no_switch is None:
        h3 = "inconclusive"
    elif not model_choice_changed:
        h3 = "inconclusive (ablation did not change realized model choices)"
    elif full_burst > no_switch + 1e-12:
        h3 = "supported by observed results"
    elif abs(full_burst - no_switch) <= 1e-12:
        h3 = "not supported"
    else:
        h3 = "not supported"
    lines.extend(
        (
            f"## H3 — {h3}",
            "",
            f"Bursty deadline satisfaction: full EdgeWeaver {_format_percent(full_burst)}; "
            f"no-model-switching {_format_percent(no_switch)}. Useful goodput was "
            f"{full_burst_goodput if full_burst_goodput is not None else 'not available'} and "
            f"{no_switch_goodput if no_switch_goodput is not None else 'not available'} "
            "requests/s, respectively.",
            "",
        )
    )

    full_energy = _metric_mean(
        summaries,
        "bursty",
        "edgeweaver",
        lambda item: item.estimated_energy_per_completed_request_units,
    )
    accuracy_first_energy = _metric_mean(
        summaries,
        "bursty",
        "edgeweaver_no_model_switching",
        lambda item: item.estimated_energy_per_completed_request_units,
    )
    full_accuracy = _metric_mean(
        summaries, "bursty", "edgeweaver", lambda item: item.actual_prediction_accuracy
    )
    accuracy_first_accuracy = _metric_mean(
        summaries,
        "bursty",
        "edgeweaver_no_model_switching",
        lambda item: item.actual_prediction_accuracy,
    )
    if None in (full_energy, accuracy_first_energy, full_accuracy, accuracy_first_accuracy):
        h4 = "inconclusive"
    elif not model_choice_changed:
        h4 = "inconclusive (accuracy-first ablation selected the same models)"
    elif (
        full_energy is not None
        and accuracy_first_energy is not None
        and full_energy < accuracy_first_energy
    ):
        h4 = "partially supported"
    else:
        h4 = "not supported"
    lines.extend(
        (
            f"## H4 — {h4}",
            "",
            "Bursty estimated energy (normalized units/request): full EdgeWeaver "
            f"{full_energy if full_energy is not None else 'not available'}; accuracy-first "
            "ablation "
            f"{accuracy_first_energy if accuracy_first_energy is not None else 'not available'}. "
            f"Actual accuracy was {_format_percent(full_accuracy)} and "
            f"{_format_percent(accuracy_first_accuracy)}. No separate aggregate acceptability "
            "threshold or significance test was imposed for H4.",
            "",
        )
    )

    full_slowdown = _metric_mean(
        summaries, "device_slowdown", "edgeweaver", lambda item: item.deadline_satisfaction_rate
    )
    no_update = _metric_mean(
        summaries,
        "device_slowdown",
        "edgeweaver_no_online_update",
        lambda item: item.deadline_satisfaction_rate,
    )
    full_slowdown_latency = _metric_mean(
        summaries,
        "device_slowdown",
        "edgeweaver",
        lambda item: item.mean_end_to_end_latency_ms,
    )
    no_update_latency = _metric_mean(
        summaries,
        "device_slowdown",
        "edgeweaver_no_online_update",
        lambda item: item.mean_end_to_end_latency_ms,
    )
    full_slowdown_goodput = _metric_mean(
        summaries,
        "device_slowdown",
        "edgeweaver",
        lambda item: item.useful_goodput_requests_per_second,
    )
    no_update_goodput = _metric_mean(
        summaries,
        "device_slowdown",
        "edgeweaver_no_online_update",
        lambda item: item.useful_goodput_requests_per_second,
    )
    h5_values = (
        full_slowdown,
        no_update,
        full_slowdown_latency,
        no_update_latency,
        full_slowdown_goodput,
        no_update_goodput,
    )
    full_edge_server_utilization = _device_utilization_mean(
        summaries, "device_slowdown", "edgeweaver", "edge-server"
    )
    if any(value is None for value in h5_values):
        h5 = "inconclusive"
    elif full_edge_server_utilization is not None and full_edge_server_utilization <= 1e-12:
        h5 = "inconclusive (EdgeWeaver did not use the slowed device)"
    else:
        assert full_slowdown is not None
        assert no_update is not None
        assert full_slowdown_latency is not None
        assert no_update_latency is not None
        assert full_slowdown_goodput is not None
        assert no_update_goodput is not None
        deadline_better = full_slowdown > no_update + 1e-12
        latency_better = full_slowdown_latency < no_update_latency - 1e-12
        goodput_better = full_slowdown_goodput > no_update_goodput + 1e-12
        improvements = (deadline_better, latency_better, goodput_better)
        h5 = (
            "supported by observed results"
            if all(improvements)
            else ("partially supported" if any(improvements) else "not supported")
        )
    lines.extend(
        (
            f"## H5 — {h5}",
            "",
            f"Device-slowdown deadline satisfaction: full EdgeWeaver "
            f"{_format_percent(full_slowdown)}; no-online-update {_format_percent(no_update)}. "
            f"Mean latency was {_format_number(full_slowdown_latency)} ms versus "
            f"{_format_number(no_update_latency)} ms; useful goodput was "
            f"{_format_number(full_slowdown_goodput)} versus "
            f"{_format_number(no_update_goodput)} requests/s.",
            "",
        )
    )

    if failure_cases is not None:
        observed = sum(item.observed for item in failure_cases.cases)
        lines.extend(
            (
                "## Representative cases",
                "",
                f"{observed}/{len(failure_cases.cases)} requested case types were observed in the "
                "saved structured outputs. Unobserved types remain explicitly marked in "
                "`failure_cases.json`.",
                "",
            )
        )
    return "\n".join(lines)


def save_findings_summary(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
