"""Deterministic per-run research metrics derived from structured simulation results."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from edgeweaver.domain import (
    Device,
    DomainModel,
    ModelProfile,
    ModelRole,
    RequestExecutionResult,
    RequestStatus,
    SimulationRunResult,
)
from edgeweaver.workloads import WorkloadTrace

METRICS_FORMAT_VERSION: Literal["edgeweaver-run-metrics-v1"] = "edgeweaver-run-metrics-v1"
_FLOAT_TOLERANCE = 1e-9


class DeadlineMissCause(StrEnum):
    """The four primary deadline-miss causes required by the project plan."""

    NETWORK_DELAY = "network_delay"
    QUEUE_DELAY = "queue_delay"
    INFERENCE_TIME = "inference_time"
    INCORRECT_STATIC_ESTIMATE = "incorrect_static_estimate"


class DeviceUtilizationMetric(DomainModel):
    device_id: str = Field(min_length=1)
    busy_time_ms: float = Field(ge=0.0)
    utilization: float = Field(ge=0.0, le=1.0)


class ModelSelectionMetric(DomainModel):
    model_id: str = Field(min_length=1)
    model_role: ModelRole | None = None
    assigned_requests: int = Field(ge=0)
    percentage: float = Field(ge=0.0, le=100.0)


class DeadlineMissCauseMetric(DomainModel):
    cause: DeadlineMissCause
    count: int = Field(ge=0)
    rate: float = Field(ge=0.0, le=1.0)


class PerRunMetrics(DomainModel):
    """Stable summary for exactly one scheduler/scenario/seed run.

    Rates whose denominator is zero are represented as ``0.0``. Latency,
    accuracy, and energy averages for a zero-completion run are also ``0.0``;
    the accompanying counts make that state explicit.
    """

    format_version: Literal["edgeweaver-run-metrics-v1"] = METRICS_FORMAT_VERSION
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scheduler_name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    simulation_duration_ms: float = Field(gt=0.0)
    total_requests: int = Field(ge=0)
    assigned_requests: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    rejected_requests: int = Field(ge=0)
    deadlines_met: int = Field(ge=0)
    deadline_misses: int = Field(ge=0)
    useful_valid_predictions: int = Field(ge=0)
    correct_predictions: int = Field(ge=0)
    deadline_satisfaction_rate: float = Field(ge=0.0, le=1.0)
    useful_goodput_requests_per_second: float = Field(ge=0.0)
    mean_end_to_end_latency_ms: float = Field(ge=0.0)
    p95_end_to_end_latency_ms: float = Field(ge=0.0)
    actual_prediction_accuracy: float = Field(ge=0.0, le=1.0)
    total_estimated_energy_units: float = Field(ge=0.0)
    estimated_energy_per_completed_request_units: float = Field(ge=0.0)
    device_utilization: list[DeviceUtilizationMetric]
    model_selection_distribution: list[ModelSelectionMetric]
    deadline_miss_causes: list[DeadlineMissCauseMetric]

    @model_validator(mode="after")
    def counts_are_consistent(self) -> PerRunMetrics:
        if self.completed_requests + self.rejected_requests != self.total_requests:
            raise ValueError("completed plus rejected requests must equal total requests")
        if self.deadlines_met + self.deadline_misses != self.completed_requests:
            raise ValueError("deadline outcomes must equal completed requests")
        if self.correct_predictions > self.completed_requests:
            raise ValueError("correct predictions cannot exceed completed requests")
        if self.useful_valid_predictions > self.completed_requests:
            raise ValueError("useful valid predictions cannot exceed completed requests")
        return self


class TidyMetricRow(DomainModel):
    """One long-form CSV row ready for Phase 7 concatenation."""

    run_id: str
    scenario_id: str
    scheduler_name: str
    seed: int
    metric: str
    dimension: str
    value: float
    unit: str


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def _linear_percentile(values: Sequence[float], percentile: float) -> float:
    """Match NumPy's documented linear percentile without requiring array conversion."""

    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def _selected_predicted_inference_ms(request: RequestExecutionResult) -> float | None:
    selected = (
        candidate
        for candidate in request.assignment.candidates
        if candidate.device_id == request.assignment.device_id
        and candidate.model_id == request.assignment.model_id
    )
    candidate = next(selected, None)
    return None if candidate is None else candidate.predicted_inference_time_ms


def classify_deadline_miss(request: RequestExecutionResult) -> DeadlineMissCause:
    """Classify one completed deadline miss using a conservative deterministic rule.

    ``incorrect_static_estimate`` is used only when the scheduler predicted success,
    actual inference exceeded the selected candidate's estimate, and removing that
    inference underestimation from actual completion would have met the deadline.
    Otherwise the largest actual component wins among aggregate network time,
    queue wait, and inference time. Exact component ties use that listed order.
    """

    if request.status is not RequestStatus.COMPLETED or request.deadline_met:
        raise ValueError("deadline-miss classification requires a completed missed request")
    assert request.completion_time_ms is not None

    predicted_inference_ms = _selected_predicted_inference_ms(request)
    predicted_success = (
        request.assignment.expected_to_meet_deadline
        and request.assignment.predicted_completion_ms
        <= request.absolute_deadline_ms + _FLOAT_TOLERANCE
    )
    if predicted_success and predicted_inference_ms is not None:
        inference_underestimate_ms = request.inference_time_ms - predicted_inference_ms
        corrected_completion_ms = request.completion_time_ms - max(inference_underestimate_ms, 0.0)
        if (
            inference_underestimate_ms > _FLOAT_TOLERANCE
            and corrected_completion_ms <= request.absolute_deadline_ms + _FLOAT_TOLERANCE
        ):
            return DeadlineMissCause.INCORRECT_STATIC_ESTIMATE

    components = (
        (DeadlineMissCause.NETWORK_DELAY, request.upload_time_ms + request.return_time_ms),
        (DeadlineMissCause.QUEUE_DELAY, request.queue_wait_ms),
        (DeadlineMissCause.INFERENCE_TIME, request.inference_time_ms),
    )
    return max(components, key=lambda item: item[1])[0]


def _merged_busy_time_ms(
    intervals: Iterable[tuple[float, float]], simulation_duration_ms: float
) -> float:
    clipped = sorted(
        (max(0.0, start), min(simulation_duration_ms, end))
        for start, end in intervals
        if end > 0.0 and start < simulation_duration_ms
    )
    clipped = [(start, end) for start, end in clipped if end > start]
    if not clipped:
        return 0.0
    busy_time_ms = 0.0
    active_start, active_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
        else:
            busy_time_ms += active_end - active_start
            active_start, active_end = start, end
    return busy_time_ms + active_end - active_start


def calculate_run_metrics(
    run: SimulationRunResult,
    trace: WorkloadTrace,
    *,
    scenario_id: str,
    simulation_duration_ms: float,
    devices: Sequence[Device],
    model_profiles: Sequence[ModelProfile],
) -> PerRunMetrics:
    """Calculate all Phase 6 metrics from one immutable run and its saved trace."""

    if simulation_duration_ms <= 0.0:
        raise ValueError("simulation duration must be positive")
    trace_requests = {request.request_id: request for request in trace.requests}
    result_ids = [request.request_id for request in run.requests]
    if len(trace_requests) != len(trace.requests) or set(result_ids) != set(trace_requests):
        raise ValueError("run results and workload trace must contain identical request IDs")

    completed = [request for request in run.requests if request.status is RequestStatus.COMPLETED]
    deadlines_met = sum(request.deadline_met for request in completed)
    deadline_misses = len(completed) - deadlines_met
    correct_predictions = sum(request.prediction_correct is True for request in completed)
    useful_valid_predictions = sum(
        request.deadline_met
        and request.prediction_correct is True
        and request.assignment.predicted_model_accuracy
        >= trace_requests[request.request_id].minimum_accuracy
        for request in completed
    )
    latencies = [
        request.end_to_end_latency_ms
        for request in completed
        if request.end_to_end_latency_ms is not None
    ]
    total_energy = sum(request.energy.estimated_total_energy_units for request in completed)

    intervals_by_device: dict[str, list[tuple[float, float]]] = {
        device.id: [] for device in devices
    }
    for request in completed:
        start_ms = (
            request.arrival_time_ms
            + request.scheduler_overhead_ms
            + request.upload_time_ms
            + request.queue_wait_ms
        )
        intervals_by_device.setdefault(request.assignment.device_id, []).append(
            (start_ms, start_ms + request.inference_time_ms)
        )
    device_utilization = []
    for device_id in sorted(intervals_by_device):
        busy_time_ms = _merged_busy_time_ms(intervals_by_device[device_id], simulation_duration_ms)
        device_utilization.append(
            DeviceUtilizationMetric(
                device_id=device_id,
                busy_time_ms=busy_time_ms,
                utilization=busy_time_ms / simulation_duration_ms,
            )
        )

    assigned_requests = len(run.requests)
    model_counts: dict[str, int] = {profile.model_id: 0 for profile in model_profiles}
    for request in run.requests:
        model_counts[request.assignment.model_id] = (
            model_counts.get(request.assignment.model_id, 0) + 1
        )
    profile_roles = {profile.model_id: profile.computational_role for profile in model_profiles}
    model_selection = [
        ModelSelectionMetric(
            model_id=model_id,
            model_role=profile_roles.get(model_id),
            assigned_requests=count,
            percentage=100.0 * _safe_ratio(count, assigned_requests),
        )
        for model_id, count in sorted(model_counts.items())
    ]

    cause_counts = dict.fromkeys(DeadlineMissCause, 0)
    for request in completed:
        if not request.deadline_met:
            cause = classify_deadline_miss(request)
            cause_counts[cause] += 1
    miss_causes = [
        DeadlineMissCauseMetric(
            cause=cause,
            count=cause_counts[cause],
            rate=_safe_ratio(cause_counts[cause], deadline_misses),
        )
        for cause in DeadlineMissCause
    ]

    return PerRunMetrics(
        run_id=run.run_id,
        scenario_id=scenario_id,
        scheduler_name=run.scheduler_name,
        seed=run.random_seed,
        simulation_duration_ms=simulation_duration_ms,
        total_requests=run.request_count,
        assigned_requests=assigned_requests,
        completed_requests=run.completed_requests,
        rejected_requests=run.rejected_requests,
        deadlines_met=deadlines_met,
        deadline_misses=deadline_misses,
        useful_valid_predictions=useful_valid_predictions,
        correct_predictions=correct_predictions,
        deadline_satisfaction_rate=_safe_ratio(deadlines_met, len(completed)),
        useful_goodput_requests_per_second=(
            useful_valid_predictions / (simulation_duration_ms / 1000.0)
        ),
        mean_end_to_end_latency_ms=_safe_ratio(sum(latencies), len(latencies)),
        p95_end_to_end_latency_ms=_linear_percentile(latencies, 95.0),
        actual_prediction_accuracy=_safe_ratio(correct_predictions, len(completed)),
        total_estimated_energy_units=total_energy,
        estimated_energy_per_completed_request_units=_safe_ratio(total_energy, len(completed)),
        device_utilization=device_utilization,
        model_selection_distribution=model_selection,
        deadline_miss_causes=miss_causes,
    )


def summary_to_tidy_rows(summary: PerRunMetrics) -> list[TidyMetricRow]:
    """Flatten one complete summary into stable long-form metric rows."""

    def row(metric: str, dimension: str, value: float, unit: str) -> TidyMetricRow:
        return TidyMetricRow(
            run_id=summary.run_id,
            scenario_id=summary.scenario_id,
            scheduler_name=summary.scheduler_name,
            seed=summary.seed,
            metric=metric,
            dimension=dimension,
            value=value,
            unit=unit,
        )

    scalar_metrics = (
        ("simulation_duration", summary.simulation_duration_ms, "ms"),
        ("total_requests", summary.total_requests, "requests"),
        ("assigned_requests", summary.assigned_requests, "requests"),
        ("completed_requests", summary.completed_requests, "requests"),
        ("rejected_requests", summary.rejected_requests, "requests"),
        ("deadlines_met", summary.deadlines_met, "requests"),
        ("deadline_misses", summary.deadline_misses, "requests"),
        ("useful_valid_predictions", summary.useful_valid_predictions, "requests"),
        ("correct_predictions", summary.correct_predictions, "requests"),
        ("deadline_satisfaction_rate", summary.deadline_satisfaction_rate, "ratio"),
        (
            "useful_goodput",
            summary.useful_goodput_requests_per_second,
            "requests_per_second",
        ),
        ("mean_end_to_end_latency", summary.mean_end_to_end_latency_ms, "ms"),
        ("p95_end_to_end_latency", summary.p95_end_to_end_latency_ms, "ms"),
        ("actual_prediction_accuracy", summary.actual_prediction_accuracy, "ratio"),
        ("total_estimated_energy", summary.total_estimated_energy_units, "normalized_units"),
        (
            "estimated_energy_per_completed_request",
            summary.estimated_energy_per_completed_request_units,
            "normalized_units_per_request",
        ),
    )
    rows = [row(metric, "", value, unit) for metric, value, unit in scalar_metrics]
    for utilization_metric in summary.device_utilization:
        rows.extend(
            (
                row(
                    "device_busy_time",
                    utilization_metric.device_id,
                    utilization_metric.busy_time_ms,
                    "ms",
                ),
                row(
                    "device_utilization",
                    utilization_metric.device_id,
                    utilization_metric.utilization,
                    "ratio",
                ),
            )
        )
    for selection_metric in summary.model_selection_distribution:
        rows.extend(
            (
                row(
                    "model_selection_count",
                    selection_metric.model_id,
                    selection_metric.assigned_requests,
                    "requests",
                ),
                row(
                    "model_selection_percentage",
                    selection_metric.model_id,
                    selection_metric.percentage,
                    "percent",
                ),
            )
        )
    for cause_metric in summary.deadline_miss_causes:
        rows.extend(
            (
                row(
                    "deadline_miss_cause_count",
                    cause_metric.cause.value,
                    cause_metric.count,
                    "requests",
                ),
                row(
                    "deadline_miss_cause_rate",
                    cause_metric.cause.value,
                    cause_metric.rate,
                    "ratio",
                ),
            )
        )
    return rows


def save_run_metrics_json(path: Path, summary: PerRunMetrics) -> None:
    """Save one complete summary as strict, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_run_metrics_csv(path: Path, summary: PerRunMetrics) -> None:
    """Save one complete summary as stable long-form CSV."""

    rows = summary_to_tidy_rows(summary)
    fieldnames = list(TidyMetricRow.model_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row.model_dump(mode="json") for row in rows)
