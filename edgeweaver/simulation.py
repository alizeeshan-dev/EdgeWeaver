"""Minimal deterministic SimPy vertical slice for the Phase 1 fixture."""

from collections.abc import Generator
from typing import Any

import simpy

from edgeweaver.config import LoadedConfiguration
from edgeweaver.domain import (
    AssignmentDecision,
    Device,
    EventType,
    FixtureSimulationResult,
    InferenceRequest,
    JsonScalar,
    ModelProfile,
    RequestMetrics,
    SimulationSummary,
    StructuredEvent,
)


def meets_deadline(request: InferenceRequest, completion_time_ms: float) -> bool:
    """Return whether an absolute completion time meets a relative deadline."""

    return completion_time_ms <= request.absolute_deadline_ms


def simulated_inference_time_ms(profile: ModelProfile, device: Device) -> float:
    """Scale the synthetic/base latency by the configured device speed multiplier."""

    return profile.local_latency_ms_mean * device.speed_multiplier


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[StructuredEvent] = []

    def record(
        self,
        *,
        timestamp_ms: float,
        event_type: EventType,
        request_id: str,
        device_id: str,
        model_id: str,
        details: dict[str, JsonScalar] | None = None,
    ) -> None:
        self.events.append(
            StructuredEvent(
                sequence=len(self.events),
                timestamp_ms=timestamp_ms,
                event_type=event_type,
                request_id=request_id,
                device_id=device_id,
                model_id=model_id,
                details=details or {},
            )
        )


def _fixed_assignment(
    request: InferenceRequest,
    device: Device,
    profile: ModelProfile,
    service_time_ms: float,
) -> AssignmentDecision:
    predicted_completion_ms = request.arrival_time_ms + service_time_ms
    predicted_energy_units = device.active_power_units * service_time_ms
    return AssignmentDecision(
        request_id=request.request_id,
        device_id=device.id,
        model_id=profile.model_id,
        predicted_completion_ms=predicted_completion_ms,
        predicted_energy_units=predicted_energy_units,
        predicted_model_accuracy=profile.accuracy,
        expected_to_meet_deadline=predicted_completion_ms <= request.absolute_deadline_ms,
        decision_reason=(
            "Phase 1 fixture uses one fixed device and model; no scheduler policy is active."
        ),
    )


def run_fixture_simulation(configuration: LoadedConfiguration) -> FixtureSimulationResult:
    """Run the one-resource, fixed-input Phase 1 simulation to completion."""

    environment = simpy.Environment()
    device = configuration.selected_device()
    profile = configuration.selected_model_profile()
    resource = simpy.Resource(environment, capacity=1)
    recorder = _EventRecorder()
    service_time_ms = simulated_inference_time_ms(profile, device)
    metrics_by_request: dict[str, RequestMetrics] = {}

    ordered_requests = sorted(
        configuration.scenario.requests,
        key=lambda request: (request.arrival_time_ms, request.request_id),
    )

    def process_request(
        request: InferenceRequest,
    ) -> Generator[simpy.events.Event, Any, None]:
        yield environment.timeout(request.arrival_time_ms)
        recorder.record(
            timestamp_ms=float(environment.now),
            event_type=EventType.REQUEST_ARRIVED,
            request_id=request.request_id,
            device_id=device.id,
            model_id=profile.model_id,
        )
        with resource.request() as resource_request:
            recorder.record(
                timestamp_ms=float(environment.now),
                event_type=EventType.REQUEST_QUEUED,
                request_id=request.request_id,
                device_id=device.id,
                model_id=profile.model_id,
                details={"queue_length": len(resource.queue)},
            )
            yield resource_request
            started_at_ms = float(environment.now)
            queue_wait_ms = started_at_ms - request.arrival_time_ms
            recorder.record(
                timestamp_ms=started_at_ms,
                event_type=EventType.INFERENCE_STARTED,
                request_id=request.request_id,
                device_id=device.id,
                model_id=profile.model_id,
                details={"queue_wait_ms": queue_wait_ms},
            )
            yield environment.timeout(service_time_ms)

            completion_time_ms = float(environment.now)
            end_to_end_latency_ms = completion_time_ms - request.arrival_time_ms
            deadline_met = meets_deadline(request, completion_time_ms)
            recorder.record(
                timestamp_ms=completion_time_ms,
                event_type=EventType.INFERENCE_COMPLETED,
                request_id=request.request_id,
                device_id=device.id,
                model_id=profile.model_id,
                details={
                    "service_time_ms": service_time_ms,
                    "end_to_end_latency_ms": end_to_end_latency_ms,
                    "deadline_met": deadline_met,
                },
            )
            metrics_by_request[request.request_id] = RequestMetrics(
                request_id=request.request_id,
                queue_wait_ms=queue_wait_ms,
                completion_time_ms=completion_time_ms,
                end_to_end_latency_ms=end_to_end_latency_ms,
                absolute_deadline_ms=request.absolute_deadline_ms,
                deadline_met=deadline_met,
            )

    for request in ordered_requests:
        environment.process(process_request(request))
    environment.run()

    request_metrics = [metrics_by_request[request.request_id] for request in ordered_requests]
    assignments = [
        _fixed_assignment(request, device, profile, service_time_ms) for request in ordered_requests
    ]
    deadlines_met = sum(metric.deadline_met for metric in request_metrics)
    summary = SimulationSummary(
        scenario_id=configuration.scenario.scenario_id,
        total_requests=len(request_metrics),
        deadlines_met=deadlines_met,
        deadline_satisfaction_rate=deadlines_met / len(request_metrics),
        requests=request_metrics,
    )
    return FixtureSimulationResult(
        events=recorder.events,
        assignments=assignments,
        summary=summary,
    )
