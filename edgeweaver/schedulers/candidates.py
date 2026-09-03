"""Shared deterministic candidate estimates used by baseline schedulers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from edgeweaver.compatibility import is_device_model_compatible
from edgeweaver.domain import (
    AssignmentDecision,
    CandidateEstimate,
    DeviceState,
    InferenceRequest,
    SimulationState,
)
from edgeweaver.energy import estimate_energy
from edgeweaver.network import network_timing
from edgeweaver.simulation import simulated_inference_time_ms


class NoValidAssignmentError(ValueError):
    """Raised when no configured device/model pair can serve a request."""


def estimate_queue_wait_ms(
    device_state: DeviceState,
    *,
    arrival_lead_time_ms: float = 0.0,
) -> float:
    """Estimate FIFO wait from the actual service work visible at a device.

    Active values are remaining durations. Queued values are each request's own
    assigned service duration, in admission order. The lane calculation also
    works for configured processing capacities greater than one.
    """

    if arrival_lead_time_ms < 0.0:
        raise ValueError("arrival lead time must be non-negative")
    wait_ms, _ = estimate_queue_arrival(
        device_state,
        current_time_ms=0.0,
        arrival_lead_time_ms=arrival_lead_time_ms,
    )
    return wait_ms


def estimate_queue_arrival(
    device_state: DeviceState,
    *,
    current_time_ms: float,
    arrival_lead_time_ms: float,
) -> tuple[float, bool]:
    """Return predicted wait and whether the bounded queue can admit the request."""

    if current_time_ms < 0.0 or arrival_lead_time_ms < 0.0:
        raise ValueError("queue timing inputs must be non-negative")
    lane_available_ms = [
        current_time_ms + remaining_ms for remaining_ms in device_state.active_remaining_ms
    ]
    lane_available_ms.extend(
        current_time_ms for _ in range(device_state.processing_capacity - len(lane_available_ms))
    )
    if device_state.pending_work:
        pending = sorted(
            (
                work.ready_time_ms,
                work.sequence,
                work.request_id,
                work.inference_time_ms,
            )
            for work in device_state.pending_work
        )
    else:
        pending = [
            (current_time_ms, index, "", service_time_ms)
            for index, service_time_ms in enumerate(device_state.queued_inference_times_ms)
        ]

    target_ready_ms = current_time_ms + arrival_lead_time_ms
    scheduled: list[tuple[float, float, float]] = []
    for ready_time_ms, sequence, request_id, service_time_ms in pending:
        if (ready_time_ms, sequence, request_id) > (
            target_ready_ms,
            2**63 - 1,
            "\uffff",
        ):
            break
        lane_index = min(
            range(device_state.processing_capacity),
            key=lambda index: (lane_available_ms[index], index),
        )
        started_ms = max(ready_time_ms, lane_available_ms[lane_index])
        completed_ms = started_ms + service_time_ms
        lane_available_ms[lane_index] = completed_ms
        scheduled.append((ready_time_ms, started_ms, completed_ms))

    active_at_arrival = sum(
        completion_ms > target_ready_ms
        for completion_ms in (
            current_time_ms + remaining_ms for remaining_ms in device_state.active_remaining_ms
        )
    ) + sum(
        started_ms <= target_ready_ms < completed_ms for _, started_ms, completed_ms in scheduled
    )
    waiting_at_arrival = sum(
        ready_ms <= target_ready_ms < started_ms for ready_ms, started_ms, _ in scheduled
    )
    queue_admissible = (
        active_at_arrival < device_state.processing_capacity
        or waiting_at_arrival < device_state.queue_capacity
    )
    target_started_ms = max(target_ready_ms, min(lane_available_ms))
    return target_started_ms - target_ready_ms, queue_admissible


def estimate_candidates(
    request: InferenceRequest,
    state: SimulationState,
    *,
    inference_estimates_ms: Mapping[tuple[str, str], float] | None = None,
) -> tuple[CandidateEstimate, ...]:
    """Estimate every configured pair, retaining deterministic rejection details."""

    config = state.simulation_config
    if config is None or not state.device_profiles:
        raise ValueError("simulation state lacks scheduler configuration or device profiles")
    states_by_id = {device.device_id: device for device in state.devices}
    estimates: list[CandidateEstimate] = []

    for device in state.device_profiles:
        try:
            device_state = states_by_id[device.id]
        except KeyError as error:
            raise ValueError(f"simulation state lacks device state for {device.id}") from error
        for profile in state.model_profiles:
            compatible = is_device_model_compatible(device, profile)
            accuracy_eligible = profile.accuracy >= request.minimum_accuracy
            network_available = True
            rejection_reason: str | None = None
            if not compatible:
                rejection_reason = "incompatible_device_model"
            elif not accuracy_eligible:
                rejection_reason = "model_accuracy_below_request_minimum"

            try:
                timing = network_timing(
                    source_device_id=request.source_device_id,
                    destination_device_id=device.id,
                    input_size_bytes=request.input_size_bytes,
                    return_size_bytes=config.return_payload_size_bytes,
                    links=state.network_links,
                    random_seed=config.random_seed,
                    request_id=request.request_id,
                )
            except ValueError:
                network_available = False
                if rejection_reason is None:
                    rejection_reason = "missing_network_link"
                upload_time_ms = 0.0
                return_time_ms = 0.0
            else:
                upload_time_ms = timing.upload_time_ms
                return_time_ms = timing.return_time_ms

            queue_wait_ms, queue_admissible = estimate_queue_arrival(
                device_state,
                current_time_ms=state.current_time_ms,
                arrival_lead_time_ms=config.scheduler_overhead_ms + upload_time_ms,
            )
            inference_time_ms = (
                simulated_inference_time_ms(profile, device)
                if inference_estimates_ms is None
                else inference_estimates_ms.get(
                    (device.id, profile.model_id),
                    simulated_inference_time_ms(profile, device),
                )
            )
            if inference_time_ms <= 0.0:
                raise ValueError("inference estimates must be positive")
            transferred_bytes = (
                0
                if request.source_device_id == device.id
                else request.input_size_bytes + config.return_payload_size_bytes
            )
            energy = estimate_energy(
                device=device,
                inference_time_ms=inference_time_ms,
                transferred_bytes=transferred_bytes,
                network_energy_per_kilobyte=config.network_energy_per_kilobyte,
            )
            completion_ms = (
                state.current_time_ms
                + config.scheduler_overhead_ms
                + upload_time_ms
                + queue_wait_ms
                + inference_time_ms
                + return_time_ms
            )
            estimates.append(
                CandidateEstimate(
                    device_id=device.id,
                    model_id=profile.model_id,
                    model_accuracy=profile.accuracy,
                    compatible=compatible,
                    accuracy_eligible=accuracy_eligible,
                    network_available=network_available,
                    eligible=compatible and accuracy_eligible and network_available,
                    rejection_reason=rejection_reason,
                    queue_admissible=queue_admissible,
                    predicted_upload_time_ms=upload_time_ms,
                    predicted_queue_wait_ms=queue_wait_ms,
                    predicted_inference_time_ms=inference_time_ms,
                    predicted_return_time_ms=return_time_ms,
                    scheduler_overhead_ms=config.scheduler_overhead_ms,
                    predicted_completion_ms=completion_ms,
                    predicted_energy_units=energy.estimated_total_energy_units,
                    expected_to_meet_deadline=(
                        rejection_reason is None
                        and queue_admissible
                        and completion_ms <= request.absolute_deadline_ms
                    ),
                )
            )
    return tuple(estimates)


def eligible_candidates(
    candidates: Sequence[CandidateEstimate],
) -> tuple[CandidateEstimate, ...]:
    return tuple(candidate for candidate in candidates if candidate.eligible)


def decision_from_candidate(
    request: InferenceRequest,
    candidate: CandidateEstimate,
    candidates: Sequence[CandidateEstimate],
    reason: str,
) -> AssignmentDecision:
    return AssignmentDecision(
        request_id=request.request_id,
        device_id=candidate.device_id,
        model_id=candidate.model_id,
        predicted_completion_ms=candidate.predicted_completion_ms,
        predicted_energy_units=candidate.predicted_energy_units,
        predicted_model_accuracy=candidate.model_accuracy,
        expected_to_meet_deadline=candidate.expected_to_meet_deadline,
        decision_reason=reason,
        candidates=tuple(candidates),
    )
