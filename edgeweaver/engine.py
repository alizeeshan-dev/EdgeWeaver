"""Scheduler-agnostic three-device SimPy simulation engine."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Any, Protocol

import simpy

from edgeweaver.compatibility import is_device_model_compatible
from edgeweaver.config import DeviceCatalog, NetworkCatalog
from edgeweaver.domain import (
    AssignmentDecision,
    Device,
    DeviceState,
    EnergyBreakdown,
    EventType,
    ExecutionObservation,
    InferenceRequest,
    JsonScalar,
    ModelProfile,
    PendingServiceWork,
    RequestExecutionResult,
    RequestStatus,
    SimulationConfig,
    SimulationRunResult,
    SimulationState,
    StructuredEvent,
)
from edgeweaver.energy import estimate_energy
from edgeweaver.network import NetworkTiming, find_network_link, network_timing
from edgeweaver.runtime_conditions import RuntimeConditionEvent, RuntimeConditions
from edgeweaver.scheduler import AssignmentProvider
from edgeweaver.simulation import meets_deadline, simulated_inference_time_ms
from edgeweaver.workloads import WorkloadTrace

DEFAULT_DEVICE_IDS = {"mobile", "gateway", "edge-server"}


class PredictionProvider(Protocol):
    def predict(self, model_id: str, feature_vector_id: int) -> int: ...


class InferenceDurationProvider(Protocol):
    def inference_time_ms(
        self,
        request: InferenceRequest,
        device: Device,
        profile: ModelProfile,
        static_inference_time_ms: float,
    ) -> float: ...


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[StructuredEvent] = []

    def record(
        self,
        *,
        timestamp_ms: float,
        event_type: EventType,
        request_id: str | None = None,
        device_id: str | None = None,
        model_id: str | None = None,
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


class _DeviceRuntime:
    def __init__(self, environment: simpy.Environment, device: Device) -> None:
        self.environment = environment
        self.device = device
        self.resource = simpy.Resource(environment, capacity=device.processing_capacity)
        self._pending_work: dict[str, PendingServiceWork] = {}
        self._active_completion_ms: dict[str, float] = {}
        self._reservation_sequence = 0

    def has_queue_space(self) -> bool:
        return (
            self.resource.count < self.device.processing_capacity
            or len(self.resource.queue) < self.device.queue_capacity
        )

    def reserve(self, request_id: str, ready_time_ms: float, service_time_ms: float) -> None:
        if request_id in self._pending_work or request_id in self._active_completion_ms:
            raise ValueError(f"request {request_id} already reserved on {self.device.id}")
        self._pending_work[request_id] = PendingServiceWork(
            request_id=request_id,
            ready_time_ms=ready_time_ms,
            inference_time_ms=service_time_ms,
            sequence=self._reservation_sequence,
        )
        self._reservation_sequence += 1

    def cancel(self, request_id: str) -> None:
        del self._pending_work[request_id]

    def update_ready_time(self, request_id: str, ready_time_ms: float) -> None:
        work = self._pending_work[request_id]
        self._pending_work[request_id] = work.model_copy(update={"ready_time_ms": ready_time_ms})

    def start(self, request_id: str) -> None:
        work = self._pending_work.pop(request_id)
        self._active_completion_ms[request_id] = (
            float(self.environment.now) + work.inference_time_ms
        )

    def complete(self, request_id: str) -> None:
        del self._active_completion_ms[request_id]

    def snapshot(self) -> DeviceState:
        now_ms = float(self.environment.now)
        return DeviceState(
            device_id=self.device.id,
            queue_length=len(self.resource.queue),
            in_service=self.resource.count,
            queue_capacity=self.device.queue_capacity,
            processing_capacity=self.device.processing_capacity,
            active_remaining_ms=tuple(
                max(0.0, completion_ms - now_ms)
                for completion_ms in self._active_completion_ms.values()
            ),
            queued_inference_times_ms=tuple(
                work.inference_time_ms
                for work in self._pending_work.values()
                if work.ready_time_ms <= now_ms
            ),
            pending_work=tuple(self._pending_work.values()),
        )


class SimulationEngine:
    """Run fixed traces using externally supplied assignment decisions."""

    def __init__(
        self,
        *,
        devices: DeviceCatalog,
        network: NetworkCatalog,
        model_profiles: Sequence[ModelProfile],
        config: SimulationConfig,
        prediction_provider: PredictionProvider,
        inference_duration_provider: InferenceDurationProvider | None = None,
        runtime_conditions: RuntimeConditions | None = None,
    ) -> None:
        if (
            len(devices.devices) != 3
            or {device.id for device in devices.devices} != DEFAULT_DEVICE_IDS
        ):
            raise ValueError(
                "the full simulation engine requires exactly mobile, gateway, edge-server"
            )
        if any(profile.profile_kind != "measured" for profile in model_profiles):
            raise ValueError("the full simulation engine accepts measured model profiles only")
        roles = [profile.computational_role for profile in model_profiles]
        if len(model_profiles) != 3 or None in roles or len(set(roles)) != 3:
            raise ValueError("exactly one measured profile is required for each model role")
        if len({profile.model_id for profile in model_profiles}) != len(model_profiles):
            raise ValueError("measured model profile IDs must be unique")
        if len(network.links) != 2:
            raise ValueError("the full simulation engine requires two mobile-to-remote links")

        self._devices = devices
        self._network = network
        self._profiles = list(model_profiles)
        self._config = config
        self._prediction_provider = prediction_provider
        self._inference_duration_provider = inference_duration_provider
        self._runtime_conditions = runtime_conditions or RuntimeConditions()
        self._devices_by_id = {device.id: device for device in devices.devices}
        self._profiles_by_id = {profile.model_id: profile for profile in model_profiles}
        for destination in ("gateway", "edge-server"):
            find_network_link(network.links, "mobile", destination)

    def _state(
        self,
        environment: simpy.Environment,
        runtimes: dict[str, _DeviceRuntime],
    ) -> SimulationState:
        return SimulationState(
            current_time_ms=float(environment.now),
            devices=tuple(runtimes[device.id].snapshot() for device in self._devices.devices),
            model_profiles=tuple(self._profiles),
            network_links=self._runtime_conditions.network_links_at(
                float(environment.now), self._network.links
            ),
            device_profiles=tuple(self._devices.devices),
            simulation_config=self._config,
        )

    def _decision_error(
        self,
        request: InferenceRequest,
        decision: AssignmentDecision,
    ) -> str | None:
        if decision.request_id != request.request_id:
            return "assignment_request_id_mismatch"
        device = self._devices_by_id.get(decision.device_id)
        if device is None:
            return "unknown_assignment_device"
        profile = self._profiles_by_id.get(decision.model_id)
        if profile is None:
            return "unknown_assignment_model"
        if not is_device_model_compatible(device, profile):
            return "incompatible_device_model"
        if profile.accuracy < request.minimum_accuracy:
            return "model_accuracy_below_request_minimum"
        if request.source_device_id != device.id:
            try:
                find_network_link(self._network.links, request.source_device_id, device.id)
            except ValueError:
                return "missing_network_link"
        return None

    def run(
        self,
        trace: WorkloadTrace,
        assignment_provider: AssignmentProvider,
    ) -> SimulationRunResult:
        reset = getattr(assignment_provider, "reset", None)
        if callable(reset):
            reset()
        self._runtime_conditions.reset()
        environment = simpy.Environment()
        runtimes = {
            device.id: _DeviceRuntime(environment, device) for device in self._devices.devices
        }
        recorder = _EventRecorder()
        results: dict[str, RequestExecutionResult] = {}

        for request in trace.requests:
            if request.source_device_id != "mobile":
                raise ValueError(f"request {request.request_id} must originate from mobile")

        def rejected_result(
            request: InferenceRequest,
            decision: AssignmentDecision,
            reason: str,
            *,
            upload_time_ms: float = 0.0,
            network_energy: EnergyBreakdown | None = None,
        ) -> None:
            timestamp_ms = float(environment.now)
            energy = network_energy or EnergyBreakdown(
                compute_energy_units=0.0,
                network_energy_units=0.0,
                estimated_total_energy_units=0.0,
            )
            recorder.record(
                timestamp_ms=timestamp_ms,
                event_type=EventType.REQUEST_REJECTED,
                request_id=request.request_id,
                device_id=decision.device_id,
                model_id=decision.model_id,
                details={"reason": reason},
            )
            results[request.request_id] = RequestExecutionResult(
                request_id=request.request_id,
                status=RequestStatus.REJECTED,
                assignment=decision,
                arrival_time_ms=request.arrival_time_ms,
                absolute_deadline_ms=request.absolute_deadline_ms,
                upload_time_ms=upload_time_ms,
                queue_wait_ms=0.0,
                inference_time_ms=0.0,
                return_time_ms=0.0,
                scheduler_overhead_ms=self._config.scheduler_overhead_ms,
                deadline_met=False,
                energy=energy,
                rejection_reason=reason,
            )

        def process_request(
            request: InferenceRequest,
        ) -> Generator[simpy.events.Event, Any, None]:
            yield environment.timeout(request.arrival_time_ms)
            recorder.record(
                timestamp_ms=float(environment.now),
                event_type=EventType.REQUEST_ARRIVED,
                request_id=request.request_id,
                device_id=request.source_device_id,
                details={
                    "deadline_ms": request.deadline_ms,
                    "minimum_accuracy": request.minimum_accuracy,
                    "feature_vector_id": request.feature_vector_id,
                },
            )
            state = self._state(environment, runtimes)
            decision = assignment_provider.select_assignment(request, state)
            error = self._decision_error(request, decision)
            device: Device | None = None
            profile: ModelProfile | None = None
            predicted_timing: NetworkTiming | None = None
            predicted_inference_time_ms = 0.0
            runtime: _DeviceRuntime | None = None
            if error is None:
                device = self._devices_by_id[decision.device_id]
                profile = self._profiles_by_id[decision.model_id]
                predicted_timing = network_timing(
                    source_device_id=request.source_device_id,
                    destination_device_id=device.id,
                    input_size_bytes=request.input_size_bytes,
                    return_size_bytes=self._config.return_payload_size_bytes,
                    links=state.network_links,
                    random_seed=self._config.random_seed,
                    request_id=request.request_id,
                )
                predicted_inference_time_ms = next(
                    (
                        candidate.predicted_inference_time_ms
                        for candidate in decision.candidates
                        if candidate.device_id == device.id
                        and candidate.model_id == profile.model_id
                    ),
                    simulated_inference_time_ms(profile, device),
                )
                runtime = runtimes[device.id]
                runtime.reserve(
                    request.request_id,
                    ready_time_ms=(
                        float(environment.now)
                        + self._config.scheduler_overhead_ms
                        + predicted_timing.upload_time_ms
                    ),
                    service_time_ms=predicted_inference_time_ms,
                )
            if self._config.scheduler_overhead_ms:
                yield environment.timeout(self._config.scheduler_overhead_ms)
            recorder.record(
                timestamp_ms=float(environment.now),
                event_type=EventType.SCHEDULER_DECISION,
                request_id=request.request_id,
                device_id=decision.device_id,
                model_id=decision.model_id,
                details={
                    "decision_request_id": decision.request_id,
                    "predicted_completion_ms": decision.predicted_completion_ms,
                    "predicted_energy_units": decision.predicted_energy_units,
                    "predicted_model_accuracy": decision.predicted_model_accuracy,
                    "expected_to_meet_deadline": decision.expected_to_meet_deadline,
                    "decision_reason": decision.decision_reason,
                },
            )
            if error is not None:
                rejected_result(request, decision, error)
                return

            assert device is not None
            assert profile is not None
            assert predicted_timing is not None
            assert runtime is not None
            upload_timing = network_timing(
                source_device_id=request.source_device_id,
                destination_device_id=device.id,
                input_size_bytes=request.input_size_bytes,
                return_size_bytes=self._config.return_payload_size_bytes,
                links=self._runtime_conditions.network_links_at(
                    float(environment.now), self._network.links
                ),
                random_seed=self._config.random_seed,
                request_id=request.request_id,
            )
            runtime.update_ready_time(
                request.request_id,
                float(environment.now) + upload_timing.upload_time_ms,
            )
            if upload_timing.upload_time_ms:
                yield environment.timeout(upload_timing.upload_time_ms)
            if not runtime.has_queue_space():
                runtime.cancel(request.request_id)
                transferred_bytes = (
                    0 if request.source_device_id == device.id else request.input_size_bytes
                )
                energy = estimate_energy(
                    device=device,
                    inference_time_ms=0.0,
                    transferred_bytes=transferred_bytes,
                    network_energy_per_kilobyte=self._config.network_energy_per_kilobyte,
                )
                rejected_result(
                    request,
                    decision,
                    "device_queue_full",
                    upload_time_ms=upload_timing.upload_time_ms,
                    network_energy=energy,
                )
                return

            queue_entered_ms = float(environment.now)
            with runtime.resource.request() as resource_request:
                recorder.record(
                    timestamp_ms=queue_entered_ms,
                    event_type=EventType.REQUEST_QUEUED,
                    request_id=request.request_id,
                    device_id=device.id,
                    model_id=profile.model_id,
                    details={"queue_length": len(runtime.resource.queue)},
                )
                yield resource_request
                runtime.start(request.request_id)
                started_ms = float(environment.now)
                queue_wait_ms = started_ms - queue_entered_ms
                static_inference_time_ms = simulated_inference_time_ms(profile, device)
                inference_time_ms = (
                    static_inference_time_ms
                    if self._inference_duration_provider is None
                    else self._inference_duration_provider.inference_time_ms(
                        request,
                        device,
                        profile,
                        static_inference_time_ms,
                    )
                )
                inference_time_ms *= self._runtime_conditions.service_time_multiplier(
                    started_ms, device.id
                )
                if inference_time_ms <= 0.0:
                    raise ValueError("simulated inference duration must be positive")
                recorder.record(
                    timestamp_ms=started_ms,
                    event_type=EventType.INFERENCE_STARTED,
                    request_id=request.request_id,
                    device_id=device.id,
                    model_id=profile.model_id,
                    details={
                        "queue_wait_ms": queue_wait_ms,
                        "measured_mean_latency_ms": profile.local_latency_ms_mean,
                        "device_speed_multiplier": device.speed_multiplier,
                        "predicted_inference_time_ms": predicted_inference_time_ms,
                        "simulated_inference_time_ms": inference_time_ms,
                    },
                )
                yield environment.timeout(inference_time_ms)
                runtime.complete(request.request_id)
                prediction = self._prediction_provider.predict(
                    profile.model_id, request.feature_vector_id
                )
                prediction_correct = prediction == request.true_label
                recorder.record(
                    timestamp_ms=float(environment.now),
                    event_type=EventType.INFERENCE_COMPLETED,
                    request_id=request.request_id,
                    device_id=device.id,
                    model_id=profile.model_id,
                    details={
                        "inference_time_ms": inference_time_ms,
                        "actual_prediction": prediction,
                        "prediction_correct": prediction_correct,
                    },
                )
                observe_execution = getattr(assignment_provider, "observe_execution", None)
                if callable(observe_execution):
                    observation = ExecutionObservation(
                        request_id=request.request_id,
                        timestamp_ms=float(environment.now),
                        device_id=device.id,
                        model_id=profile.model_id,
                        inference_time_ms=inference_time_ms,
                    )
                    update = observe_execution(observation)
                    if update is not None:
                        if (
                            update.request_id != observation.request_id
                            or update.timestamp_ms != observation.timestamp_ms
                            or update.device_id != observation.device_id
                            or update.model_id != observation.model_id
                            or update.observed_inference_time_ms != observation.inference_time_ms
                        ):
                            raise ValueError("scheduler returned an inconsistent profile update")
                        recorder.record(
                            timestamp_ms=update.timestamp_ms,
                            event_type=EventType.PROFILE_UPDATED,
                            request_id=update.request_id,
                            device_id=update.device_id,
                            model_id=update.model_id,
                            details={
                                "old_estimate_ms": update.old_estimate_ms,
                                "observed_inference_time_ms": (update.observed_inference_time_ms),
                                "new_estimate_ms": update.new_estimate_ms,
                                "alpha": update.alpha,
                            },
                        )

            return_timing = network_timing(
                source_device_id=request.source_device_id,
                destination_device_id=device.id,
                input_size_bytes=request.input_size_bytes,
                return_size_bytes=self._config.return_payload_size_bytes,
                links=self._runtime_conditions.network_links_at(
                    float(environment.now), self._network.links
                ),
                random_seed=self._config.random_seed,
                request_id=request.request_id,
            )
            if return_timing.return_time_ms:
                yield environment.timeout(return_timing.return_time_ms)
            completion_time_ms = float(environment.now)
            end_to_end_latency_ms = completion_time_ms - request.arrival_time_ms
            deadline_met = meets_deadline(request, completion_time_ms)
            transferred_bytes = (
                0
                if request.source_device_id == device.id
                else request.input_size_bytes + self._config.return_payload_size_bytes
            )
            energy = estimate_energy(
                device=device,
                inference_time_ms=inference_time_ms,
                transferred_bytes=transferred_bytes,
                network_energy_per_kilobyte=self._config.network_energy_per_kilobyte,
            )
            recorder.record(
                timestamp_ms=completion_time_ms,
                event_type=EventType.REQUEST_RETURNED,
                request_id=request.request_id,
                device_id=device.id,
                model_id=profile.model_id,
                details={
                    "upload_time_ms": upload_timing.upload_time_ms,
                    "queue_wait_ms": queue_wait_ms,
                    "inference_time_ms": inference_time_ms,
                    "return_time_ms": return_timing.return_time_ms,
                    "scheduler_overhead_ms": self._config.scheduler_overhead_ms,
                    "end_to_end_latency_ms": end_to_end_latency_ms,
                    "deadline_met": deadline_met,
                    "estimated_energy_units": energy.estimated_total_energy_units,
                },
            )
            if not deadline_met:
                recorder.record(
                    timestamp_ms=completion_time_ms,
                    event_type=EventType.DEADLINE_MISSED,
                    request_id=request.request_id,
                    device_id=device.id,
                    model_id=profile.model_id,
                    details={"absolute_deadline_ms": request.absolute_deadline_ms},
                )
            results[request.request_id] = RequestExecutionResult(
                request_id=request.request_id,
                status=RequestStatus.COMPLETED,
                assignment=decision,
                arrival_time_ms=request.arrival_time_ms,
                absolute_deadline_ms=request.absolute_deadline_ms,
                upload_time_ms=upload_timing.upload_time_ms,
                queue_wait_ms=queue_wait_ms,
                inference_time_ms=inference_time_ms,
                return_time_ms=return_timing.return_time_ms,
                scheduler_overhead_ms=self._config.scheduler_overhead_ms,
                completion_time_ms=completion_time_ms,
                end_to_end_latency_ms=end_to_end_latency_ms,
                deadline_met=deadline_met,
                actual_prediction=prediction,
                prediction_correct=prediction_correct,
                energy=energy,
            )

        ordered_requests = sorted(
            trace.requests, key=lambda request: (request.arrival_time_ms, request.request_id)
        )
        condition_events = sorted(
            self._runtime_conditions.transition_events(),
            key=lambda event: (event.timestamp_ms, event.event_type.value),
        )

        def record_condition_event(
            condition_event: RuntimeConditionEvent,
        ) -> Generator[simpy.events.Event, Any, None]:
            yield environment.timeout(condition_event.timestamp_ms)
            recorder.record(
                timestamp_ms=float(environment.now),
                event_type=condition_event.event_type,
                device_id=condition_event.device_id,
                details=condition_event.details,
            )

        for condition_event in condition_events:
            environment.process(record_condition_event(condition_event))
        for request in ordered_requests:
            environment.process(process_request(request))
        environment.run()

        ordered_results = [results[request.request_id] for request in trace.requests]
        completed = sum(result.status is RequestStatus.COMPLETED for result in ordered_results)
        return SimulationRunResult(
            run_id=self._config.run_id,
            scheduler_name=assignment_provider.name,
            random_seed=self._config.random_seed,
            request_count=len(trace.requests),
            completed_requests=completed,
            rejected_requests=len(trace.requests) - completed,
            requests=ordered_results,
            events=recorder.events,
        )
