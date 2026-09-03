"""Validated domain objects shared by configuration and simulation code."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonScalar = str | int | float | bool | None


class DomainModel(BaseModel):
    """Common strict, immutable behaviour for EdgeWeaver domain values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRole(StrEnum):
    LIGHT = "light"
    BALANCED = "balanced"
    HEAVY = "heavy"


class ModelProfile(DomainModel):
    model_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    model_size_bytes: int | None = Field(default=None, ge=0)
    local_latency_ms_mean: float = Field(gt=0.0)
    local_latency_ms_median: float | None = Field(default=None, gt=0.0)
    local_latency_ms_std: float | None = Field(default=None, ge=0.0)
    local_latency_ms_p95: float | None = Field(default=None, gt=0.0)
    profile_kind: Literal["synthetic_fixture", "measured"]
    computational_role: ModelRole | None = None
    profiling_prediction_count: int | None = Field(default=None, ge=1)
    profiling_seed: int | None = Field(default=None, ge=0)
    raw_latency_observations_file: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def measured_profiles_are_complete(self) -> Self:
        if self.profile_kind != "measured":
            return self
        required_values = {
            "macro_f1": self.macro_f1,
            "model_size_bytes": self.model_size_bytes,
            "local_latency_ms_median": self.local_latency_ms_median,
            "local_latency_ms_std": self.local_latency_ms_std,
            "local_latency_ms_p95": self.local_latency_ms_p95,
            "computational_role": self.computational_role,
            "profiling_prediction_count": self.profiling_prediction_count,
            "profiling_seed": self.profiling_seed,
            "raw_latency_observations_file": self.raw_latency_observations_file,
        }
        missing = [name for name, value in required_values.items() if value is None]
        if missing:
            raise ValueError(f"measured model profile is missing: {', '.join(missing)}")
        if self.model_size_bytes == 0:
            raise ValueError("measured model profile must have a non-empty model artifact")
        if self.profiling_prediction_count is not None and self.profiling_prediction_count < 500:
            raise ValueError("measured profiles require at least 500 timed predictions")
        return self


class Device(DomainModel):
    id: str = Field(min_length=1)
    speed_multiplier: float = Field(gt=0.0)
    active_power_units: float = Field(ge=0.0)
    queue_capacity: int = Field(ge=0)
    processing_capacity: int = Field(default=1, ge=1)
    supported_models: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def supported_models_are_unique(self) -> Self:
        if len(self.supported_models) != len(set(self.supported_models)):
            raise ValueError("supported_models must not contain duplicates")
        return self


class NetworkLink(DomainModel):
    link_id: str = Field(min_length=1)
    source_device_id: str = Field(min_length=1)
    destination_device_id: str = Field(min_length=1)
    base_latency_ms: float = Field(ge=0.0)
    bandwidth_mbps: float = Field(gt=0.0)
    jitter_ms: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def endpoints_are_different(self) -> Self:
        if self.source_device_id == self.destination_device_id:
            raise ValueError("network link endpoints must be different")
        return self


class InferenceRequest(DomainModel):
    request_id: str = Field(min_length=1)
    arrival_time_ms: float = Field(ge=0.0)
    deadline_ms: float = Field(gt=0.0)
    minimum_accuracy: float = Field(ge=0.0, le=1.0)
    input_size_bytes: int = Field(ge=0)
    source_device_id: str = Field(min_length=1)
    true_label: int
    feature_vector_id: int = Field(ge=0)

    @property
    def absolute_deadline_ms(self) -> float:
        """Absolute simulation time by which this request must complete."""

        return self.arrival_time_ms + self.deadline_ms


class CandidateEstimate(DomainModel):
    """Scheduler-side estimate for one device/model candidate.

    ``predicted_completion_ms`` is an absolute simulation timestamp. Energy values
    are normalized estimates, not physical measurements.
    """

    device_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_accuracy: float = Field(ge=0.0, le=1.0)
    compatible: bool
    accuracy_eligible: bool
    network_available: bool
    eligible: bool
    rejection_reason: str | None = None
    queue_admissible: bool
    predicted_upload_time_ms: float = Field(ge=0.0)
    predicted_queue_wait_ms: float = Field(ge=0.0)
    predicted_inference_time_ms: float = Field(ge=0.0)
    predicted_return_time_ms: float = Field(ge=0.0)
    scheduler_overhead_ms: float = Field(ge=0.0)
    predicted_completion_ms: float = Field(ge=0.0)
    predicted_energy_units: float = Field(ge=0.0)
    expected_to_meet_deadline: bool

    @model_validator(mode="after")
    def eligibility_matches_rejection(self) -> Self:
        expected_eligibility = self.compatible and self.accuracy_eligible and self.network_available
        if self.eligible != expected_eligibility:
            raise ValueError("eligibility must match compatibility, accuracy, and network status")
        if self.eligible == (self.rejection_reason is not None):
            raise ValueError("eligibility and rejection reason are inconsistent")
        return self


class AssignmentDecision(DomainModel):
    request_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    predicted_completion_ms: float = Field(ge=0.0)
    predicted_energy_units: float = Field(ge=0.0)
    predicted_model_accuracy: float = Field(ge=0.0, le=1.0)
    expected_to_meet_deadline: bool
    decision_reason: str = Field(min_length=1)
    candidates: tuple[CandidateEstimate, ...] = ()


class ScenarioConfig(DomainModel):
    scenario_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    device_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    requests: list[InferenceRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def request_ids_are_unique(self) -> Self:
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique within a scenario")
        return self


class EventType(StrEnum):
    REQUEST_ARRIVED = "REQUEST_ARRIVED"
    SCHEDULER_DECISION = "SCHEDULER_DECISION"
    REQUEST_QUEUED = "REQUEST_QUEUED"
    INFERENCE_STARTED = "INFERENCE_STARTED"
    INFERENCE_COMPLETED = "INFERENCE_COMPLETED"
    REQUEST_RETURNED = "REQUEST_RETURNED"
    DEADLINE_MISSED = "DEADLINE_MISSED"
    REQUEST_REJECTED = "REQUEST_REJECTED"
    DEVICE_SLOWDOWN_STARTED = "DEVICE_SLOWDOWN_STARTED"
    DEVICE_SLOWDOWN_ENDED = "DEVICE_SLOWDOWN_ENDED"
    NETWORK_SLOWDOWN_STARTED = "NETWORK_SLOWDOWN_STARTED"
    NETWORK_SLOWDOWN_ENDED = "NETWORK_SLOWDOWN_ENDED"
    PROFILE_UPDATED = "PROFILE_UPDATED"


class StructuredEvent(DomainModel):
    sequence: int = Field(ge=0)
    timestamp_ms: float = Field(ge=0.0)
    event_type: EventType
    request_id: str | None = None
    device_id: str | None = None
    model_id: str | None = None
    details: dict[str, JsonScalar] = Field(default_factory=dict)


class RequestMetrics(DomainModel):
    request_id: str
    queue_wait_ms: float = Field(ge=0.0)
    completion_time_ms: float = Field(ge=0.0)
    end_to_end_latency_ms: float = Field(ge=0.0)
    absolute_deadline_ms: float = Field(ge=0.0)
    deadline_met: bool


class SimulationSummary(DomainModel):
    scenario_id: str
    total_requests: int = Field(ge=0)
    deadlines_met: int = Field(ge=0)
    deadline_satisfaction_rate: float = Field(ge=0.0, le=1.0)
    requests: list[RequestMetrics]


class FixtureSimulationResult(DomainModel):
    events: list[StructuredEvent]
    assignments: list[AssignmentDecision]
    summary: SimulationSummary


class SimulationConfig(DomainModel):
    run_id: str = Field(min_length=1)
    random_seed: int = Field(ge=0)
    scheduler_overhead_ms: float = Field(default=0.0, ge=0.0)
    network_energy_per_kilobyte: float = Field(default=0.0, ge=0.0)
    return_payload_size_bytes: int = Field(default=64, ge=0)
    representative_latency_statistic: Literal["mean"] = "mean"
    edgeweaver_ewma_alpha: float = Field(default=0.2, gt=0.0, le=1.0)


class ExecutionObservation(DomainModel):
    """Completed inference-service observation supplied to adaptive schedulers."""

    request_id: str = Field(min_length=1)
    timestamp_ms: float = Field(ge=0.0)
    device_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    inference_time_ms: float = Field(gt=0.0)


class LatencyEstimateUpdate(DomainModel):
    """One scheduler-side latency estimate update; static profiles are unchanged."""

    request_id: str = Field(min_length=1)
    timestamp_ms: float = Field(ge=0.0)
    device_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    old_estimate_ms: float = Field(gt=0.0)
    observed_inference_time_ms: float = Field(gt=0.0)
    new_estimate_ms: float = Field(gt=0.0)
    alpha: float = Field(gt=0.0, le=1.0)


class PendingServiceWork(DomainModel):
    """Assigned inference work that has not started on its device."""

    request_id: str = Field(min_length=1)
    ready_time_ms: float = Field(ge=0.0)
    inference_time_ms: float = Field(ge=0.0)
    sequence: int = Field(ge=0)


class DeviceState(DomainModel):
    device_id: str
    queue_length: int = Field(ge=0)
    in_service: int = Field(ge=0)
    queue_capacity: int = Field(ge=0)
    processing_capacity: int = Field(ge=1)
    active_remaining_ms: tuple[float, ...] = ()
    queued_inference_times_ms: tuple[float, ...] = ()
    pending_work: tuple[PendingServiceWork, ...] = ()

    @model_validator(mode="after")
    def service_work_is_valid(self) -> Self:
        if self.in_service > self.processing_capacity:
            raise ValueError("in-service count exceeds processing capacity")
        if len(self.active_remaining_ms) > self.in_service:
            raise ValueError("active service work exceeds in-service count")
        if len(self.active_remaining_ms) > self.processing_capacity:
            raise ValueError("active service work exceeds processing capacity")
        if any(value < 0.0 for value in self.active_remaining_ms):
            raise ValueError("active remaining durations must be non-negative")
        if any(value < 0.0 for value in self.queued_inference_times_ms):
            raise ValueError("queued inference durations must be non-negative")
        return self


class SimulationState(DomainModel):
    current_time_ms: float = Field(ge=0.0)
    devices: tuple[DeviceState, ...]
    model_profiles: tuple[ModelProfile, ...]
    network_links: tuple[NetworkLink, ...]
    device_profiles: tuple[Device, ...] = ()
    simulation_config: SimulationConfig | None = None


class RequestStatus(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"


class EnergyBreakdown(DomainModel):
    compute_energy_units: float = Field(ge=0.0)
    network_energy_units: float = Field(ge=0.0)
    estimated_total_energy_units: float = Field(ge=0.0)

    @model_validator(mode="after")
    def total_matches_components(self) -> Self:
        expected = self.compute_energy_units + self.network_energy_units
        if abs(self.estimated_total_energy_units - expected) > 1e-12:
            raise ValueError("estimated total energy must equal compute plus network energy")
        return self


class RequestExecutionResult(DomainModel):
    request_id: str
    status: RequestStatus
    assignment: AssignmentDecision
    arrival_time_ms: float = Field(ge=0.0)
    absolute_deadline_ms: float = Field(ge=0.0)
    upload_time_ms: float = Field(ge=0.0)
    queue_wait_ms: float = Field(ge=0.0)
    inference_time_ms: float = Field(ge=0.0)
    return_time_ms: float = Field(ge=0.0)
    scheduler_overhead_ms: float = Field(ge=0.0)
    completion_time_ms: float | None = Field(default=None, ge=0.0)
    end_to_end_latency_ms: float | None = Field(default=None, ge=0.0)
    deadline_met: bool
    actual_prediction: int | None = None
    prediction_correct: bool | None = None
    energy: EnergyBreakdown
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> Self:
        if self.status is RequestStatus.COMPLETED:
            completed_values = (
                self.completion_time_ms,
                self.end_to_end_latency_ms,
                self.actual_prediction,
                self.prediction_correct,
            )
            if any(value is None for value in completed_values):
                raise ValueError("completed request results require timing and prediction outputs")
            if self.rejection_reason is not None:
                raise ValueError("completed request results cannot have a rejection reason")
            assert self.completion_time_ms is not None
            assert self.end_to_end_latency_ms is not None
            component_total = (
                self.scheduler_overhead_ms
                + self.upload_time_ms
                + self.queue_wait_ms
                + self.inference_time_ms
                + self.return_time_ms
            )
            if abs(self.end_to_end_latency_ms - component_total) > 1e-9:
                raise ValueError("end-to-end latency must equal its timing components")
            expected_completion = self.arrival_time_ms + self.end_to_end_latency_ms
            if abs(self.completion_time_ms - expected_completion) > 1e-9:
                raise ValueError("completion time must equal arrival plus end-to-end latency")
        elif not self.rejection_reason:
            raise ValueError("rejected request results require a rejection reason")
        return self


class SimulationRunResult(DomainModel):
    run_id: str
    scheduler_name: str
    random_seed: int = Field(ge=0)
    request_count: int = Field(ge=0)
    completed_requests: int = Field(ge=0)
    rejected_requests: int = Field(ge=0)
    requests: list[RequestExecutionResult]
    events: list[StructuredEvent]

    @model_validator(mode="after")
    def counts_and_event_order_are_consistent(self) -> Self:
        if self.request_count != len(self.requests):
            raise ValueError("request_count must match request results")
        completed = sum(request.status is RequestStatus.COMPLETED for request in self.requests)
        rejected = sum(request.status is RequestStatus.REJECTED for request in self.requests)
        if (self.completed_requests, self.rejected_requests) != (completed, rejected):
            raise ValueError("completed/rejected counts must match request statuses")
        if completed + rejected != self.request_count:
            raise ValueError("each request must have exactly one terminal status")
        if [event.sequence for event in self.events] != list(range(len(self.events))):
            raise ValueError("event sequences must be contiguous")
        timestamps = [event.timestamp_ms for event in self.events]
        if timestamps != sorted(timestamps):
            raise ValueError("events must be ordered by nondecreasing timestamp")
        return self
