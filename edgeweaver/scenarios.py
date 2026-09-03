"""Validated Phase 6 scenario definitions and deterministic workload generation."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from edgeweaver.domain import DomainModel, InferenceRequest
from edgeweaver.ml.data import UCIHARDataset
from edgeweaver.workloads import (
    TRACE_GENERATION_VERSION,
    TraceGenerationMetadata,
    WorkloadTrace,
    WorkloadTraceError,
    validate_trace_against_dataset,
)

CoreScenarioId = Literal["normal", "bursty", "network_slowdown", "device_slowdown"]


class BurstWindow(DomainModel):
    """One half-open interval with an elevated, constant arrival rate."""

    start_time_ms: float = Field(ge=0.0)
    end_time_ms: float = Field(gt=0.0)
    arrival_rate_per_second: float = Field(gt=0.0)

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end_time_ms <= self.start_time_ms:
            raise ValueError("burst end_time_ms must be greater than start_time_ms")
        return self


class ArrivalConfig(DomainModel):
    """Piecewise-constant Poisson arrival process."""

    base_rate_per_second: float = Field(gt=0.0)
    bursts: tuple[BurstWindow, ...] = ()

    @model_validator(mode="after")
    def bursts_are_ordered_and_nonoverlapping(self) -> Self:
        starts = [burst.start_time_ms for burst in self.bursts]
        if starts != sorted(starts):
            raise ValueError("burst windows must be ordered by start_time_ms")
        for previous, current in zip(self.bursts, self.bursts[1:], strict=False):
            if current.start_time_ms < previous.end_time_ms:
                raise ValueError("burst windows must not overlap")
        return self


class DeadlineDistribution(DomainModel):
    values_ms: tuple[float, ...] = Field(min_length=1)
    weights: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def values_and_weights_are_valid(self) -> Self:
        if len(self.values_ms) != len(self.weights):
            raise ValueError("deadline values and weights must have the same length")
        if any(value <= 0.0 for value in self.values_ms):
            raise ValueError("deadline values must be positive")
        _validate_weights(self.weights, "deadline")
        return self


class MinimumAccuracyDistribution(DomainModel):
    values: tuple[float, ...] = Field(min_length=1)
    weights: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def values_and_weights_are_valid(self) -> Self:
        if len(self.values) != len(self.weights):
            raise ValueError("minimum-accuracy values and weights must have the same length")
        if any(value < 0.0 or value > 1.0 for value in self.values):
            raise ValueError("minimum-accuracy values must be between zero and one")
        _validate_weights(self.weights, "minimum-accuracy")
        return self


def _validate_weights(weights: tuple[float, ...], label: str) -> None:
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError(f"{label} weights must be non-negative with a positive total")


class RequestInputConfig(DomainModel):
    source_device_id: str = Field(default="mobile", min_length=1)
    size_mode: Literal["dataset_row", "fixed"] = "dataset_row"
    fixed_input_size_bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def fixed_size_matches_mode(self) -> Self:
        if self.size_mode == "fixed" and self.fixed_input_size_bytes is None:
            raise ValueError("fixed input-size mode requires fixed_input_size_bytes")
        if self.size_mode == "dataset_row" and self.fixed_input_size_bytes is not None:
            raise ValueError("dataset-row input-size mode cannot set fixed_input_size_bytes")
        return self


class NetworkSlowdownConfig(DomainModel):
    link_id: str = Field(min_length=1)
    start_time_ms: float = Field(ge=0.0)
    end_time_ms: float = Field(gt=0.0)
    base_latency_multiplier: float = Field(gt=0.0)
    bandwidth_multiplier: float = Field(gt=0.0)

    @model_validator(mode="after")
    def interval_and_effect_are_valid(self) -> Self:
        if self.end_time_ms <= self.start_time_ms:
            raise ValueError("network slowdown end_time_ms must follow start_time_ms")
        if self.base_latency_multiplier <= 1.0 and self.bandwidth_multiplier >= 1.0:
            raise ValueError("network slowdown must worsen latency or bandwidth")
        return self


class DeviceSlowdownConfig(DomainModel):
    device_id: str = Field(min_length=1)
    start_time_ms: float = Field(ge=0.0)
    end_time_ms: float = Field(gt=0.0)
    service_time_multiplier: float = Field(gt=1.0)

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end_time_ms <= self.start_time_ms:
            raise ValueError("device slowdown end_time_ms must follow start_time_ms")
        return self


class ResearchScenarioConfig(DomainModel):
    """One editable, scheduler-independent controlled simulation condition."""

    scenario_id: CoreScenarioId
    description: str = Field(min_length=1)
    simulation_duration_ms: float = Field(gt=0.0)
    random_seed: int = Field(ge=0)
    arrival: ArrivalConfig
    deadline_distribution: DeadlineDistribution
    minimum_accuracy_distribution: MinimumAccuracyDistribution
    request_input: RequestInputConfig = RequestInputConfig()
    network_slowdown: NetworkSlowdownConfig | None = None
    device_slowdown: DeviceSlowdownConfig | None = None

    @model_validator(mode="after")
    def scenario_shape_is_consistent(self) -> Self:
        for burst in self.arrival.bursts:
            if burst.end_time_ms > self.simulation_duration_ms:
                raise ValueError("burst window must end within the simulation duration")

        slowdowns = (self.network_slowdown, self.device_slowdown)
        for slowdown in slowdowns:
            if slowdown is not None and slowdown.end_time_ms > self.simulation_duration_ms:
                raise ValueError("slowdown must end within the simulation duration")

        if self.scenario_id == "bursty":
            if not 2 <= len(self.arrival.bursts) <= 3:
                raise ValueError("bursty scenario requires two or three burst windows")
        elif self.arrival.bursts:
            raise ValueError("only the bursty scenario may define burst windows")

        if self.scenario_id == "network_slowdown":
            if self.network_slowdown is None or self.device_slowdown is not None:
                raise ValueError("network_slowdown requires only a network slowdown")
        elif self.scenario_id == "device_slowdown":
            if self.device_slowdown is None or self.network_slowdown is not None:
                raise ValueError("device_slowdown requires only a device slowdown")
        elif any(slowdown is not None for slowdown in slowdowns):
            raise ValueError("normal and bursty scenarios cannot define slowdowns")
        return self


def generate_scenario_trace(
    config: ResearchScenarioConfig,
    dataset: UCIHARDataset,
    *,
    seed: int | None = None,
) -> WorkloadTrace:
    """Generate a trace using only scenario, seed, and held-out dataset state.

    Arrivals form independent Poisson processes in each configured constant-rate
    interval. The function never accepts a scheduler, which keeps paired traces
    independent of the policy that will later replay them.
    """

    effective_seed = config.random_seed if seed is None else seed
    if effective_seed < 0:
        raise WorkloadTraceError("workload seed must be non-negative")
    rng = np.random.default_rng(effective_seed)
    arrivals = _generate_arrival_times(config, rng)
    if not arrivals:
        raise WorkloadTraceError("scenario generated zero requests; increase duration or rate")

    test_row_count = dataset.test.features.shape[0]
    if test_row_count == 0:
        raise WorkloadTraceError("UCI HAR test split contains no samples")
    sample_indices = rng.integers(0, test_row_count, size=len(arrivals))
    deadlines = rng.choice(
        config.deadline_distribution.values_ms,
        size=len(arrivals),
        p=_normalized_weights(config.deadline_distribution.weights),
    )
    minimum_accuracies = rng.choice(
        config.minimum_accuracy_distribution.values,
        size=len(arrivals),
        p=_normalized_weights(config.minimum_accuracy_distribution.weights),
    )
    input_size_bytes = (
        dataset.feature_count * dataset.test.features.dtype.itemsize
        if config.request_input.size_mode == "dataset_row"
        else config.request_input.fixed_input_size_bytes
    )
    assert input_size_bytes is not None

    requests = [
        InferenceRequest(
            request_id=f"request-{position:05d}",
            arrival_time_ms=arrival_time_ms,
            deadline_ms=float(deadlines[position - 1]),
            minimum_accuracy=float(minimum_accuracies[position - 1]),
            input_size_bytes=int(input_size_bytes),
            source_device_id=config.request_input.source_device_id,
            true_label=int(dataset.test.labels[int(sample_indices[position - 1])]),
            feature_vector_id=int(sample_indices[position - 1]),
        )
        for position, arrival_time_ms in enumerate(arrivals, start=1)
    ]
    metadata = TraceGenerationMetadata(
        scenario_id=config.scenario_id,
        generation_config_version=TRACE_GENERATION_VERSION,
        scenario_config_sha256=scenario_config_sha256(config),
        simulation_duration_ms=config.simulation_duration_ms,
        request_count=len(requests),
    )
    trace = WorkloadTrace(
        trace_id=f"{config.scenario_id}-seed-{effective_seed}",
        seed=effective_seed,
        generation=metadata,
        requests=requests,
    )
    validate_trace_against_dataset(trace, dataset)
    return trace


def _normalized_weights(weights: tuple[float, ...]) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    return values / values.sum()


def _generate_arrival_times(
    config: ResearchScenarioConfig,
    rng: np.random.Generator,
) -> list[float]:
    boundaries = {0.0, config.simulation_duration_ms}
    for burst in config.arrival.bursts:
        boundaries.update((burst.start_time_ms, burst.end_time_ms))
    ordered_boundaries = sorted(boundaries)
    arrivals: list[float] = []
    for start_ms, end_ms in zip(ordered_boundaries, ordered_boundaries[1:], strict=False):
        rate = config.arrival.base_rate_per_second
        for burst in config.arrival.bursts:
            if start_ms >= burst.start_time_ms and end_ms <= burst.end_time_ms:
                rate = burst.arrival_rate_per_second
                break
        current_ms = start_ms
        scale_ms = 1000.0 / rate
        while True:
            current_ms += float(rng.exponential(scale_ms))
            if current_ms >= end_ms:
                break
            arrivals.append(current_ms)
    return arrivals


def scenario_config_sha256(config: ResearchScenarioConfig) -> str:
    """Return the stable hash stored with traces generated from this scenario."""

    canonical = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
