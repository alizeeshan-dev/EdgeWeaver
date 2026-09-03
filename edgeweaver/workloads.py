"""Replayable request traces backed by the preserved UCI HAR test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, ValidationError, model_validator

from edgeweaver.domain import DomainModel, InferenceRequest
from edgeweaver.ml.data import UCIHARDataset

TRACE_FORMAT_VERSION = "edgeweaver-workload-trace-v1"
TRACE_GENERATION_VERSION: Literal["edgeweaver-scenario-generator-v1"] = (
    "edgeweaver-scenario-generator-v1"
)
UCI_HAR_DATASET_NAME = "UCI Human Activity Recognition Using Smartphones"


class WorkloadTraceError(ValueError):
    """Raised when a workload trace cannot be parsed or resolved against its dataset."""


class TraceGenerationMetadata(DomainModel):
    """Compact provenance for a generated, scheduler-independent scenario trace."""

    scenario_id: str = Field(min_length=1)
    generation_config_version: Literal["edgeweaver-scenario-generator-v1"] = (
        "edgeweaver-scenario-generator-v1"
    )
    scenario_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_duration_ms: float = Field(gt=0.0)
    request_count: int = Field(ge=1)


class WorkloadTrace(DomainModel):
    """Ordered, scheduler-independent requests referencing the UCI HAR test split."""

    format_version: Literal["edgeweaver-workload-trace-v1"] = "edgeweaver-workload-trace-v1"
    trace_id: str = Field(min_length=1)
    dataset_name: Literal["UCI Human Activity Recognition Using Smartphones"] = (
        "UCI Human Activity Recognition Using Smartphones"
    )
    dataset_split: Literal["test"] = "test"
    seed: int = Field(ge=0)
    generation: TraceGenerationMetadata | None = None
    requests: list[InferenceRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def requests_have_stable_identity_and_order(self) -> Self:
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("workload request IDs must be unique")

        arrival_times = [request.arrival_time_ms for request in self.requests]
        if arrival_times != sorted(arrival_times):
            raise ValueError("workload requests must be ordered by nondecreasing arrival_time_ms")
        if self.generation is not None:
            if self.generation.request_count != len(self.requests):
                raise ValueError("trace generation request_count must match requests")
            if any(
                request.arrival_time_ms >= self.generation.simulation_duration_ms
                for request in self.requests
            ):
                raise ValueError("generated request arrivals must be within simulation duration")
        return self


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WorkloadTraceError(f"workload trace JSON contains duplicate key: {key!r}")
        result[key] = value
    return result


def save_workload_trace(path: Path, trace: WorkloadTrace) -> None:
    """Save a trace in a stable, human-readable JSON representation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(trace.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_workload_trace(
    path: Path,
    *,
    dataset: UCIHARDataset | None = None,
) -> WorkloadTrace:
    """Load a strict trace and optionally verify every test-sample reference."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda value: _raise_invalid_json_constant(value),
        )
        trace = WorkloadTrace.model_validate(payload, strict=True)
    except OSError as error:
        raise WorkloadTraceError(f"could not read workload trace {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise WorkloadTraceError(f"invalid workload trace JSON at {path}: {error}") from error
    except ValidationError as error:
        raise WorkloadTraceError(f"invalid workload trace schema at {path}: {error}") from error

    if dataset is not None:
        validate_trace_against_dataset(trace, dataset)
    return trace


def _raise_invalid_json_constant(value: str) -> None:
    raise WorkloadTraceError(f"workload trace JSON contains invalid constant: {value}")


def validate_trace_against_dataset(trace: WorkloadTrace, dataset: UCIHARDataset) -> None:
    """Ensure IDs point to held-out samples and recorded labels match those samples."""

    test_row_count = dataset.test.features.shape[0]
    for request in trace.requests:
        sample_index = request.feature_vector_id
        if sample_index >= test_row_count:
            raise WorkloadTraceError(
                f"request {request.request_id!r} feature_vector_id {sample_index} is outside "
                f"the UCI HAR test split with {test_row_count} rows"
            )
        dataset_label = int(dataset.test.labels[sample_index])
        if request.true_label != dataset_label:
            raise WorkloadTraceError(
                f"request {request.request_id!r} true_label {request.true_label} does not "
                f"match UCI HAR test label {dataset_label} at feature_vector_id {sample_index}"
            )


def resolve_test_sample(
    dataset: UCIHARDataset,
    feature_vector_id: int,
) -> tuple[NDArray[np.float64], int]:
    """Return one held-out feature row in inference-ready 2D form and its label."""

    row_count = dataset.test.features.shape[0]
    if feature_vector_id < 0 or feature_vector_id >= row_count:
        raise WorkloadTraceError(
            f"feature_vector_id {feature_vector_id} is outside the UCI HAR test split "
            f"with {row_count} rows"
        )
    sample = dataset.test.features[feature_vector_id : feature_vector_id + 1]
    return sample, int(dataset.test.labels[feature_vector_id])


def build_representative_trace(
    dataset: UCIHARDataset,
    *,
    trace_id: str = "phase3-representative",
    seed: int = 2025,
    request_count: int = 6,
    source_device_id: str = "mobile",
) -> WorkloadTrace:
    """Build a small deterministic Phase 3 trace, not a scenario workload generator."""

    if request_count < 2:
        raise WorkloadTraceError("a representative trace requires at least two requests")
    test_row_count, feature_count = dataset.test.features.shape
    if test_row_count < 2:
        raise WorkloadTraceError("a representative trace requires at least two test samples")

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    while len(selected_indices) < request_count:
        selected_indices.extend(int(index) for index in rng.permutation(test_row_count))
    selected_indices = selected_indices[:request_count]

    arrival_pattern = (0.0, 4.0, 11.0, 11.0, 23.0, 41.0)
    deadline_pattern = (25.0, 60.0, 140.0, 45.0, 90.0, 220.0)
    accuracy_pattern = (0.80, 0.90, 0.94, 0.85, 0.92, 0.95)
    input_size_bytes = feature_count * dataset.test.features.dtype.itemsize
    requests = [
        InferenceRequest(
            request_id=f"request-{position + 1:04d}",
            arrival_time_ms=(
                arrival_pattern[position % len(arrival_pattern)]
                + (position // len(arrival_pattern)) * 50.0
            ),
            deadline_ms=deadline_pattern[position % len(deadline_pattern)],
            minimum_accuracy=accuracy_pattern[position % len(accuracy_pattern)],
            input_size_bytes=input_size_bytes,
            source_device_id=source_device_id,
            true_label=int(dataset.test.labels[sample_index]),
            feature_vector_id=sample_index,
        )
        for position, sample_index in enumerate(selected_indices)
    ]
    trace = WorkloadTrace(trace_id=trace_id, seed=seed, requests=requests)
    validate_trace_against_dataset(trace, dataset)
    return trace
