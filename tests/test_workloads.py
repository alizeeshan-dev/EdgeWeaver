import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from edgeweaver.domain import InferenceRequest
from edgeweaver.ml.data import load_uci_har
from edgeweaver.workloads import (
    WorkloadTrace,
    WorkloadTraceError,
    build_representative_trace,
    load_workload_trace,
    resolve_test_sample,
    save_workload_trace,
    validate_trace_against_dataset,
)


def _request(
    request_id: str,
    arrival_time_ms: float,
    feature_vector_id: int,
    true_label: int,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        arrival_time_ms=arrival_time_ms,
        deadline_ms=50.0,
        minimum_accuracy=0.8,
        input_size_bytes=24,
        source_device_id="mobile",
        true_label=true_label,
        feature_vector_id=feature_vector_id,
    )


def test_trace_requires_unique_ids_and_canonical_arrival_order() -> None:
    first = _request("same", 1.0, 0, 1)
    second = _request("same", 2.0, 1, 2)
    with pytest.raises(ValidationError, match="request IDs must be unique"):
        WorkloadTrace(trace_id="duplicate", seed=1, requests=[first, second])

    later = _request("later", 2.0, 0, 1)
    earlier = _request("earlier", 1.0, 1, 2)
    with pytest.raises(ValidationError, match="nondecreasing arrival_time_ms"):
        WorkloadTrace(trace_id="unordered", seed=1, requests=[later, earlier])


def test_trace_round_trip_preserves_request_order(
    mini_uci_har_root: Path,
    tmp_path: Path,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)
    trace = WorkloadTrace(
        trace_id="round-trip",
        seed=7,
        requests=[_request("first", 0.0, 0, 1), _request("second", 0.0, 1, 2)],
    )
    path = tmp_path / "nested" / "trace.json"

    save_workload_trace(path, trace)
    loaded = load_workload_trace(path, dataset=dataset)

    assert loaded == trace
    assert [request.request_id for request in loaded.requests] == ["first", "second"]
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"].endswith("v1")


def test_dataset_validation_rejects_missing_sample_and_wrong_label(
    mini_uci_har_root: Path,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)
    missing = WorkloadTrace(
        trace_id="missing",
        seed=1,
        requests=[_request("outside", 0.0, 2, 1)],
    )
    with pytest.raises(WorkloadTraceError, match="outside the UCI HAR test split"):
        validate_trace_against_dataset(missing, dataset)

    mislabeled = WorkloadTrace(
        trace_id="mislabeled",
        seed=1,
        requests=[_request("wrong-label", 0.0, 0, 2)],
    )
    with pytest.raises(WorkloadTraceError, match="does not match UCI HAR test label 1"):
        validate_trace_against_dataset(mislabeled, dataset)


def test_representative_trace_is_deterministic_and_resolvable(
    mini_uci_har_root: Path,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)

    first = build_representative_trace(dataset, seed=42, request_count=6)
    second = build_representative_trace(dataset, seed=42, request_count=6)

    assert first == second
    assert len(first.requests) == 6
    assert len({request.feature_vector_id for request in first.requests}) == 2
    assert len({request.arrival_time_ms for request in first.requests}) > 1
    assert len({request.deadline_ms for request in first.requests}) > 1
    assert len({request.minimum_accuracy for request in first.requests}) > 1
    for request in first.requests:
        sample, label = resolve_test_sample(dataset, request.feature_vector_id)
        assert sample.shape == (1, dataset.feature_count)
        assert label == request.true_label


def test_load_rejects_unknown_schema_fields_and_duplicate_json_keys(tmp_path: Path) -> None:
    unknown_field = tmp_path / "unknown.json"
    unknown_field.write_text(
        json.dumps(
            {
                "format_version": "edgeweaver-workload-trace-v1",
                "trace_id": "bad",
                "dataset_name": "UCI Human Activity Recognition Using Smartphones",
                "dataset_split": "test",
                "seed": 1,
                "requests": [],
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkloadTraceError, match="invalid workload trace schema"):
        load_workload_trace(unknown_field)

    duplicate_key = tmp_path / "duplicate-key.json"
    duplicate_key.write_text('{"trace_id":"one","trace_id":"two"}', encoding="utf-8")
    with pytest.raises(WorkloadTraceError, match="duplicate key"):
        load_workload_trace(duplicate_key)
