import csv
import json
from pathlib import Path

import pytest

from edgeweaver.domain import (
    AssignmentDecision,
    CandidateEstimate,
    Device,
    EnergyBreakdown,
    InferenceRequest,
    ModelProfile,
    ModelRole,
    RequestExecutionResult,
    RequestStatus,
    SimulationRunResult,
)
from edgeweaver.events import event_log_from_run, save_event_log
from edgeweaver.metrics import (
    DeadlineMissCause,
    PerRunMetrics,
    calculate_run_metrics,
    classify_deadline_miss,
    save_run_metrics_csv,
    save_run_metrics_json,
)
from edgeweaver.workloads import WorkloadTrace


def _device(device_id: str) -> Device:
    return Device(
        id=device_id,
        speed_multiplier=1.0,
        active_power_units=1.0,
        queue_capacity=5,
        supported_models=["light", "balanced", "heavy"],
    )


def _profile(model_id: str, role: ModelRole) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        display_name=model_id,
        accuracy=0.9,
        local_latency_ms_mean=10.0,
        profile_kind="synthetic_fixture",
        computational_role=role,
    )


def _request(request_id: str, *, minimum_accuracy: float = 0.8) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        arrival_time_ms=0.0,
        deadline_ms=100.0,
        minimum_accuracy=minimum_accuracy,
        input_size_bytes=100,
        source_device_id="mobile",
        true_label=1,
        feature_vector_id=int(request_id.split("-")[-1]),
    )


def _candidate(
    *, device_id: str, model_id: str, predicted_inference_ms: float
) -> CandidateEstimate:
    return CandidateEstimate(
        device_id=device_id,
        model_id=model_id,
        model_accuracy=0.9,
        compatible=True,
        accuracy_eligible=True,
        network_available=True,
        eligible=True,
        queue_admissible=True,
        predicted_upload_time_ms=0.0,
        predicted_queue_wait_ms=0.0,
        predicted_inference_time_ms=predicted_inference_ms,
        predicted_return_time_ms=0.0,
        scheduler_overhead_ms=0.0,
        predicted_completion_ms=predicted_inference_ms,
        predicted_energy_units=predicted_inference_ms,
        expected_to_meet_deadline=True,
    )


def _decision(
    request_id: str,
    *,
    device_id: str = "mobile",
    model_id: str = "light-v1",
    predicted_completion_ms: float = 50.0,
    predicted_inference_ms: float | None = None,
    expected_to_meet_deadline: bool = True,
    predicted_accuracy: float = 0.9,
) -> AssignmentDecision:
    candidates: tuple[CandidateEstimate, ...] = ()
    if predicted_inference_ms is not None:
        candidates = (
            _candidate(
                device_id=device_id,
                model_id=model_id,
                predicted_inference_ms=predicted_inference_ms,
            ),
        )
    return AssignmentDecision(
        request_id=request_id,
        device_id=device_id,
        model_id=model_id,
        predicted_completion_ms=predicted_completion_ms,
        predicted_energy_units=10.0,
        predicted_model_accuracy=predicted_accuracy,
        expected_to_meet_deadline=expected_to_meet_deadline,
        decision_reason="fixture",
        candidates=candidates,
    )


def _completed(
    request_id: str,
    *,
    upload: float,
    queue: float,
    inference: float,
    returned: float,
    deadline_ms: float,
    correct: bool,
    energy: float,
    decision: AssignmentDecision | None = None,
    arrival: float = 0.0,
    overhead: float = 0.0,
) -> RequestExecutionResult:
    latency = overhead + upload + queue + inference + returned
    return RequestExecutionResult(
        request_id=request_id,
        status=RequestStatus.COMPLETED,
        assignment=decision or _decision(request_id),
        arrival_time_ms=arrival,
        absolute_deadline_ms=arrival + deadline_ms,
        upload_time_ms=upload,
        queue_wait_ms=queue,
        inference_time_ms=inference,
        return_time_ms=returned,
        scheduler_overhead_ms=overhead,
        completion_time_ms=arrival + latency,
        end_to_end_latency_ms=latency,
        deadline_met=latency <= deadline_ms,
        actual_prediction=1 if correct else 2,
        prediction_correct=correct,
        energy=EnergyBreakdown(
            compute_energy_units=energy,
            network_energy_units=0.0,
            estimated_total_energy_units=energy,
        ),
    )


def _rejected(request_id: str) -> RequestExecutionResult:
    return RequestExecutionResult(
        request_id=request_id,
        status=RequestStatus.REJECTED,
        assignment=_decision(request_id, model_id="balanced-v1"),
        arrival_time_ms=0.0,
        absolute_deadline_ms=100.0,
        upload_time_ms=0.0,
        queue_wait_ms=0.0,
        inference_time_ms=0.0,
        return_time_ms=0.0,
        scheduler_overhead_ms=0.0,
        deadline_met=False,
        energy=EnergyBreakdown(
            compute_energy_units=0.0,
            network_energy_units=0.0,
            estimated_total_energy_units=0.0,
        ),
        rejection_reason="fixture rejection",
    )


def _run(results: list[RequestExecutionResult]) -> SimulationRunResult:
    completed = sum(result.status is RequestStatus.COMPLETED for result in results)
    return SimulationRunResult(
        run_id="run-1",
        scheduler_name="edgeweaver",
        random_seed=42,
        request_count=len(results),
        completed_requests=completed,
        rejected_requests=len(results) - completed,
        requests=results,
        events=[],
    )


def _trace(count: int, *, minimum_accuracy: float = 0.8) -> WorkloadTrace:
    return WorkloadTrace(
        trace_id="trace-1",
        seed=42,
        requests=[
            _request(f"request-{index}", minimum_accuracy=minimum_accuracy)
            for index in range(count)
        ],
    )


def _calculate(
    run: SimulationRunResult, trace: WorkloadTrace, *, duration: float = 100.0
) -> PerRunMetrics:
    return calculate_run_metrics(
        run,
        trace,
        scenario_id="normal",
        simulation_duration_ms=duration,
        devices=[_device("mobile"), _device("gateway")],
        model_profiles=[
            _profile("light-v1", ModelRole.LIGHT),
            _profile("balanced-v1", ModelRole.BALANCED),
        ],
    )


def test_required_metrics_use_completed_request_denominators_and_linear_p95() -> None:
    results = [
        _completed(
            "request-0",
            upload=0.0,
            queue=0.0,
            inference=10.0,
            returned=0.0,
            deadline_ms=20.0,
            correct=True,
            energy=20.0,
        ),
        _completed(
            "request-1",
            upload=5.0,
            queue=10.0,
            inference=20.0,
            returned=5.0,
            deadline_ms=30.0,
            correct=False,
            energy=40.0,
            decision=_decision("request-1", device_id="gateway"),
        ),
        _rejected("request-2"),
    ]
    summary = _calculate(_run(results), _trace(3), duration=1000.0)

    assert summary.completed_requests == 2
    assert summary.rejected_requests == 1
    assert summary.deadline_satisfaction_rate == 0.5
    assert summary.useful_valid_predictions == 1
    assert summary.useful_goodput_requests_per_second == 1.0
    assert summary.mean_end_to_end_latency_ms == 25.0
    assert summary.p95_end_to_end_latency_ms == pytest.approx(38.5)
    assert summary.actual_prediction_accuracy == 0.5
    assert summary.total_estimated_energy_units == 60.0
    assert summary.estimated_energy_per_completed_request_units == 30.0


def test_goodput_requires_actual_correctness_and_profile_accuracy_eligibility() -> None:
    results = [
        _completed(
            "request-0",
            upload=0.0,
            queue=0.0,
            inference=10.0,
            returned=0.0,
            deadline_ms=20.0,
            correct=True,
            energy=1.0,
            decision=_decision("request-0", predicted_accuracy=0.85),
        ),
        _completed(
            "request-1",
            upload=0.0,
            queue=0.0,
            inference=10.0,
            returned=0.0,
            deadline_ms=20.0,
            correct=False,
            energy=1.0,
        ),
    ]
    summary = _calculate(_run(results), _trace(2, minimum_accuracy=0.9), duration=2000.0)

    assert summary.deadlines_met == 2
    assert summary.correct_predictions == 1
    assert summary.useful_valid_predictions == 0
    assert summary.useful_goodput_requests_per_second == 0.0


def test_device_utilization_uses_compute_busy_time_only_and_merges_overlap() -> None:
    results = [
        _completed(
            "request-0",
            upload=20.0,
            queue=0.0,
            inference=50.0,
            returned=10.0,
            deadline_ms=100.0,
            correct=True,
            energy=1.0,
        ),
        _completed(
            "request-1",
            upload=0.0,
            queue=25.0,
            inference=50.0,
            returned=0.0,
            deadline_ms=100.0,
            correct=True,
            energy=1.0,
        ),
    ]
    summary = _calculate(_run(results), _trace(2), duration=100.0)
    utilization = {item.device_id: item for item in summary.device_utilization}

    # Compute intervals [20, 70] and [25, 75] overlap, so union busy time is 55 ms.
    assert utilization["mobile"].busy_time_ms == 55.0
    assert utilization["mobile"].utilization == 0.55
    assert utilization["gateway"].busy_time_ms == 0.0
    assert utilization["gateway"].utilization == 0.0


def test_model_distribution_uses_explicit_assigned_request_denominator() -> None:
    results = [
        _completed(
            "request-0",
            upload=0.0,
            queue=0.0,
            inference=10.0,
            returned=0.0,
            deadline_ms=20.0,
            correct=True,
            energy=1.0,
        ),
        _rejected("request-1"),
    ]
    summary = _calculate(_run(results), _trace(2))
    distribution = {item.model_id: item for item in summary.model_selection_distribution}

    assert summary.assigned_requests == 2
    assert distribution["light-v1"].assigned_requests == 1
    assert distribution["light-v1"].percentage == 50.0
    assert distribution["balanced-v1"].assigned_requests == 1
    assert distribution["balanced-v1"].percentage == 50.0


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ((30.0, 5.0, 10.0, 20.0), DeadlineMissCause.NETWORK_DELAY),
        ((5.0, 50.0, 10.0, 5.0), DeadlineMissCause.QUEUE_DELAY),
        ((5.0, 10.0, 50.0, 5.0), DeadlineMissCause.INFERENCE_TIME),
    ],
)
def test_deadline_miss_dominant_component_classification(
    components: tuple[float, float, float, float], expected: DeadlineMissCause
) -> None:
    upload, queue, inference, returned = components
    result = _completed(
        "request-0",
        upload=upload,
        queue=queue,
        inference=inference,
        returned=returned,
        deadline_ms=40.0,
        correct=True,
        energy=1.0,
        decision=_decision("request-0", expected_to_meet_deadline=False),
    )
    assert classify_deadline_miss(result) is expected


def test_incorrect_estimate_requires_inference_error_to_explain_miss() -> None:
    attributable = _completed(
        "request-0",
        upload=5.0,
        queue=0.0,
        inference=50.0,
        returned=5.0,
        deadline_ms=40.0,
        correct=True,
        energy=1.0,
        decision=_decision("request-0", predicted_completion_ms=30.0, predicted_inference_ms=20.0),
    )
    not_attributable = _completed(
        "request-1",
        upload=30.0,
        queue=20.0,
        inference=30.0,
        returned=30.0,
        deadline_ms=50.0,
        correct=True,
        energy=1.0,
        decision=_decision("request-1", predicted_completion_ms=40.0, predicted_inference_ms=20.0),
    )

    assert classify_deadline_miss(attributable) is DeadlineMissCause.INCORRECT_STATIC_ESTIMATE
    assert classify_deadline_miss(not_attributable) is DeadlineMissCause.NETWORK_DELAY


def test_zero_completion_metrics_are_explicit_and_deterministic() -> None:
    summary = _calculate(_run([_rejected("request-0")]), _trace(1))

    assert summary.completed_requests == 0
    assert summary.deadline_satisfaction_rate == 0.0
    assert summary.useful_goodput_requests_per_second == 0.0
    assert summary.mean_end_to_end_latency_ms == 0.0
    assert summary.p95_end_to_end_latency_ms == 0.0
    assert summary.actual_prediction_accuracy == 0.0
    assert summary.estimated_energy_per_completed_request_units == 0.0
    assert all(metric.rate == 0.0 for metric in summary.deadline_miss_causes)


def test_trace_and_run_request_identity_must_match() -> None:
    with pytest.raises(ValueError, match="identical request IDs"):
        _calculate(_run([_rejected("request-0")]), _trace(2))


def test_json_and_tidy_csv_exports_are_stable(tmp_path: Path) -> None:
    result = _completed(
        "request-0",
        upload=0.0,
        queue=0.0,
        inference=10.0,
        returned=0.0,
        deadline_ms=20.0,
        correct=True,
        energy=4.0,
    )
    summary = _calculate(_run([result]), _trace(1))
    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"

    save_run_metrics_json(json_path, summary)
    save_run_metrics_csv(csv_path, summary)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == "edgeweaver-run-metrics-v1"
    assert payload["estimated_energy_per_completed_request_units"] == 4.0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "run_id",
        "scenario_id",
        "scheduler_name",
        "seed",
        "metric",
        "dimension",
        "value",
        "unit",
    ]
    assert any(
        row["metric"] == "device_utilization" and row["dimension"] == "mobile" for row in rows
    )
    assert any(
        row["metric"] == "deadline_miss_cause_count"
        and row["dimension"] == "incorrect_static_estimate"
        for row in rows
    )


def test_raw_event_log_export_is_separate_from_summary(tmp_path: Path) -> None:
    run = _run([_rejected("request-0")])
    event_log = event_log_from_run(run, scenario_id="normal")
    path = tmp_path / "events.json"
    save_event_log(path, event_log)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "format_version": "edgeweaver-event-log-v1",
        "run_id": "run-1",
        "scenario_id": "normal",
        "scheduler_name": "edgeweaver",
        "seed": 42,
        "events": [],
    }
