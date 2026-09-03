import json
from pathlib import Path

from edgeweaver.domain import (
    AssignmentDecision,
    CandidateEstimate,
    EnergyBreakdown,
    EventType,
    InferenceRequest,
    ModelRole,
    RequestExecutionResult,
    RequestStatus,
    SimulationRunResult,
    StructuredEvent,
)
from edgeweaver.metrics import (
    DeadlineMissCause,
    DeadlineMissCauseMetric,
    DeviceUtilizationMetric,
    ModelSelectionMetric,
    PerRunMetrics,
)
from edgeweaver.reporting import (
    ABLATION_VARIANTS,
    CORE_SCENARIOS,
    CORE_SCHEDULERS,
    ReportingRun,
    extract_failure_cases,
    generate_findings_summary,
    generate_research_figures,
    load_reporting_runs_from_experiment,
    load_run_metrics,
    save_failure_cases,
)
from edgeweaver.workloads import WorkloadTrace


def _summary(scenario: str, scheduler: str, seed: int) -> PerRunMetrics:
    burst_penalty = 0.15 if scenario == "bursty" and scheduler == "fastest_device" else 0.0
    no_switch_penalty = 0.08 if scheduler == "edgeweaver_no_model_switching" else 0.0
    no_update_penalty = 0.10 if scheduler == "edgeweaver_no_online_update" else 0.0
    deadline_rate = 0.92 - burst_penalty - no_switch_penalty - no_update_penalty
    latency = 30.0 + burst_penalty * 200.0 + no_update_penalty * 100.0 + seed
    energy = 5.0 + (1.0 if scheduler == "edgeweaver_no_model_switching" else 0.0)
    return PerRunMetrics(
        run_id=f"{scenario}-{scheduler}-{seed}",
        scenario_id=scenario,
        scheduler_name=scheduler,
        seed=seed,
        simulation_duration_ms=1000.0,
        total_requests=10,
        assigned_requests=10,
        completed_requests=10,
        rejected_requests=0,
        deadlines_met=round(deadline_rate * 10),
        deadline_misses=10 - round(deadline_rate * 10),
        useful_valid_predictions=8,
        correct_predictions=9,
        deadline_satisfaction_rate=deadline_rate,
        useful_goodput_requests_per_second=8.0 - no_switch_penalty,
        mean_end_to_end_latency_ms=latency,
        p95_end_to_end_latency_ms=latency * 1.5,
        actual_prediction_accuracy=0.9,
        total_estimated_energy_units=energy * 10.0,
        estimated_energy_per_completed_request_units=energy,
        device_utilization=[
            DeviceUtilizationMetric(device_id="mobile", busy_time_ms=200.0, utilization=0.2),
            DeviceUtilizationMetric(device_id="gateway", busy_time_ms=300.0, utilization=0.3),
            DeviceUtilizationMetric(device_id="edge-server", busy_time_ms=400.0, utilization=0.4),
        ],
        model_selection_distribution=[
            ModelSelectionMetric(
                model_id="light-v1",
                model_role=ModelRole.LIGHT,
                assigned_requests=(0 if scheduler == "edgeweaver_no_model_switching" else 5),
                percentage=(0.0 if scheduler == "edgeweaver_no_model_switching" else 50.0),
            ),
            ModelSelectionMetric(
                model_id="balanced-v1",
                model_role=ModelRole.BALANCED,
                assigned_requests=(0 if scheduler == "edgeweaver_no_model_switching" else 3),
                percentage=(0.0 if scheduler == "edgeweaver_no_model_switching" else 30.0),
            ),
            ModelSelectionMetric(
                model_id="heavy-v1",
                model_role=ModelRole.HEAVY,
                assigned_requests=(10 if scheduler == "edgeweaver_no_model_switching" else 2),
                percentage=(100.0 if scheduler == "edgeweaver_no_model_switching" else 20.0),
            ),
        ],
        deadline_miss_causes=[
            DeadlineMissCauseMetric(cause=cause, count=0, rate=0.0) for cause in DeadlineMissCause
        ],
    )


def _all_summaries() -> list[PerRunMetrics]:
    summaries = [
        _summary(scenario, scheduler, seed)
        for scenario in CORE_SCENARIOS
        for scheduler in CORE_SCHEDULERS
        for seed in (1, 2)
    ]
    summaries.extend(_summary("bursty", ABLATION_VARIANTS[0], seed) for seed in (1, 2))
    summaries.extend(_summary("device_slowdown", ABLATION_VARIANTS[1], seed) for seed in (1, 2))
    return summaries


def test_generates_all_eight_required_chart_groups_from_saved_metrics(tmp_path: Path) -> None:
    paths = generate_research_figures(_all_summaries(), tmp_path)

    assert set(paths) == {
        "deadline_satisfaction",
        "useful_goodput",
        "latency",
        "actual_prediction_accuracy",
        "estimated_energy",
        "device_utilization",
        "model_selection",
        "edgeweaver_ablations",
    }
    assert all(path.suffix == ".png" and path.stat().st_size > 1000 for path in paths.values())


def test_metrics_can_be_reloaded_in_stable_path_order(tmp_path: Path) -> None:
    first = _summary("normal", "edgeweaver", 1)
    second = _summary("normal", "edgeweaver", 2)
    (tmp_path / "b.json").write_text(second.model_dump_json(), encoding="utf-8")
    (tmp_path / "a.json").write_text(first.model_dump_json(), encoding="utf-8")

    loaded = load_run_metrics(tmp_path.glob("*.json"))

    assert [summary.seed for summary in loaded] == [1, 2]


def _impossible_reporting_run() -> ReportingRun:
    request = InferenceRequest(
        request_id="request-1",
        arrival_time_ms=0.0,
        deadline_ms=5.0,
        minimum_accuracy=0.8,
        input_size_bytes=100,
        source_device_id="mobile",
        true_label=1,
        feature_vector_id=0,
    )
    candidate = CandidateEstimate(
        device_id="mobile",
        model_id="light-v1",
        model_accuracy=0.9,
        compatible=True,
        accuracy_eligible=True,
        network_available=True,
        eligible=True,
        queue_admissible=True,
        predicted_upload_time_ms=0.0,
        predicted_queue_wait_ms=0.0,
        predicted_inference_time_ms=10.0,
        predicted_return_time_ms=0.0,
        scheduler_overhead_ms=0.0,
        predicted_completion_ms=10.0,
        predicted_energy_units=10.0,
        expected_to_meet_deadline=False,
    )
    decision = AssignmentDecision(
        request_id=request.request_id,
        device_id="mobile",
        model_id="light-v1",
        predicted_completion_ms=10.0,
        predicted_energy_units=10.0,
        predicted_model_accuracy=0.9,
        expected_to_meet_deadline=False,
        decision_reason="no candidate predicted to meet deadline",
        candidates=(candidate,),
    )
    result = RequestExecutionResult(
        request_id=request.request_id,
        status=RequestStatus.COMPLETED,
        assignment=decision,
        arrival_time_ms=0.0,
        absolute_deadline_ms=5.0,
        upload_time_ms=0.0,
        queue_wait_ms=0.0,
        inference_time_ms=10.0,
        return_time_ms=0.0,
        scheduler_overhead_ms=0.0,
        completion_time_ms=10.0,
        end_to_end_latency_ms=10.0,
        deadline_met=False,
        actual_prediction=1,
        prediction_correct=True,
        energy=EnergyBreakdown(
            compute_energy_units=10.0,
            network_energy_units=0.0,
            estimated_total_energy_units=10.0,
        ),
    )
    run = SimulationRunResult(
        run_id="normal-edgeweaver-1",
        scheduler_name="edgeweaver",
        random_seed=1,
        request_count=1,
        completed_requests=1,
        rejected_requests=0,
        requests=[result],
        events=[
            StructuredEvent(
                sequence=0,
                timestamp_ms=0.0,
                event_type=EventType.SCHEDULER_DECISION,
                request_id=request.request_id,
                device_id="mobile",
                model_id="light-v1",
            )
        ],
    )
    return ReportingRun(
        scenario_id="normal",
        run=run,
        trace=WorkloadTrace(trace_id="trace-1", seed=1, requests=[request]),
    )


def test_failure_case_extraction_marks_absent_cases_instead_of_inventing_them(
    tmp_path: Path,
) -> None:
    report = extract_failure_cases([_impossible_reporting_run()])
    by_id = {case.case_id: case for case in report.cases}

    assert len(report.cases) == 5
    assert by_id["no_assignment_meets_deadline"].observed is True
    evidence = by_id["no_assignment_meets_deadline"].evidence
    assert evidence is not None
    assert evidence.request_deadline_ms == 5.0
    assert evidence.actual_completion_ms == 10.0
    assert evidence.candidates[0].expected_to_meet_deadline is False
    assert by_id["network_slowdown_remote_miss"].observed is False
    assert by_id["network_slowdown_remote_miss"].evidence is None

    path = tmp_path / "failure_cases.json"
    save_failure_cases(path, report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == "edgeweaver-failure-cases-v1"
    assert len(payload["cases"]) == 5


def test_completed_records_join_raw_runs_to_paired_traces(tmp_path: Path) -> None:
    reporting_run = _impossible_reporting_run()
    raw_path = tmp_path / "raw" / "run.json"
    trace_path = tmp_path / "traces" / "trace.json"
    records_path = tmp_path / "records"
    raw_path.parent.mkdir()
    trace_path.parent.mkdir()
    records_path.mkdir()
    raw_path.write_text(reporting_run.run.model_dump_json(), encoding="utf-8")
    trace_path.write_text(reporting_run.trace.model_dump_json(), encoding="utf-8")
    (records_path / "record.json").write_text(
        json.dumps(
            {
                "scenario_id": "normal",
                "trace_reference": "traces/trace.json",
                "artifact_references": {"raw_run": "raw/run.json"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_reporting_runs_from_experiment(
        project_root=tmp_path,
        run_records_directory=records_path,
    )

    assert len(loaded) == 1
    assert loaded[0].run.run_id == reporting_run.run.run_id
    assert loaded[0].trace.trace_id == reporting_run.trace.trace_id


def test_findings_summary_is_cautious_and_uses_actual_values() -> None:
    text = generate_findings_summary(_all_summaries())

    assert all(f"## H{number}" in text for number in range(1, 6))
    assert "no significance tests were performed" in text
    assert "normalized units/request" in text
    assert "No separate aggregate acceptability threshold" in text
    assert "H3 — supported by observed results" in text
