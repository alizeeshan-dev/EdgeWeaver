from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from edgeweaver.domain import (
    AssignmentDecision,
    EnergyBreakdown,
    InferenceRequest,
    RequestExecutionResult,
    RequestStatus,
    SimulationRunResult,
)
from edgeweaver.events import EventLog
from edgeweaver.experiments import (
    CompletedRunRecord,
    CoreExperimentConfig,
    ExperimentPaths,
    ExperimentRunner,
    ExperimentRunSpec,
    ExperimentValidationError,
    atomic_write_json,
    build_ablation_matrix,
    build_core_matrix,
    load_experiment_config,
    select_specs,
)
from edgeweaver.metrics import PerRunMetrics
from edgeweaver.workloads import WorkloadTrace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_runner(tmp_path: Path) -> tuple[CoreExperimentConfig, ExperimentRunner]:
    for relative in (
        "configs/devices.yaml",
        "configs/network.yaml",
        "configs/simulation.yaml",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {relative}\n", encoding="utf-8")
    for scenario in ("normal", "bursty", "network_slowdown", "device_slowdown"):
        path = tmp_path / "configs" / "scenarios" / f"{scenario}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"scenario_id: {scenario}\n", encoding="utf-8")
    for relative in ("artifacts/profiles/profile.json", "artifacts/models/model.bin"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    dataset_marker = tmp_path / "data" / "raw" / "UCI HAR Dataset" / "fixture.txt"
    dataset_marker.parent.mkdir(parents=True, exist_ok=True)
    dataset_marker.write_text("controlled fixture dataset", encoding="utf-8")

    base = load_experiment_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    config = base.model_copy(
        update={
            "paths": ExperimentPaths(
                output_root=Path("output"),
            )
        }
    )
    return config, ExperimentRunner(config, project_root=tmp_path)


def _trace(seed: int = 1) -> WorkloadTrace:
    return WorkloadTrace(
        trace_id=f"normal-seed-{seed}",
        seed=seed,
        requests=[
            InferenceRequest(
                request_id="request-1",
                arrival_time_ms=0.0,
                deadline_ms=10.0,
                minimum_accuracy=0.5,
                input_size_bytes=1,
                source_device_id="mobile",
                true_label=1,
                feature_vector_id=0,
            )
        ],
    )


def _rejected_run(spec: ExperimentRunSpec) -> SimulationRunResult:
    assignment = AssignmentDecision(
        request_id="request-1",
        device_id="mobile",
        model_id="model-1",
        predicted_completion_ms=1.0,
        predicted_energy_units=1.0,
        predicted_model_accuracy=0.9,
        expected_to_meet_deadline=True,
        decision_reason="fixture",
    )
    request = RequestExecutionResult(
        request_id="request-1",
        status=RequestStatus.REJECTED,
        assignment=assignment,
        arrival_time_ms=0.0,
        absolute_deadline_ms=10.0,
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
        rejection_reason="fixture",
    )
    return SimulationRunResult(
        run_id=spec.run_id,
        scheduler_name=spec.scheduler_name,
        random_seed=spec.seed,
        request_count=1,
        completed_requests=0,
        rejected_requests=1,
        requests=[request],
        events=[],
    )


def _empty_summary(spec: ExperimentRunSpec) -> PerRunMetrics:
    return PerRunMetrics(
        run_id=spec.run_id,
        scenario_id=spec.scenario_id,
        scheduler_name=spec.scheduler_name,
        seed=spec.seed,
        simulation_duration_ms=10.0,
        total_requests=1,
        assigned_requests=1,
        completed_requests=0,
        rejected_requests=1,
        deadlines_met=0,
        deadline_misses=0,
        useful_valid_predictions=0,
        correct_predictions=0,
        deadline_satisfaction_rate=0.0,
        useful_goodput_requests_per_second=0.0,
        mean_end_to_end_latency_ms=0.0,
        p95_end_to_end_latency_ms=0.0,
        actual_prediction_accuracy=0.0,
        total_estimated_energy_units=0.0,
        estimated_energy_per_completed_request_units=0.0,
        device_utilization=[],
        model_selection_distribution=[],
        deadline_miss_causes=[],
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_core_and_ablation_matrices_have_exact_unique_scope() -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs" / "experiment.yaml")

    core = build_core_matrix(config)
    ablations = build_ablation_matrix(config)

    assert len(core) == 80
    assert len({spec.run_id for spec in core}) == 80
    assert {spec.scheduler_name for spec in core} == set(config.schedulers)
    assert {spec.scenario_id for spec in core} == set(config.scenarios)
    assert {spec.seed for spec in core} == set(config.seeds)
    assert len(ablations) == 10
    assert {(spec.scheduler_name, spec.scenario_id) for spec in ablations} == {
        ("edgeweaver_no_model_switching", "bursty"),
        ("edgeweaver_no_online_update", "device_slowdown"),
    }


def test_filters_are_explicit_and_do_not_change_matrix_generation() -> None:
    config = load_experiment_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    selected = select_specs(
        build_core_matrix(config),
        scenarios=["normal"],
        schedulers=["edgeweaver"],
        seeds=[3],
    )
    assert [(spec.scenario_id, spec.scheduler_name, spec.seed) for spec in selected] == [
        ("normal", "edgeweaver", 3)
    ]


def test_all_schedulers_share_one_trace_path_for_scenario_and_seed(tmp_path: Path) -> None:
    config, runner = _configured_runner(tmp_path)
    specs = [
        spec
        for spec in build_core_matrix(config)
        if spec.scenario_id == "bursty" and spec.seed == 2
    ]

    assert len({runner.paths_for(spec).trace for spec in specs}) == 1


def test_run_identity_changes_when_controlled_dataset_changes(tmp_path: Path) -> None:
    config, runner = _configured_runner(tmp_path)
    spec = build_core_matrix(config)[0]
    trace = _trace(spec.seed)
    atomic_write_json(runner.paths_for(spec).trace, trace)
    original_identity = runner._identity(spec, trace)

    dataset_marker = tmp_path / "data" / "raw" / "UCI HAR Dataset" / "fixture.txt"
    dataset_marker.write_text("changed controlled fixture dataset", encoding="utf-8")
    changed_runner = ExperimentRunner(config, project_root=tmp_path)

    assert changed_runner._identity(spec, trace) != original_identity
    assert changed_runner._config_digests["source"] == runner._config_digests["source"]


def test_completed_artifacts_are_validated_and_corruption_forces_rerun(
    tmp_path: Path,
) -> None:
    config, runner = _configured_runner(tmp_path)
    spec = build_core_matrix(config)[0]
    trace = _trace(spec.seed)
    paths = runner.paths_for(spec)
    atomic_write_json(paths.trace, trace)
    run = _rejected_run(spec)
    event_log = EventLog(
        run_id=spec.run_id,
        scenario_id=spec.scenario_id,
        scheduler_name=spec.scheduler_name,
        seed=spec.seed,
        events=[],
    )
    summary = _empty_summary(spec)
    atomic_write_json(paths.raw_run, run)
    atomic_write_json(paths.event_log, event_log)
    atomic_write_json(paths.summary_json, summary)
    paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.summary_csv.write_text("run_id,metric\nfixture,value\n", encoding="utf-8")
    artifacts = {
        "raw_run": paths.raw_run,
        "event_log": paths.event_log,
        "summary_json": paths.summary_json,
        "summary_csv": paths.summary_csv,
    }
    record = CompletedRunRecord(
        experiment_id=config.experiment_id,
        code_version=config.code_version,
        run_id=spec.run_id,
        kind=spec.kind,
        scenario_id=spec.scenario_id,
        scheduler_name=spec.scheduler_name,
        seed=spec.seed,
        identity_sha256=runner._identity(spec, trace),
        trace_id=trace.trace_id,
        trace_reference=paths.trace.relative_to(tmp_path).as_posix(),
        trace_sha256=_hash(paths.trace),
        scenario_config_sha256=_hash(
            tmp_path / "configs" / "scenarios" / f"{spec.scenario_id}.yaml"
        ),
        devices_config_sha256=runner._config_digests["devices"],
        network_config_sha256=runner._config_digests["network"],
        simulation_config_sha256=runner._config_digests["simulation"],
        model_profiles_sha256=runner._config_digests["profiles"],
        model_artifacts_sha256=runner._config_digests["models"],
        dataset_sha256=runner._config_digests["dataset"],
        source_sha256=runner._config_digests["source"],
        runtime_versions=runner._runtime_versions,
        model_profile_ids=("model-1",),
        configuration_references={},
        request_count=1,
        simulation_duration_ms=10.0,
        executed_at_utc=datetime.now(UTC),
        artifact_references={
            name: path.relative_to(tmp_path).as_posix() for name, path in artifacts.items()
        },
        artifact_sha256={name: _hash(path) for name, path in artifacts.items()},
    )
    atomic_write_json(paths.completion_record, record)

    assert runner.is_complete(spec, trace)
    paths.summary_json.write_text("{", encoding="utf-8")
    assert not runner.is_complete(spec, trace)


class _StubRunner(ExperimentRunner):
    def __init__(self, *args: Any, complete: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.complete = complete
        self.executions = 0

    def prepare_trace(self, scenario_id: Any, seed: int) -> WorkloadTrace:
        return _trace(seed)

    def is_complete(self, spec: ExperimentRunSpec, trace: WorkloadTrace) -> bool:
        return self.complete

    def _execute(self, spec: ExperimentRunSpec, trace: WorkloadTrace) -> CompletedRunRecord:
        self.executions += 1
        raise AssertionError("test stub should not construct a record")


def test_resume_skips_completed_run_and_force_requests_execution(tmp_path: Path) -> None:
    config, _ = _configured_runner(tmp_path)
    spec = build_core_matrix(config)[0]
    runner = _StubRunner(config, project_root=tmp_path, complete=True)

    report = runner.run([spec])
    assert (report.executed_runs, report.skipped_runs, runner.executions) == (0, 1, 0)

    with pytest.raises(AssertionError, match="test stub"):
        runner.run([spec], force=True)
    assert runner.executions == 1


def test_corrupt_or_incomplete_run_is_executed(tmp_path: Path) -> None:
    config, _ = _configured_runner(tmp_path)
    spec = build_core_matrix(config)[0]
    runner = _StubRunner(config, project_root=tmp_path, complete=False)

    with pytest.raises(AssertionError, match="test stub"):
        runner.run([spec])
    assert runner.executions == 1


def test_dry_run_never_prepares_inputs_or_writes(tmp_path: Path) -> None:
    config, runner = _configured_runner(tmp_path)
    specs = build_core_matrix(config)

    report = runner.run(specs, dry_run=True)

    assert report.planned_runs == 80
    assert report.executed_runs == report.skipped_runs == 0
    assert not runner.output_root.exists()


def test_completeness_validation_rejects_missing_runs(tmp_path: Path) -> None:
    config, _ = _configured_runner(tmp_path)
    runner = _StubRunner(config, project_root=tmp_path, complete=False)

    with pytest.raises(ExperimentValidationError, match="missing or invalid completed run"):
        runner.validate_matrix(include_ablations=False)
