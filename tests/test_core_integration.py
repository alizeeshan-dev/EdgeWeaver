from pathlib import Path

from edgeweaver.config import (
    load_device_catalog,
    load_network_catalog,
    load_research_scenario,
    load_simulation_config,
)
from edgeweaver.domain import InferenceRequest
from edgeweaver.engine import SimulationEngine
from edgeweaver.events import event_log_from_run, save_event_log
from edgeweaver.metrics import calculate_run_metrics, save_run_metrics_csv, save_run_metrics_json
from edgeweaver.ml.artifacts import write_domain_json
from edgeweaver.ml.profile import load_measured_profiles
from edgeweaver.runtime_conditions import ScenarioRuntimeConditions
from edgeweaver.schedulers import SCHEDULER_NAMES, create_scheduler
from edgeweaver.workloads import WorkloadTrace, load_workload_trace, save_workload_trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FixturePredictor:
    def predict(self, model_id: str, feature_vector_id: int) -> int:
        assert model_id
        return feature_vector_id + 1


def test_small_core_workflow_runs_all_schedulers_and_exports_metrics(tmp_path: Path) -> None:
    devices = load_device_catalog(PROJECT_ROOT / "configs" / "devices.yaml")
    network = load_network_catalog(PROJECT_ROOT / "configs" / "network.yaml")
    profiles = list(load_measured_profiles(PROJECT_ROOT / "artifacts" / "profiles").values())
    scenario = load_research_scenario(PROJECT_ROOT / "configs" / "scenarios" / "normal.yaml")
    scenario = scenario.model_copy(update={"simulation_duration_ms": 100.0})
    base_config = load_simulation_config(PROJECT_ROOT / "configs" / "simulation.yaml")
    trace = WorkloadTrace(
        trace_id="core-integration-fixture",
        seed=23,
        requests=[
            InferenceRequest(
                request_id=f"request-{index}",
                arrival_time_ms=float(index * 3),
                deadline_ms=100.0,
                minimum_accuracy=0.7,
                input_size_bytes=16,
                source_device_id="mobile",
                true_label=index + 1,
                feature_vector_id=index,
            )
            for index in range(3)
        ],
    )
    trace_path = tmp_path / "trace.json"
    save_workload_trace(trace_path, trace)
    loaded_trace = load_workload_trace(trace_path)

    for scheduler_name in SCHEDULER_NAMES:
        config = base_config.model_copy(
            update={"run_id": f"integration-{scheduler_name}", "random_seed": trace.seed}
        )
        result = SimulationEngine(
            devices=devices,
            network=network,
            model_profiles=profiles,
            config=config,
            prediction_provider=_FixturePredictor(),
            runtime_conditions=ScenarioRuntimeConditions(scenario, devices, network),
        ).run(loaded_trace, create_scheduler(scheduler_name))
        summary = calculate_run_metrics(
            result,
            loaded_trace,
            scenario_id=scenario.scenario_id,
            simulation_duration_ms=scenario.simulation_duration_ms,
            devices=devices.devices,
            model_profiles=profiles,
        )

        run_path = tmp_path / scheduler_name / "run.json"
        events_path = tmp_path / scheduler_name / "events.json"
        summary_path = tmp_path / scheduler_name / "summary.json"
        summary_csv_path = tmp_path / scheduler_name / "summary.csv"
        write_domain_json(run_path, result)
        save_event_log(events_path, event_log_from_run(result, scenario_id=scenario.scenario_id))
        save_run_metrics_json(summary_path, summary)
        save_run_metrics_csv(summary_csv_path, summary)

        assert result.completed_requests == 3
        assert summary.completed_requests == 3
        assert summary.deadline_satisfaction_rate == 1.0
        assert summary.actual_prediction_accuracy == 1.0
        assert summary.device_utilization
        assert summary.model_selection_distribution
        assert summary.deadline_miss_causes
        exported_paths = (run_path, events_path, summary_path, summary_csv_path)
        assert all(path.is_file() for path in exported_paths)
