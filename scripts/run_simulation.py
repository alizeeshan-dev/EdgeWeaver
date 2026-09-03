"""Run one deterministic EdgeWeaver simulation or one configured Phase 6 scenario."""

import argparse
from pathlib import Path

from edgeweaver.config import (
    load_device_catalog,
    load_network_catalog,
    load_research_scenario,
    load_simulation_config,
)
from edgeweaver.engine import SimulationEngine
from edgeweaver.events import event_log_from_run, save_event_log
from edgeweaver.metrics import (
    calculate_run_metrics,
    save_run_metrics_csv,
    save_run_metrics_json,
)
from edgeweaver.ml.artifacts import write_domain_json
from edgeweaver.ml.data import DATASET_DIRECTORY_NAME, load_uci_har
from edgeweaver.ml.inference import ModelPredictionService
from edgeweaver.ml.profile import load_measured_profiles
from edgeweaver.runtime_conditions import ScenarioRuntimeConditions
from edgeweaver.scenarios import generate_scenario_trace, scenario_config_sha256
from edgeweaver.scheduler import FixedAssignmentProvider, load_assignment_plan
from edgeweaver.schedulers import SCHEDULER_NAMES, create_scheduler
from edgeweaver.workloads import load_workload_trace, save_workload_trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=Path, default=PROJECT_ROOT / "configs" / "devices.yaml")
    parser.add_argument("--network", type=Path, default=PROJECT_ROOT / "configs" / "network.yaml")
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "simulation.yaml",
    )
    parser.add_argument(
        "--profiles-root", type=Path, default=PROJECT_ROOT / "artifacts" / "profiles"
    )
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / DATASET_DIRECTORY_NAME,
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Replay a saved trace; with --scenario it must match that scenario config.",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase3_assignments.json",
    )
    parser.add_argument(
        "--scheduler",
        choices=(*SCHEDULER_NAMES, "fixed"),
        default="round_robin",
        help="Scheduler, or 'fixed' to replay the Phase 3 assignment plan.",
    )
    parser.add_argument(
        "--scenario",
        choices=("normal", "bursty", "network_slowdown", "device_slowdown"),
        default=None,
        help="Generate and run one configured Phase 6 scenario.",
    )
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=None,
        help="Explicit scenario YAML; may be used instead of --scenario.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the scenario seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--events-output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--summary-csv-output", type=Path, default=None)
    args = parser.parse_args()

    dataset = load_uci_har(args.dataset_root)
    profiles = load_measured_profiles(args.profiles_root)
    prediction_service = ModelPredictionService.from_model_directories(
        dataset, args.models_root, list(profiles)
    )
    devices = load_device_catalog(args.devices)
    network = load_network_catalog(args.network)
    base_simulation_config = load_simulation_config(args.simulation_config)

    scenario_requested = args.scenario is not None or args.scenario_config is not None
    if scenario_requested and args.scheduler == "fixed":
        parser.error("Phase 6 scenarios require a registered scheduler, not --scheduler fixed")

    runtime_conditions = None
    scenario_config = None
    generated_trace = False
    if scenario_requested:
        scenario_path = args.scenario_config
        if scenario_path is None:
            assert args.scenario is not None
            scenario_path = PROJECT_ROOT / "configs" / "scenarios" / f"{args.scenario}.yaml"
        scenario_config = load_research_scenario(scenario_path)
        if args.scenario is not None and scenario_config.scenario_id != args.scenario:
            parser.error(
                f"scenario YAML identifies {scenario_config.scenario_id!r}, "
                f"not requested {args.scenario!r}"
            )
        if args.trace is not None:
            trace = load_workload_trace(args.trace, dataset=dataset)
            if trace.generation is None:
                parser.error("a Phase 6 scenario requires trace generation metadata")
            if trace.generation.scenario_id != scenario_config.scenario_id:
                parser.error("saved trace scenario does not match the selected scenario")
            if trace.generation.scenario_config_sha256 != scenario_config_sha256(scenario_config):
                parser.error("saved trace was generated from different scenario configuration")
            if args.seed is not None and args.seed != trace.seed:
                parser.error("--seed does not match the saved trace seed")
            seed = trace.seed
        else:
            seed = scenario_config.random_seed if args.seed is None else args.seed
            if seed < 0:
                parser.error("--seed must be non-negative")
            trace = generate_scenario_trace(scenario_config, dataset, seed=seed)
            generated_trace = True
        run_slug = f"{scenario_config.scenario_id}-{args.scheduler}-seed-{seed}"
        simulation_config = base_simulation_config.model_copy(
            update={"run_id": run_slug, "random_seed": seed}
        )
        runtime_conditions = ScenarioRuntimeConditions(scenario_config, devices, network)
    else:
        trace_path = args.trace or (
            PROJECT_ROOT / "artifacts" / "workload_traces" / "phase3-representative.json"
        )
        trace = load_workload_trace(trace_path, dataset=dataset)
        seed = base_simulation_config.random_seed if args.seed is None else args.seed
        simulation_config = base_simulation_config.model_copy(update={"random_seed": seed})
        run_slug = f"phase5-{args.scheduler}"

    engine = SimulationEngine(
        devices=devices,
        network=network,
        model_profiles=list(profiles.values()),
        config=simulation_config,
        prediction_provider=prediction_service,
        runtime_conditions=runtime_conditions,
    )
    assignment_provider = (
        FixedAssignmentProvider(load_assignment_plan(args.assignments))
        if args.scheduler == "fixed"
        else create_scheduler(
            args.scheduler,
            edgeweaver_alpha=simulation_config.edgeweaver_ewma_alpha,
        )
    )
    result = engine.run(trace, assignment_provider)
    output_phase = "phase5" if args.scheduler == "edgeweaver" else "phase4"
    legacy_name = f"{output_phase}-{args.scheduler}-run"
    output = args.output or PROJECT_ROOT / "experiments" / "raw" / (
        f"{run_slug}.json" if scenario_requested else f"{legacy_name}.json"
    )
    write_domain_json(output, result)

    if scenario_config is not None:
        trace_output = (
            args.trace_output
            or args.trace
            or (PROJECT_ROOT / "artifacts" / "workload_traces" / f"{trace.trace_id}.json")
        )
        events_output = args.events_output or (
            PROJECT_ROOT / "experiments" / "events" / f"{run_slug}-events.json"
        )
        summary_output = args.summary_output or (
            PROJECT_ROOT / "experiments" / "summaries" / f"{run_slug}.json"
        )
        summary_csv_output = args.summary_csv_output or summary_output.with_suffix(".csv")
        if generated_trace or args.trace_output is not None:
            save_workload_trace(trace_output, trace)
        save_event_log(
            events_output,
            event_log_from_run(result, scenario_id=scenario_config.scenario_id),
        )
        summary = calculate_run_metrics(
            result,
            trace,
            scenario_id=scenario_config.scenario_id,
            simulation_duration_ms=scenario_config.simulation_duration_ms,
            devices=devices.devices,
            model_profiles=list(profiles.values()),
        )
        save_run_metrics_json(summary_output, summary)
        save_run_metrics_csv(summary_csv_output, summary)
        print(
            f"{result.run_id}: {result.completed_requests}/{result.request_count} completed, "
            f"deadline satisfaction={summary.deadline_satisfaction_rate:.3f}, "
            f"goodput={summary.useful_goodput_requests_per_second:.3f} requests/s"
        )
        trace_verb = "Saved" if generated_trace or args.trace_output is not None else "Reused"
        print(f"{trace_verb} trace at {trace_output}")
        print(f"Saved raw run to {output}")
        print(f"Saved event log to {events_output}")
        print(f"Saved summaries to {summary_output} and {summary_csv_output}")
        return

    print(
        f"{result.run_id}: {result.completed_requests}/{result.request_count} completed, "
        f"{result.rejected_requests} rejected, {len(result.events)} events"
    )
    for request in result.requests:
        print(
            f"{request.request_id}: {request.status}, {request.assignment.device_id}/"
            f"{request.assignment.model_id}, latency={request.end_to_end_latency_ms}, "
            f"deadline_met={request.deadline_met}, prediction={request.actual_prediction}"
        )
    print(f"Saved structured run to {output}")


if __name__ == "__main__":
    main()
