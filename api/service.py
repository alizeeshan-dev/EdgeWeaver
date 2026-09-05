"""Filesystem and simulation adapters used by the local presentation API."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from api.schemas import SimulationRequest
from edgeweaver.config import load_device_catalog, load_network_catalog, load_research_scenario
from edgeweaver.experiments import (
    ExperimentRunner,
    ExperimentRunSpec,
    load_experiment_config,
)
from edgeweaver.ml.data import OFFICIAL_ACTIVITIES
from edgeweaver.ml.profile import load_measured_profiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"
PHASE7_ROOT = PROJECT_ROOT / "experiments" / "phase7"
UI_RUN_ROOT = PROJECT_ROOT / "experiments" / "ui"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

JsonObject = dict[str, Any]

SCHEDULERS: tuple[JsonObject, ...] = (
    {
        "id": "round_robin",
        "name": "Round Robin",
        "description": "Rotates across compatible devices and ignores queue and network state.",
        "accent": "#8b5cff",
    },
    {
        "id": "fastest_device",
        "name": "Fastest Device",
        "description": (
            "Chooses the fastest isolated device and its most accurate compatible model."
        ),
        "accent": "#ff5470",
    },
    {
        "id": "min_completion",
        "name": "Minimum Completion Time",
        "description": (
            "Minimizes predicted completion using current queue, network, and inference time."
        ),
        "accent": "#00f5ff",
    },
    {
        "id": "edgeweaver",
        "name": "EdgeWeaver",
        "description": (
            "Minimizes estimated energy among deadline-feasible candidates and adapts latency "
            "estimates."
        ),
        "accent": "#b7ff4a",
    },
)


def _read_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")


def _run_root(run_id: str) -> Path:
    _validate_run_id(run_id)
    for root in (UI_RUN_ROOT, PHASE7_ROOT):
        if (root / "results" / "per_run" / f"{run_id}.json").is_file():
            return root
    raise FileNotFoundError(run_id)


def _activity_labels() -> dict[int, str]:
    """Return canonical display labels without making saved-run reads depend on raw data."""

    return {
        identifier: name.replace("_", " ").title()
        for identifier, name in OFFICIAL_ACTIVITIES.items()
    }


def project_payload() -> JsonObject:
    devices = load_device_catalog(PROJECT_ROOT / "configs" / "devices.yaml")
    network = load_network_catalog(PROJECT_ROOT / "configs" / "network.yaml")
    return {
        "name": "EdgeWeaver",
        "subtitle": "Deadline-, accuracy-, and energy-aware edge inference scheduling",
        "research_question": (
            "How should inference requests be assigned to simulated heterogeneous edge devices "
            "when deadlines, model accuracy, queues, network delay, and estimated energy conflict?"
        ),
        "scope": "Local deterministic research prototype; not production edge infrastructure.",
        "devices": [item.model_dump(mode="json") for item in devices.devices],
        "network_links": [item.model_dump(mode="json") for item in network.links],
        "pipeline": [
            "UCI HAR",
            "Training",
            "Measured profiles",
            "SimPy engine",
            "Schedulers",
            "Events and metrics",
        ],
        "study": {"core_runs": 80, "ablation_runs": 10, "paired_trace_groups": 20},
        "measurement_boundary": {
            "measured": ["test accuracy", "macro F1", "artifact size", "local latency"],
            "simulated": ["devices", "network", "queues", "slowdowns", "normalized energy"],
        },
    }


def model_payload() -> list[JsonObject]:
    profiles = load_measured_profiles(PROJECT_ROOT / "artifacts" / "profiles")
    order = {"light": 0, "balanced": 1, "heavy": 2}
    return sorted(
        (profile.model_dump(mode="json") for profile in profiles.values()),
        key=lambda profile: order[str(profile["computational_role"])],
    )


def scenario_payload() -> list[JsonObject]:
    config = load_experiment_config(EXPERIMENT_CONFIG_PATH)
    return [
        load_research_scenario(
            PROJECT_ROOT / config.paths.scenario_config_directory / f"{scenario_id}.yaml"
        ).model_dump(mode="json")
        for scenario_id in config.scenarios
    ]


def scheduler_payload() -> list[JsonObject]:
    return [dict(item) for item in SCHEDULERS]


def list_runs() -> list[JsonObject]:
    runs: dict[str, JsonObject] = {}
    for root in (UI_RUN_ROOT, PHASE7_ROOT):
        directory = root / "results" / "per_run"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            run_id = str(payload["run_id"])
            runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "scenario_id": payload["scenario_id"],
                    "scheduler_name": payload["scheduler_name"],
                    "seed": payload["seed"],
                    "completed_requests": payload["completed_requests"],
                    "deadline_satisfaction_rate": payload["deadline_satisfaction_rate"],
                    "source": "interactive" if root == UI_RUN_ROOT else "phase7",
                },
            )
    return sorted(runs.values(), key=lambda run: str(run["run_id"]))


def _trace_from_record(record: JsonObject) -> JsonObject:
    reference = record.get("trace_reference")
    if not isinstance(reference, str):
        raise ValueError("run record is missing trace_reference")
    path = Path(reference)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _read_json(path)


def run_payload(run_id: str) -> JsonObject:
    root = _run_root(run_id)
    summary = _read_json(root / "results" / "per_run" / f"{run_id}.json")
    run = _read_json(root / "raw" / "runs" / f"{run_id}.json")
    events = _read_json(root / "raw" / "events" / f"{run_id}.json")
    record = _read_json(root / "results" / "run_records" / f"{run_id}.json")
    trace = _trace_from_record(record)
    labels = _activity_labels()
    trace_requests = {
        str(request["request_id"]): request
        for request in trace.get("requests", [])
        if isinstance(request, dict)
    }
    profile_updates = {
        str(event["request_id"]): event
        for event in events.get("events", [])
        if isinstance(event, dict)
        and event.get("event_type") == "PROFILE_UPDATED"
        and event.get("request_id") is not None
    }
    requests: list[JsonObject] = []
    for execution in run.get("requests", []):
        if not isinstance(execution, dict):
            continue
        request_id = str(execution["request_id"])
        request = dict(trace_requests.get(request_id, {}))
        label = request.get("true_label")
        request["true_label_name"] = (
            labels.get(int(label), str(label)) if label is not None else "—"
        )
        actual_prediction = execution.get("actual_prediction")
        execution_copy = dict(execution)
        execution_copy["actual_prediction_name"] = (
            labels.get(int(actual_prediction), str(actual_prediction))
            if actual_prediction is not None
            else "—"
        )
        requests.append(
            {
                "request": request,
                "execution": execution_copy,
                "profile_update": profile_updates.get(request_id),
            }
        )
    return {
        "run": {
            "run_id": run["run_id"],
            "scheduler_name": run["scheduler_name"],
            "random_seed": run["random_seed"],
            "request_count": run["request_count"],
            "completed_requests": run["completed_requests"],
            "rejected_requests": run["rejected_requests"],
        },
        "summary": summary,
        "trace": {key: value for key, value in trace.items() if key != "requests"},
        "requests": requests,
    }


def event_payload(run_id: str) -> JsonObject:
    root = _run_root(run_id)
    return _read_json(root / "raw" / "events" / f"{run_id}.json")


def experiment_results_payload() -> JsonObject:
    table_path = PHASE7_ROOT / "results" / "tables" / "aggregate_results.csv"
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("run_count", "mean", "sample_std"):
            row[key] = int(row[key]) if key == "run_count" else float(row[key])
        row["seed_values"] = json.loads(str(row["seed_values"]))
    findings = (PHASE7_ROOT / "results" / "findings_summary.md").read_text(encoding="utf-8")
    return {
        "rows": rows,
        "findings_markdown": findings,
        "figures": sorted(path.name for path in (PHASE7_ROOT / "figures").glob("*.png")),
    }


def failure_cases_payload() -> JsonObject:
    return _read_json(PHASE7_ROOT / "results" / "failure_cases.json")


def execute_simulation(request: SimulationRequest) -> JsonObject:
    config = load_experiment_config(EXPERIMENT_CONFIG_PATH)
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"output_root": Path("experiments/ui")})}
    )
    runner = ExperimentRunner(config, project_root=PROJECT_ROOT)
    run_id = f"core-{request.scenario_id}-{request.scheduler_name}-seed-{request.seed}"
    spec = ExperimentRunSpec(
        kind="core",
        scenario_id=request.scenario_id,
        scheduler_name=request.scheduler_name,
        seed=request.seed,
        run_id=run_id,
    )
    report = runner.run((spec,), force=request.force)
    payload = run_payload(spec.run_id)
    return {
        "run_id": spec.run_id,
        "executed": report.executed_runs == 1,
        "summary": payload["summary"],
    }
