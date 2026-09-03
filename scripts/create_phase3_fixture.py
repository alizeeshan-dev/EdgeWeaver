"""Create the deterministic Phase 3 trace and explicit verification assignments."""

from pathlib import Path

from edgeweaver.config import (
    load_device_catalog,
    load_network_catalog,
    load_simulation_config,
)
from edgeweaver.domain import AssignmentDecision, ModelRole
from edgeweaver.energy import estimate_energy
from edgeweaver.ml.data import DATASET_DIRECTORY_NAME, load_uci_har
from edgeweaver.ml.profile import load_measured_profiles
from edgeweaver.network import network_timing
from edgeweaver.scheduler import AssignmentPlan, save_assignment_plan
from edgeweaver.simulation import simulated_inference_time_ms
from edgeweaver.workloads import build_representative_trace, save_workload_trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = load_uci_har(PROJECT_ROOT / "data" / "raw" / DATASET_DIRECTORY_NAME)
    devices = load_device_catalog(PROJECT_ROOT / "configs" / "devices.yaml")
    network = load_network_catalog(PROJECT_ROOT / "configs" / "network.yaml")
    config = load_simulation_config(PROJECT_ROOT / "configs" / "simulation.yaml")
    profiles = load_measured_profiles(PROJECT_ROOT / "artifacts" / "profiles")
    trace = build_representative_trace(dataset, seed=config.random_seed)

    fixed_pairs = (
        ("mobile", ModelRole.LIGHT),
        ("gateway", ModelRole.BALANCED),
        ("edge-server", ModelRole.HEAVY),
        ("edge-server", ModelRole.LIGHT),
        ("gateway", ModelRole.HEAVY),
        ("mobile", ModelRole.LIGHT),
    )
    devices_by_id = {device.id: device for device in devices.devices}
    profiles_by_role = {profile.computational_role: profile for profile in profiles.values()}
    decisions: list[AssignmentDecision] = []
    for request, (device_id, role) in zip(trace.requests, fixed_pairs, strict=True):
        device = devices_by_id[device_id]
        profile = profiles_by_role[role]
        transfers = network_timing(
            source_device_id=request.source_device_id,
            destination_device_id=device_id,
            input_size_bytes=request.input_size_bytes,
            return_size_bytes=config.return_payload_size_bytes,
            links=network.links,
            random_seed=config.random_seed,
            request_id=request.request_id,
        )
        inference_ms = simulated_inference_time_ms(profile, device)
        transferred_bytes = (
            0
            if request.source_device_id == device_id
            else request.input_size_bytes + config.return_payload_size_bytes
        )
        energy = estimate_energy(
            device=device,
            inference_time_ms=inference_ms,
            transferred_bytes=transferred_bytes,
            network_energy_per_kilobyte=config.network_energy_per_kilobyte,
        )
        predicted_completion = (
            request.arrival_time_ms
            + config.scheduler_overhead_ms
            + transfers.upload_time_ms
            + inference_ms
            + transfers.return_time_ms
        )
        decisions.append(
            AssignmentDecision(
                request_id=request.request_id,
                device_id=device_id,
                model_id=profile.model_id,
                predicted_completion_ms=predicted_completion,
                predicted_energy_units=energy.estimated_total_energy_units,
                predicted_model_accuracy=profile.accuracy,
                expected_to_meet_deadline=predicted_completion <= request.absolute_deadline_ms,
                decision_reason="Explicit Phase 3 engine-verification assignment; not a policy.",
            )
        )

    trace_path = PROJECT_ROOT / "artifacts" / "workload_traces" / "phase3-representative.json"
    assignments_path = PROJECT_ROOT / "configs" / "phase3_assignments.json"
    save_workload_trace(trace_path, trace)
    save_assignment_plan(
        assignments_path,
        AssignmentPlan(scheduler_name="phase3-fixed-verification", decisions=decisions),
    )
    print(f"Saved {len(trace.requests)} requests to {trace_path}")
    print(f"Saved explicit assignments to {assignments_path}")


if __name__ == "__main__":
    main()
