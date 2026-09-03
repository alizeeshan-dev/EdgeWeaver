from pathlib import Path

import pytest
from pydantic import ValidationError

from edgeweaver.config import load_research_scenario
from edgeweaver.ml.data import load_uci_har
from edgeweaver.scenarios import ResearchScenarioConfig, generate_scenario_trace
from edgeweaver.workloads import load_workload_trace, save_workload_trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_ROOT = PROJECT_ROOT / "configs" / "scenarios"


def test_all_four_scenario_configs_load_with_expected_conditions() -> None:
    scenarios = {
        name: load_research_scenario(SCENARIO_ROOT / f"{name}.yaml")
        for name in ("normal", "bursty", "network_slowdown", "device_slowdown")
    }

    assert scenarios["normal"].arrival.bursts == ()
    assert len(scenarios["bursty"].arrival.bursts) == 3
    network = scenarios["network_slowdown"].network_slowdown
    assert network is not None
    assert network.link_id == "mobile-to-edge-server"
    assert network.start_time_ms < network.end_time_ms
    assert network.base_latency_multiplier > 1.0
    assert network.bandwidth_multiplier < 1.0
    device = scenarios["device_slowdown"].device_slowdown
    assert device is not None
    assert device.device_id == "edge-server"
    assert device.start_time_ms < device.end_time_ms
    assert device.service_time_multiplier > 1.0


def test_scenario_trace_is_seeded_dataset_backed_and_replayable(
    mini_uci_har_root: Path,
    tmp_path: Path,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)
    scenario = load_research_scenario(SCENARIO_ROOT / "normal.yaml")

    first = generate_scenario_trace(scenario, dataset, seed=127)
    second = generate_scenario_trace(scenario, dataset, seed=127)
    changed_seed = generate_scenario_trace(scenario, dataset, seed=128)

    assert first == second
    assert first.requests != changed_seed.requests
    assert first.generation is not None
    assert first.generation.scenario_id == "normal"
    assert first.generation.request_count == len(first.requests)
    assert first.generation.simulation_duration_ms == scenario.simulation_duration_ms
    assert all(
        request.arrival_time_ms < scenario.simulation_duration_ms for request in first.requests
    )
    assert all(request.input_size_bytes == dataset.feature_count * 8 for request in first.requests)
    assert all(
        request.true_label == int(dataset.test.labels[request.feature_vector_id])
        for request in first.requests
    )

    saved_path = tmp_path / "scheduler-independent-trace.json"
    save_workload_trace(saved_path, first)
    assert load_workload_trace(saved_path, dataset=dataset) == first


def test_bursty_generation_uses_declarative_high_rate_windows(
    mini_uci_har_root: Path,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)
    scenario = load_research_scenario(SCENARIO_ROOT / "bursty.yaml")
    trace = generate_scenario_trace(scenario, dataset)

    burst_duration_ms = sum(
        burst.end_time_ms - burst.start_time_ms for burst in scenario.arrival.bursts
    )
    burst_requests = sum(
        any(
            burst.start_time_ms <= request.arrival_time_ms < burst.end_time_ms
            for burst in scenario.arrival.bursts
        )
        for request in trace.requests
    )
    background_requests = len(trace.requests) - burst_requests
    background_duration_ms = scenario.simulation_duration_ms - burst_duration_ms

    assert burst_requests > 0
    assert background_requests > 0
    assert burst_requests / burst_duration_ms > background_requests / background_duration_ms


def test_scenario_validation_rejects_mismatched_or_leaking_conditions() -> None:
    payload = load_research_scenario(SCENARIO_ROOT / "normal.yaml").model_dump(mode="json")
    payload["network_slowdown"] = {
        "link_id": "mobile-to-edge-server",
        "start_time_ms": 5000.0,
        "end_time_ms": 10000.0,
        "base_latency_multiplier": 2.0,
        "bandwidth_multiplier": 0.5,
    }
    with pytest.raises(ValidationError, match="normal and bursty scenarios cannot define"):
        ResearchScenarioConfig.model_validate(payload)

    bursty_payload = load_research_scenario(SCENARIO_ROOT / "bursty.yaml").model_dump(mode="json")
    bursty_payload["arrival"]["bursts"][1]["start_time_ms"] = 4000.0
    with pytest.raises(ValidationError, match="must not overlap"):
        ResearchScenarioConfig.model_validate(bursty_payload)
