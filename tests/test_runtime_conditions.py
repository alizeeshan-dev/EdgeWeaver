from pathlib import Path

import pytest

from edgeweaver.config import DeviceCatalog, NetworkCatalog, load_research_scenario
from edgeweaver.domain import EventType
from edgeweaver.engine import SimulationEngine
from edgeweaver.runtime_conditions import ScenarioRuntimeConditions
from edgeweaver.scheduler import AssignmentPlan, FixedAssignmentProvider
from edgeweaver.schedulers import EdgeWeaverScheduler, create_scheduler
from edgeweaver.workloads import WorkloadTrace
from tests.test_engine import (
    CountingPredictionProvider,
    _config,
    _decision,
    _devices,
    _network,
    _profiles,
    _request,
)
from tests.test_schedulers import (
    _config as scheduler_config,
)
from tests.test_schedulers import (
    _devices as scheduler_devices,
)
from tests.test_schedulers import (
    _links as scheduler_links,
)
from tests.test_schedulers import (
    _Predictor,
)
from tests.test_schedulers import (
    _profiles as scheduler_profiles,
)
from tests.test_schedulers import (
    _request as scheduler_request,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _scenario(name: str):
    return load_research_scenario(PROJECT_ROOT / "configs" / "scenarios" / f"{name}.yaml")


def test_network_condition_is_half_open_and_reports_reconstructable_events() -> None:
    scenario = _scenario("network_slowdown")
    devices = _devices()
    network = _network()
    conditions = ScenarioRuntimeConditions(scenario, devices, network)
    slowdown = scenario.network_slowdown
    assert slowdown is not None

    before = conditions.network_links_at(slowdown.start_time_ms - 0.001, network.links)[1]
    during = conditions.network_links_at(slowdown.start_time_ms, network.links)[1]
    after = conditions.network_links_at(slowdown.end_time_ms, network.links)[1]
    assert before == after == network.links[1]
    assert during.base_latency_ms == pytest.approx(
        network.links[1].base_latency_ms * slowdown.base_latency_multiplier
    )
    assert during.bandwidth_mbps == pytest.approx(
        network.links[1].bandwidth_mbps * slowdown.bandwidth_multiplier
    )
    events = conditions.transition_events()
    assert [event.event_type for event in events] == [
        EventType.NETWORK_SLOWDOWN_STARTED,
        EventType.NETWORK_SLOWDOWN_ENDED,
    ]
    assert [event.timestamp_ms for event in events] == [
        slowdown.start_time_ms,
        slowdown.end_time_ms,
    ]
    assert events[0].details is not None
    assert events[0].details["original_base_latency_ms"] == network.links[1].base_latency_ms
    assert events[0].details["temporary_bandwidth_mbps"] == during.bandwidth_mbps


def test_device_condition_is_half_open_and_does_not_mutate_device_config() -> None:
    scenario = _scenario("device_slowdown")
    devices = _devices()
    original = devices.model_dump()
    conditions = ScenarioRuntimeConditions(scenario, devices, _network())
    slowdown = scenario.device_slowdown
    assert slowdown is not None

    assert conditions.service_time_multiplier(slowdown.start_time_ms - 0.001, "edge-server") == 1
    assert (
        conditions.service_time_multiplier(slowdown.start_time_ms, "edge-server")
        == slowdown.service_time_multiplier
    )
    assert conditions.service_time_multiplier(slowdown.end_time_ms, "edge-server") == 1
    assert conditions.service_time_multiplier(slowdown.start_time_ms, "gateway") == 1
    assert devices.model_dump() == original


def _runtime_engine(scenario_name: str) -> tuple[SimulationEngine, object]:
    devices = _devices()
    network = _network()
    scenario = _scenario(scenario_name)
    if scenario_name == "network_slowdown":
        slowdown = scenario.network_slowdown
        assert slowdown is not None
        scenario = scenario.model_copy(
            update={
                "simulation_duration_ms": 400.0,
                "network_slowdown": slowdown.model_copy(
                    update={"start_time_ms": 50.0, "end_time_ms": 300.0}
                ),
            }
        )
    else:
        slowdown = scenario.device_slowdown
        assert slowdown is not None
        scenario = scenario.model_copy(
            update={
                "simulation_duration_ms": 400.0,
                "device_slowdown": slowdown.model_copy(
                    update={"start_time_ms": 50.0, "end_time_ms": 300.0}
                ),
            }
        )
    return (
        SimulationEngine(
            devices=devices,
            network=network,
            model_profiles=_profiles(),
            config=_config(),
            prediction_provider=CountingPredictionProvider(),
            runtime_conditions=ScenarioRuntimeConditions(scenario, devices, network),
        ),
        scenario,
    )


def test_network_slowdown_affects_only_transfer_legs_started_inside_window() -> None:
    engine, _ = _runtime_engine("network_slowdown")
    requests = [
        _request("before", arrival=0.0),
        _request("during", arrival=100.0),
        _request("after", arrival=310.0),
    ]
    plan = AssignmentPlan(
        scheduler_name="fixed-test",
        decisions=[
            _decision(request, "edge-server", "logistic-regression-v1") for request in requests
        ],
    )
    result = engine.run(
        WorkloadTrace(trace_id="network-window", seed=7, requests=requests),
        FixedAssignmentProvider(plan),
    )

    before, during, after = result.requests
    assert during.upload_time_ms > before.upload_time_ms
    assert during.return_time_ms > before.return_time_ms
    assert after.upload_time_ms == pytest.approx(before.upload_time_ms)
    assert after.return_time_ms == pytest.approx(before.return_time_ms)
    transition_events = [
        event
        for event in result.events
        if event.event_type
        in {EventType.NETWORK_SLOWDOWN_STARTED, EventType.NETWORK_SLOWDOWN_ENDED}
    ]
    assert [(event.event_type, event.timestamp_ms) for event in transition_events] == [
        (EventType.NETWORK_SLOWDOWN_STARTED, 50.0),
        (EventType.NETWORK_SLOWDOWN_ENDED, 300.0),
    ]


def test_scheduler_predictions_see_only_network_condition_at_decision_time() -> None:
    engine, _ = _runtime_engine("network_slowdown")
    requests = [
        _request("before", arrival=0.0, deadline=500.0),
        _request("during", arrival=100.0, deadline=500.0),
        _request("after", arrival=310.0, deadline=500.0),
    ]
    result = engine.run(
        WorkloadTrace(trace_id="network-visibility", seed=7, requests=requests),
        create_scheduler("min_completion"),
    )

    edge_uploads = []
    for request in result.requests:
        edge_candidate = next(
            candidate
            for candidate in request.assignment.candidates
            if candidate.device_id == "edge-server"
            and candidate.model_id == "logistic-regression-v1"
        )
        edge_uploads.append(edge_candidate.predicted_upload_time_ms)
    assert edge_uploads[1] > edge_uploads[0]
    assert edge_uploads[2] == pytest.approx(edge_uploads[0])


def test_slowdown_transition_is_recorded_before_same_timestamp_request_arrival() -> None:
    engine, _ = _runtime_engine("network_slowdown")
    request = _request("at-boundary", arrival=50.0, deadline=500.0)
    result = engine.run(
        WorkloadTrace(trace_id="boundary-order", seed=7, requests=[request]),
        FixedAssignmentProvider(
            AssignmentPlan(
                scheduler_name="fixed-test",
                decisions=[_decision(request, "edge-server", "logistic-regression-v1")],
            )
        ),
    )
    at_boundary = [event.event_type for event in result.events if event.timestamp_ms == 50.0]
    assert at_boundary[:2] == [EventType.NETWORK_SLOWDOWN_STARTED, EventType.REQUEST_ARRIVED]


def test_device_slowdown_changes_actual_service_only_within_window() -> None:
    engine, scenario = _runtime_engine("device_slowdown")
    requests = [
        _request("before", arrival=0.0),
        _request("during", arrival=100.0),
        _request("after", arrival=310.0),
    ]
    plan = AssignmentPlan(
        scheduler_name="fixed-test",
        decisions=[
            _decision(request, "edge-server", "logistic-regression-v1") for request in requests
        ],
    )
    result = engine.run(
        WorkloadTrace(trace_id="device-window", seed=7, requests=requests),
        FixedAssignmentProvider(plan),
    )
    before, during, after = result.requests
    slowdown = scenario.device_slowdown
    assert slowdown is not None
    assert during.inference_time_ms == pytest.approx(
        before.inference_time_ms * slowdown.service_time_multiplier
    )
    assert after.inference_time_ms == pytest.approx(before.inference_time_ms)
    assert all(event.event_type is not EventType.PROFILE_UPDATED for event in result.events)


def test_runtime_conditions_and_engine_state_reset_between_runs() -> None:
    engine, _ = _runtime_engine("device_slowdown")
    request = _request("repeat", arrival=100.0)
    plan = AssignmentPlan(
        scheduler_name="fixed-test",
        decisions=[_decision(request, "edge-server", "logistic-regression-v1")],
    )
    trace = WorkloadTrace(trace_id="repeat-runtime", seed=7, requests=[request])

    first = engine.run(trace, FixedAssignmentProvider(plan))
    second = engine.run(trace, FixedAssignmentProvider(plan))
    assert first.model_dump() == second.model_dump()


def test_edgeweaver_learns_device_slowdown_only_after_completed_observation() -> None:
    devices = DeviceCatalog(devices=list(scheduler_devices()))
    network = NetworkCatalog(
        links=[link.model_copy(update={"base_latency_ms": 0.0}) for link in scheduler_links()]
    )
    scenario = _scenario("device_slowdown")
    slowdown = scenario.device_slowdown
    assert slowdown is not None
    scenario = scenario.model_copy(
        update={
            "simulation_duration_ms": 100.0,
            "device_slowdown": slowdown.model_copy(
                update={"start_time_ms": 0.0, "end_time_ms": 50.0}
            ),
        }
    )
    config = scheduler_config().model_copy(
        update={
            "scheduler_overhead_ms": 0.0,
            "network_energy_per_kilobyte": 0.0,
            "return_payload_size_bytes": 0,
        }
    )
    profiles = list(scheduler_profiles())
    profile_snapshot = [profile.model_dump() for profile in profiles]
    requests = [
        scheduler_request("first", arrival_ms=0.0, deadline_ms=100.0, input_size_bytes=0),
        scheduler_request("second", arrival_ms=10.0, deadline_ms=100.0, input_size_bytes=0),
    ]
    scheduler = EdgeWeaverScheduler(alpha=0.2)
    result = SimulationEngine(
        devices=devices,
        network=network,
        model_profiles=profiles,
        config=config,
        prediction_provider=_Predictor(),
        runtime_conditions=ScenarioRuntimeConditions(scenario, devices, network),
    ).run(WorkloadTrace(trace_id="causal-slowdown", seed=17, requests=requests), scheduler)

    first, second = result.requests
    assert first.assignment.device_id == "edge-server"
    assert first.inference_time_ms == pytest.approx(7.5)
    first_edge_candidate = next(
        candidate
        for candidate in first.assignment.candidates
        if candidate.device_id == "edge-server" and candidate.model_id == "light-v1"
    )
    assert first_edge_candidate.predicted_inference_time_ms == pytest.approx(2.5)
    update = next(
        event
        for event in result.events
        if event.request_id == "first" and event.event_type is EventType.PROFILE_UPDATED
    )
    assert update.timestamp_ms == pytest.approx(7.5)
    assert update.details["new_estimate_ms"] == pytest.approx(3.5)
    second_edge_candidate = next(
        candidate
        for candidate in second.assignment.candidates
        if candidate.device_id == "edge-server" and candidate.model_id == "light-v1"
    )
    assert second_edge_candidate.predicted_inference_time_ms == pytest.approx(3.5)
    assert [profile.model_dump() for profile in profiles] == profile_snapshot


@pytest.mark.parametrize(
    "scheduler_name", ["round_robin", "fastest_device", "min_completion", "edgeweaver"]
)
def test_same_saved_trace_is_consumed_unchanged_by_every_scheduler(scheduler_name: str) -> None:
    trace = WorkloadTrace(
        trace_id="paired-trace",
        seed=17,
        requests=[
            scheduler_request("first", arrival_ms=0.0, deadline_ms=300.0),
            scheduler_request("second", arrival_ms=20.0, deadline_ms=300.0),
        ],
    )
    snapshot = trace.model_dump()
    result = SimulationEngine(
        devices=DeviceCatalog(devices=list(scheduler_devices())),
        network=NetworkCatalog(links=list(scheduler_links())),
        model_profiles=list(scheduler_profiles()),
        config=scheduler_config(),
        prediction_provider=_Predictor(),
    ).run(trace, create_scheduler(scheduler_name))

    assert [request.request_id for request in result.requests] == ["first", "second"]
    assert trace.model_dump() == snapshot
