import pytest
from pydantic import ValidationError

from edgeweaver.config import DeviceCatalog, NetworkCatalog
from edgeweaver.domain import (
    EventType,
    ExecutionObservation,
    InferenceRequest,
    SimulationConfig,
    SimulationState,
)
from edgeweaver.engine import SimulationEngine
from edgeweaver.schedulers import (
    EdgeWeaverScheduler,
    create_edgeweaver_variant,
    create_scheduler,
)
from edgeweaver.schedulers.candidates import NoValidAssignmentError
from edgeweaver.schedulers.edgeweaver import ewma_latency_ms
from edgeweaver.workloads import WorkloadTrace
from tests.test_schedulers import _config, _devices, _links, _Predictor, _profiles, _request, _state


def _zero_network_state() -> SimulationState:
    links = tuple(link.model_copy(update={"base_latency_ms": 0.0}) for link in _links())
    config = _config().model_copy(
        update={
            "scheduler_overhead_ms": 0.0,
            "network_energy_per_kilobyte": 0.0,
            "return_payload_size_bytes": 0,
        }
    )
    return _state().model_copy(update={"network_links": links, "simulation_config": config})


def _replace_devices(state: SimulationState, **updates: dict[str, float]) -> SimulationState:
    devices = tuple(
        device.model_copy(update=updates.get(device.id, {})) for device in state.device_profiles
    )
    return state.model_copy(update={"device_profiles": devices})


def _candidate(decision, device_id: str, model_id: str):
    return next(
        candidate
        for candidate in decision.candidates
        if candidate.device_id == device_id and candidate.model_id == model_id
    )


def test_lowest_energy_feasible_wins_even_when_another_candidate_is_faster() -> None:
    state = _replace_devices(
        _zero_network_state(),
        mobile={"active_power_units": 0.1},
    )
    decision = EdgeWeaverScheduler().select_assignment(
        _request(deadline_ms=100.0, input_size_bytes=0),
        state,
    )
    selected = _candidate(decision, decision.device_id, decision.model_id)
    edge_light = _candidate(decision, "edge-server", "light-v1")
    assert (decision.device_id, decision.model_id) == ("mobile", "light-v1")
    assert selected.predicted_energy_units < edge_light.predicted_energy_units
    assert selected.predicted_completion_ms > edge_light.predicted_completion_ms


def test_energy_tie_prefers_earliest_completion() -> None:
    decision = EdgeWeaverScheduler().select_assignment(
        _request(deadline_ms=100.0, input_size_bytes=0),
        _zero_network_state(),
    )
    mobile = _candidate(decision, "mobile", "light-v1")
    edge = _candidate(decision, "edge-server", "light-v1")
    assert mobile.predicted_energy_units == edge.predicted_energy_units
    assert edge.predicted_completion_ms < mobile.predicted_completion_ms
    assert (decision.device_id, decision.model_id) == ("edge-server", "light-v1")


def test_full_tie_uses_stable_ids() -> None:
    state = _replace_devices(
        _zero_network_state(),
        mobile={"speed_multiplier": 1.0, "active_power_units": 1.0},
        gateway={"speed_multiplier": 1.0, "active_power_units": 1.0},
        **{"edge-server": {"speed_multiplier": 100.0}},
    )
    decision = EdgeWeaverScheduler().select_assignment(
        _request(deadline_ms=100.0, input_size_bytes=0),
        state,
    )
    mobile = _candidate(decision, "mobile", "light-v1")
    gateway = _candidate(decision, "gateway", "light-v1")
    assert mobile.predicted_energy_units == gateway.predicted_energy_units
    assert mobile.predicted_completion_ms == gateway.predicted_completion_ms
    assert (decision.device_id, decision.model_id) == ("gateway", "light-v1")


def test_tight_deadline_selects_only_feasible_faster_model() -> None:
    state = _replace_devices(
        _zero_network_state(),
        mobile={"speed_multiplier": 1.0},
        gateway={"speed_multiplier": 1.0},
        **{"edge-server": {"speed_multiplier": 1.0}},
    )
    decision = EdgeWeaverScheduler().select_assignment(
        _request(deadline_ms=6.0, input_size_bytes=0),
        state,
    )
    assert decision.model_id == "light-v1"
    assert decision.expected_to_meet_deadline
    assert all(
        not candidate.expected_to_meet_deadline
        for candidate in decision.candidates
        if candidate.eligible and candidate.model_id != "light-v1"
    )


def test_accuracy_excludes_lower_energy_light_model() -> None:
    decision = EdgeWeaverScheduler().select_assignment(
        _request(deadline_ms=200.0, minimum_accuracy=0.85, input_size_bytes=0),
        _zero_network_state(),
    )
    assert decision.model_id != "light-v1"
    assert all(
        candidate.rejection_reason == "model_accuracy_below_request_minimum"
        for candidate in decision.candidates
        if candidate.model_id == "light-v1" and candidate.eligible is False
    )
    mobile_heavy = _candidate(decision, "mobile", "heavy-v1")
    assert not mobile_heavy.compatible
    assert mobile_heavy.accuracy_eligible
    assert mobile_heavy.rejection_reason == "incompatible_device_model"


def test_no_feasible_candidate_falls_back_to_earliest_completion_not_energy() -> None:
    state = _replace_devices(
        _zero_network_state(),
        mobile={"active_power_units": 0.01},
        **{"edge-server": {"active_power_units": 100.0}},
    )
    decision = EdgeWeaverScheduler().select_assignment(
        _request(deadline_ms=1.0, input_size_bytes=0),
        state,
    )
    selected = _candidate(decision, decision.device_id, decision.model_id)
    mobile = _candidate(decision, "mobile", "light-v1")
    assert (decision.device_id, decision.model_id) == ("edge-server", "light-v1")
    assert selected.predicted_energy_units > mobile.predicted_energy_units
    assert not decision.expected_to_meet_deadline
    assert decision.decision_reason.startswith("no candidate predicted")


def test_no_valid_candidate_is_distinct_from_deadline_fallback() -> None:
    with pytest.raises(NoValidAssignmentError, match="no valid compatible assignment"):
        EdgeWeaverScheduler().select_assignment(
            _request(minimum_accuracy=0.99),
            _state(),
        )


def test_ewma_formula_multiple_observations_and_configurable_alpha() -> None:
    assert ewma_latency_ms(10.0, 30.0, 0.2) == 14.0
    first = ewma_latency_ms(6.0, 10.0, 0.5)
    second = ewma_latency_ms(first, 14.0, 0.5)
    assert (first, second) == (8.0, 11.0)
    assert ewma_latency_ms(10.0, 30.0, 0.5) == 20.0
    with pytest.raises(ValueError, match="alpha"):
        ewma_latency_ms(10.0, 30.0, 0.0)


def test_pair_isolation_and_causal_use_of_updated_estimate() -> None:
    scheduler = EdgeWeaverScheduler(alpha=0.5)
    state = _zero_network_state()
    profiles_before = [profile.model_dump() for profile in state.model_profiles]
    first = scheduler.select_assignment(
        _request("first", deadline_ms=100.0, input_size_bytes=0),
        state,
    )
    before = dict(scheduler.latency_estimates_ms)
    chosen_key = (first.device_id, first.model_id)
    update = scheduler.observe_execution(
        ExecutionObservation(
            request_id="first",
            timestamp_ms=110.0,
            device_id=first.device_id,
            model_id=first.model_id,
            inference_time_ms=30.0,
        )
    )
    after = dict(scheduler.latency_estimates_ms)
    assert _candidate(first, *chosen_key).predicted_inference_time_ms == before[chosen_key]
    assert update.new_estimate_ms == 0.5 * 30.0 + 0.5 * before[chosen_key]
    assert after[chosen_key] == update.new_estimate_ms
    assert all(after[key] == value for key, value in before.items() if key != chosen_key)
    assert [profile.model_dump() for profile in state.model_profiles] == profiles_before

    second = scheduler.select_assignment(
        _request("second", deadline_ms=100.0, input_size_bytes=0),
        state,
    )
    assert _candidate(second, *chosen_key).predicted_inference_time_ms == update.new_estimate_ms


def test_repeated_scheduler_observations_follow_ewma_sequence() -> None:
    scheduler = EdgeWeaverScheduler(alpha=0.5)
    decision = scheduler.select_assignment(
        _request(deadline_ms=100.0, input_size_bytes=0),
        _zero_network_state(),
    )
    key = (decision.device_id, decision.model_id)
    old = scheduler.latency_estimates_ms[key]
    first = scheduler.observe_execution(
        ExecutionObservation(
            request_id="first",
            timestamp_ms=10.0,
            device_id=key[0],
            model_id=key[1],
            inference_time_ms=10.0,
        )
    )
    second = scheduler.observe_execution(
        ExecutionObservation(
            request_id="second",
            timestamp_ms=20.0,
            device_id=key[0],
            model_id=key[1],
            inference_time_ms=14.0,
        )
    )
    assert first.new_estimate_ms == 0.5 * 10.0 + 0.5 * old
    assert second.old_estimate_ms == first.new_estimate_ms
    assert second.new_estimate_ms == 0.5 * 14.0 + 0.5 * first.new_estimate_ms


def test_unchanged_observation_does_not_emit_a_no_op_profile_update() -> None:
    scheduler = EdgeWeaverScheduler(alpha=0.2)
    decision = scheduler.select_assignment(
        _request(deadline_ms=100.0, input_size_bytes=0),
        _zero_network_state(),
    )
    key = (decision.device_id, decision.model_id)
    estimate = scheduler.latency_estimates_ms[key]

    update = scheduler.observe_execution(
        ExecutionObservation(
            request_id="unchanged",
            timestamp_ms=10.0,
            device_id=key[0],
            model_id=key[1],
            inference_time_ms=estimate,
        )
    )

    assert update is None
    assert scheduler.latency_estimates_ms[key] == estimate


def test_engine_does_not_record_no_op_profile_update_event() -> None:
    request = _request(arrival_ms=0.0, deadline_ms=300.0, input_size_bytes=0)
    result = SimulationEngine(
        devices=DeviceCatalog(devices=list(_devices())),
        network=NetworkCatalog(links=list(_links())),
        model_profiles=list(_profiles()),
        config=_config(),
        prediction_provider=_Predictor(),
    ).run(
        WorkloadTrace(trace_id="unchanged-observation", seed=17, requests=[request]),
        EdgeWeaverScheduler(),
    )

    assert all(event.event_type is not EventType.PROFILE_UPDATED for event in result.events)


def test_no_model_switching_ablation_only_restricts_selected_model() -> None:
    request = _request(deadline_ms=100.0, input_size_bytes=0)
    state = _zero_network_state()
    full = create_edgeweaver_variant("edgeweaver").select_assignment(request, state)
    ablated_scheduler = create_edgeweaver_variant("edgeweaver_no_model_switching")
    ablated = ablated_scheduler.select_assignment(request, state)

    assert full.model_id == "light-v1"
    assert ablated.model_id == "heavy-v1"
    assert ablated.predicted_model_accuracy == max(
        candidate.model_accuracy for candidate in ablated.candidates if candidate.eligible
    )
    assert [candidate.model_dump() for candidate in ablated.candidates] == [
        candidate.model_dump() for candidate in full.candidates
    ]
    update = ablated_scheduler.observe_execution(
        ExecutionObservation(
            request_id=request.request_id,
            timestamp_ms=10.0,
            device_id=ablated.device_id,
            model_id=ablated.model_id,
            inference_time_ms=30.0,
        )
    )
    assert update is not None


def test_no_online_update_ablation_preserves_selection_and_never_adapts() -> None:
    request = _request(deadline_ms=100.0, input_size_bytes=0)
    state = _zero_network_state()
    full = create_edgeweaver_variant("edgeweaver").select_assignment(request, state)
    scheduler = create_edgeweaver_variant("edgeweaver_no_online_update")
    ablated = scheduler.select_assignment(request, state)
    before = dict(scheduler.latency_estimates_ms)

    update = scheduler.observe_execution(
        ExecutionObservation(
            request_id=request.request_id,
            timestamp_ms=10.0,
            device_id=ablated.device_id,
            model_id=ablated.model_id,
            inference_time_ms=30.0,
        )
    )

    assert (ablated.device_id, ablated.model_id) == (full.device_id, full.model_id)
    assert [candidate.model_dump() for candidate in ablated.candidates] == [
        candidate.model_dump() for candidate in full.candidates
    ]
    assert update is None
    assert dict(scheduler.latency_estimates_ms) == before
    scheduler.reset()
    replay = scheduler.select_assignment(request, state)
    assert replay.model_dump() == ablated.model_dump()


def test_combined_ablation_is_rejected() -> None:
    with pytest.raises(ValueError, match="isolated"):
        EdgeWeaverScheduler(
            model_switching_enabled=False,
            online_updates_enabled=False,
        )


class _ControlledDurationProvider:
    def __init__(self, durations_ms: dict[str, float]) -> None:
        self.durations_ms = durations_ms

    def inference_time_ms(
        self,
        request: InferenceRequest,
        device,
        profile,
        static_inference_time_ms: float,
    ) -> float:
        del device, profile
        return self.durations_ms.get(request.request_id, static_inference_time_ms)


def _adaptive_engine() -> SimulationEngine:
    devices = [
        device.model_copy(update={"active_power_units": 3.0})
        if device.id == "edge-server"
        else device
        for device in _devices()
    ]
    links = [
        link.model_copy(update={"base_latency_ms": 2.0})
        if link.destination_device_id == "edge-server"
        else link.model_copy(update={"base_latency_ms": 0.0})
        for link in _links()
    ]
    config = _config().model_copy(
        update={
            "scheduler_overhead_ms": 0.0,
            "network_energy_per_kilobyte": 0.0,
            "return_payload_size_bytes": 0,
            "edgeweaver_ewma_alpha": 0.2,
        }
    )
    return SimulationEngine(
        devices=DeviceCatalog(devices=devices),
        network=NetworkCatalog(links=links),
        model_profiles=list(_profiles()),
        config=config,
        prediction_provider=_Predictor(),
        inference_duration_provider=_ControlledDurationProvider({"slow-first": 50.0}),
    )


def _adaptive_trace() -> WorkloadTrace:
    return WorkloadTrace(
        trace_id="phase5-adaptation-fixture",
        seed=17,
        requests=[
            _request("slow-first", arrival_ms=0.0, deadline_ms=200.0, input_size_bytes=0),
            _request("after-update", arrival_ms=60.0, deadline_ms=200.0, input_size_bytes=0),
        ],
    )


def test_engine_observes_inference_only_emits_updates_and_adapts_later_choice() -> None:
    engine = _adaptive_engine()
    scheduler = EdgeWeaverScheduler(alpha=0.2)
    result = engine.run(_adaptive_trace(), scheduler)

    first, second = result.requests
    assert first.assignment.device_id == "edge-server"
    assert first.inference_time_ms == 50.0
    assert first.end_to_end_latency_ms == 54.0
    assert second.assignment.device_id == "gateway"
    assert (
        _candidate(second.assignment, "edge-server", "light-v1").predicted_inference_time_ms == 12.0
    )

    first_events = [event for event in result.events if event.request_id == "slow-first"]
    completed_index = next(
        index
        for index, event in enumerate(first_events)
        if event.event_type is EventType.INFERENCE_COMPLETED
    )
    update_event = first_events[completed_index + 1]
    assert update_event.event_type is EventType.PROFILE_UPDATED
    assert update_event.details == {
        "old_estimate_ms": 2.5,
        "observed_inference_time_ms": 50.0,
        "new_estimate_ms": 12.0,
        "alpha": 0.2,
    }


def test_edgeweaver_state_resets_and_replays_deterministically() -> None:
    engine = _adaptive_engine()
    scheduler = EdgeWeaverScheduler(alpha=0.2)
    first = engine.run(_adaptive_trace(), scheduler)
    second = engine.run(_adaptive_trace(), scheduler)
    assert first.model_dump() == second.model_dump()
    assert first.requests[0].assignment.device_id == "edge-server"


@pytest.mark.parametrize("name", ["round_robin", "fastest_device", "min_completion"])
def test_baselines_remain_non_adaptive(name: str) -> None:
    request = _request(arrival_ms=0.0, deadline_ms=300.0, input_size_bytes=0)
    result = SimulationEngine(
        devices=DeviceCatalog(devices=list(_devices())),
        network=NetworkCatalog(links=list(_links())),
        model_profiles=list(_profiles()),
        config=_config(),
        prediction_provider=_Predictor(),
    ).run(
        WorkloadTrace(trace_id=f"{name}-non-adaptive", seed=17, requests=[request]),
        create_scheduler(name),
    )
    assert all(event.event_type is not EventType.PROFILE_UPDATED for event in result.events)


def test_edgeweaver_config_and_registry_alpha_validation() -> None:
    scheduler = create_scheduler("edgeweaver", edgeweaver_alpha=0.4)
    assert isinstance(scheduler, EdgeWeaverScheduler)
    assert scheduler.alpha == 0.4
    with pytest.raises(ValidationError):
        SimulationConfig(run_id="bad", random_seed=1, edgeweaver_ewma_alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        create_scheduler("edgeweaver", edgeweaver_alpha=1.1)
