import pytest

from edgeweaver.config import DeviceCatalog, NetworkCatalog
from edgeweaver.domain import (
    Device,
    DeviceState,
    InferenceRequest,
    ModelProfile,
    ModelRole,
    NetworkLink,
    SimulationConfig,
    SimulationState,
)
from edgeweaver.engine import SimulationEngine
from edgeweaver.schedulers import (
    FastestDeviceScheduler,
    MinimumCompletionTimeScheduler,
    RoundRobinScheduler,
    create_scheduler,
)
from edgeweaver.schedulers.candidates import (
    NoValidAssignmentError,
    estimate_candidates,
    estimate_queue_wait_ms,
)
from edgeweaver.workloads import WorkloadTrace


class _Predictor:
    def predict(self, model_id: str, feature_vector_id: int) -> int:
        del model_id
        return feature_vector_id % 6 + 1


def _profile(
    model_id: str,
    role: ModelRole,
    latency_ms: float,
    accuracy: float,
) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        display_name=model_id,
        accuracy=accuracy,
        macro_f1=accuracy,
        model_size_bytes=100,
        local_latency_ms_mean=latency_ms,
        local_latency_ms_median=latency_ms,
        local_latency_ms_std=0.1,
        local_latency_ms_p95=latency_ms + 0.1,
        profile_kind="measured",
        computational_role=role,
        profiling_prediction_count=500,
        profiling_seed=7,
        raw_latency_observations_file=f"{model_id}.json",
    )


def _profiles() -> tuple[ModelProfile, ...]:
    return (
        _profile("light-v1", ModelRole.LIGHT, 5.0, 0.80),
        _profile("balanced-v1", ModelRole.BALANCED, 8.0, 0.90),
        _profile("heavy-v1", ModelRole.HEAVY, 12.0, 0.95),
    )


def _devices() -> tuple[Device, ...]:
    return (
        Device(
            id="mobile",
            speed_multiplier=2.0,
            active_power_units=1.0,
            queue_capacity=10,
            supported_models=["light", "balanced"],
        ),
        Device(
            id="gateway",
            speed_multiplier=1.0,
            active_power_units=2.0,
            queue_capacity=10,
            supported_models=["light", "balanced", "heavy"],
        ),
        Device(
            id="edge-server",
            speed_multiplier=0.5,
            active_power_units=4.0,
            queue_capacity=10,
            supported_models=["light", "balanced", "heavy"],
        ),
    )


def _links(edge_latency_ms: float = 50.0) -> tuple[NetworkLink, ...]:
    return (
        NetworkLink(
            link_id="mobile-gateway",
            source_device_id="mobile",
            destination_device_id="gateway",
            base_latency_ms=10.0,
            bandwidth_mbps=10.0,
        ),
        NetworkLink(
            link_id="mobile-edge",
            source_device_id="mobile",
            destination_device_id="edge-server",
            base_latency_ms=edge_latency_ms,
            bandwidth_mbps=10.0,
        ),
    )


def _config() -> SimulationConfig:
    return SimulationConfig(
        run_id="phase4-test",
        random_seed=17,
        scheduler_overhead_ms=1.0,
        network_energy_per_kilobyte=0.25,
        return_payload_size_bytes=64,
    )


def _request(
    request_id: str = "request-1",
    *,
    arrival_ms: float = 100.0,
    deadline_ms: float = 100.0,
    minimum_accuracy: float = 0.75,
    input_size_bytes: int = 1000,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        arrival_time_ms=arrival_ms,
        deadline_ms=deadline_ms,
        minimum_accuracy=minimum_accuracy,
        input_size_bytes=input_size_bytes,
        source_device_id="mobile",
        true_label=1,
        feature_vector_id=0,
    )


def _state(
    *,
    current_time_ms: float = 100.0,
    device_states: tuple[DeviceState, ...] | None = None,
    edge_latency_ms: float = 50.0,
) -> SimulationState:
    devices = _devices()
    states = device_states or tuple(
        DeviceState(
            device_id=device.id,
            queue_length=0,
            in_service=0,
            queue_capacity=device.queue_capacity,
            processing_capacity=device.processing_capacity,
        )
        for device in devices
    )
    return SimulationState(
        current_time_ms=current_time_ms,
        devices=states,
        model_profiles=_profiles(),
        network_links=_links(edge_latency_ms),
        device_profiles=devices,
        simulation_config=_config(),
    )


def test_queue_delay_uses_actual_active_and_queued_work() -> None:
    state = DeviceState(
        device_id="mobile",
        queue_length=2,
        in_service=1,
        queue_capacity=10,
        processing_capacity=1,
        active_remaining_ms=(7.0,),
        queued_inference_times_ms=(11.0, 3.0),
    )
    assert estimate_queue_wait_ms(state, arrival_lead_time_ms=2.0) == 19.0


def test_candidate_estimates_use_absolute_completion_and_energy() -> None:
    request = _request(deadline_ms=15.0, input_size_bytes=0)
    candidate = next(
        candidate
        for candidate in estimate_candidates(request, _state())
        if candidate.device_id == "mobile" and candidate.model_id == "light-v1"
    )
    assert candidate.predicted_completion_ms == 111.0
    assert candidate.expected_to_meet_deadline
    assert candidate.predicted_energy_units == 10.0


def test_round_robin_rotates_skips_invalid_device_and_uses_fastest_model() -> None:
    scheduler = RoundRobinScheduler()
    first = scheduler.select_assignment(_request("first"), _state())
    second = scheduler.select_assignment(_request("second"), _state())
    third = scheduler.select_assignment(_request("third"), _state())
    strict = scheduler.select_assignment(
        _request("strict", minimum_accuracy=0.93),
        _state(),
    )
    fifth = scheduler.select_assignment(_request("fifth"), _state())

    assert (first.device_id, first.model_id) == ("mobile", "light-v1")
    assert (second.device_id, second.model_id) == ("gateway", "light-v1")
    assert third.device_id == "edge-server"
    assert (strict.device_id, strict.model_id) == ("gateway", "heavy-v1")
    assert fifth.device_id == "edge-server"


def test_round_robin_device_selection_ignores_congestion() -> None:
    states = list(_state().devices)
    states[0] = states[0].model_copy(update={"in_service": 1, "active_remaining_ms": (1000.0,)})
    decision = RoundRobinScheduler().select_assignment(
        _request(),
        _state(device_states=tuple(states)),
    )
    assert decision.device_id == "mobile"


def test_fastest_device_ignores_congestion_and_uses_highest_accuracy_model() -> None:
    states = list(_state().devices)
    states[2] = states[2].model_copy(update={"in_service": 1, "active_remaining_ms": (100.0,)})
    decision = FastestDeviceScheduler().select_assignment(
        _request(),
        _state(device_states=tuple(states)),
    )
    assert (decision.device_id, decision.model_id) == ("edge-server", "heavy-v1")


def test_mct_accounts_for_congestion_and_remote_network() -> None:
    states = list(_state().devices)
    states[2] = states[2].model_copy(update={"in_service": 1, "active_remaining_ms": (100.0,)})
    decision = MinimumCompletionTimeScheduler().select_assignment(
        _request(),
        _state(device_states=tuple(states)),
    )
    assert (decision.device_id, decision.model_id) == ("mobile", "light-v1")

    poor_network = MinimumCompletionTimeScheduler().select_assignment(
        _request("poor-network"),
        _state(edge_latency_ms=100.0),
    )
    assert poor_network.device_id == "mobile"


def test_mct_avoids_a_queue_known_to_be_full_at_arrival() -> None:
    states = list(_state().devices)
    states[0] = states[0].model_copy(
        update={
            "queue_length": 10,
            "in_service": 1,
            "active_remaining_ms": (100.0,),
            "queued_inference_times_ms": (100.0,) * 10,
        }
    )
    decision = MinimumCompletionTimeScheduler().select_assignment(
        _request(),
        _state(device_states=tuple(states)),
    )
    mobile_candidates = [
        candidate for candidate in decision.candidates if candidate.device_id == "mobile"
    ]
    assert decision.device_id != "mobile"
    assert all(not candidate.queue_admissible for candidate in mobile_candidates)


@pytest.mark.parametrize(
    "scheduler",
    [RoundRobinScheduler(), FastestDeviceScheduler(), MinimumCompletionTimeScheduler()],
)
def test_accuracy_constraint_excludes_light_and_mobile_never_uses_heavy(scheduler) -> None:
    decision = scheduler.select_assignment(
        _request(minimum_accuracy=0.85),
        _state(),
    )
    assert decision.model_id != "light-v1"
    assert not (decision.device_id == "mobile" and decision.model_id == "heavy-v1")


def test_mct_selects_earliest_candidate_even_when_deadline_is_impossible() -> None:
    decision = MinimumCompletionTimeScheduler().select_assignment(
        _request(deadline_ms=1.0),
        _state(),
    )
    assert (decision.device_id, decision.model_id) == ("mobile", "light-v1")
    assert not decision.expected_to_meet_deadline


def test_no_eligible_candidate_is_an_explicit_failure() -> None:
    with pytest.raises(NoValidAssignmentError, match="no admissible compatible model"):
        MinimumCompletionTimeScheduler().select_assignment(
            _request(minimum_accuracy=0.99),
            _state(),
        )


def test_deterministic_ties_and_registry_return_fresh_round_robin_instances() -> None:
    first = FastestDeviceScheduler().select_assignment(_request(), _state())
    second = FastestDeviceScheduler().select_assignment(_request(), _state())
    assert first == second

    registry_first = create_scheduler("round_robin")
    registry_first.select_assignment(_request(), _state())
    registry_second = create_scheduler("round_robin")
    reset_decision = registry_second.select_assignment(_request(), _state())
    assert reset_decision.device_id == "mobile"


def test_fastest_device_tie_breaks_by_stable_device_id() -> None:
    devices = tuple(
        device.model_copy(update={"speed_multiplier": 1.0})
        if device.id == "edge-server"
        else device
        for device in _devices()
    )
    decision = FastestDeviceScheduler().select_assignment(
        _request(),
        _state().model_copy(update={"device_profiles": devices}),
    )
    tied = [
        candidate
        for candidate in decision.candidates
        if candidate.model_id == "heavy-v1" and candidate.device_id in {"gateway", "edge-server"}
    ]
    assert [candidate.predicted_inference_time_ms for candidate in tied] == [12.0, 12.0]
    assert (decision.device_id, decision.model_id) == ("edge-server", "heavy-v1")


def test_mct_tie_breaks_by_stable_device_and_model_ids() -> None:
    devices = tuple(
        device.model_copy(update={"speed_multiplier": 1.0 if device.id != "edge-server" else 100.0})
        for device in _devices()
    )
    zero_links = tuple(link.model_copy(update={"base_latency_ms": 0.0}) for link in _links())
    base = _state()
    state = base.model_copy(
        update={
            "device_profiles": devices,
            "network_links": zero_links,
            "simulation_config": _config().model_copy(
                update={"scheduler_overhead_ms": 0.0, "return_payload_size_bytes": 0}
            ),
        }
    )
    decision = MinimumCompletionTimeScheduler().select_assignment(
        _request(input_size_bytes=0),
        state,
    )
    assert (decision.device_id, decision.model_id) == ("gateway", "light-v1")


def _run_baseline(name: str) -> object:
    request = _request(arrival_ms=0.0, deadline_ms=200.0, input_size_bytes=0)
    engine = SimulationEngine(
        devices=DeviceCatalog(devices=list(_devices())),
        network=NetworkCatalog(links=list(_links())),
        model_profiles=list(_profiles()),
        config=_config().model_copy(update={"scheduler_overhead_ms": 0.0}),
        prediction_provider=_Predictor(),
    )
    return engine.run(
        WorkloadTrace(trace_id="shared-trace", seed=17, requests=[request]),
        create_scheduler(name),
    )


def test_inflight_remote_reservations_keep_predictions_consistent() -> None:
    requests = [
        _request("remote-1", arrival_ms=0.0, deadline_ms=300.0, input_size_bytes=0),
        _request("remote-2", arrival_ms=0.0, deadline_ms=300.0, input_size_bytes=0),
    ]
    engine = SimulationEngine(
        devices=DeviceCatalog(devices=list(_devices())),
        network=NetworkCatalog(links=list(_links())),
        model_profiles=list(_profiles()),
        config=_config(),
        prediction_provider=_Predictor(),
    )
    result = engine.run(
        WorkloadTrace(trace_id="simultaneous-remote", seed=17, requests=requests),
        FastestDeviceScheduler(),
    )
    assert all(request.assignment.device_id == "edge-server" for request in result.requests)
    assert [request.assignment.predicted_completion_ms for request in result.requests] == (
        pytest.approx([request.completion_time_ms for request in result.requests])
    )


def test_engine_resets_round_robin_between_independent_runs() -> None:
    request = _request(arrival_ms=0.0, deadline_ms=200.0, input_size_bytes=0)
    trace = WorkloadTrace(trace_id="round-robin-reset", seed=17, requests=[request])
    engine = SimulationEngine(
        devices=DeviceCatalog(devices=list(_devices())),
        network=NetworkCatalog(links=list(_links())),
        model_profiles=list(_profiles()),
        config=_config(),
        prediction_provider=_Predictor(),
    )
    scheduler = RoundRobinScheduler()
    first = engine.run(trace, scheduler)
    second = engine.run(trace, scheduler)
    assert first.model_dump() == second.model_dump()
    assert first.requests[0].assignment.device_id == "mobile"


def test_prediction_matches_actual_execution_and_same_trace_replays() -> None:
    first = _run_baseline("min_completion")
    second = _run_baseline("min_completion")
    assert first.model_dump() == second.model_dump()
    request_result = first.requests[0]
    assert request_result.assignment.predicted_completion_ms == pytest.approx(
        request_result.completion_time_ms
    )

    results = [_run_baseline(name) for name in ("round_robin", "fastest_device", "min_completion")]
    assert all(result.request_count == 1 for result in results)
    assert [result.scheduler_name for result in results] == [
        "round_robin",
        "fastest_device",
        "min_completion",
    ]
