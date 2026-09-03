from datetime import UTC, datetime

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from edgeweaver.config import DeviceCatalog, NetworkCatalog
from edgeweaver.domain import (
    AssignmentDecision,
    Device,
    EventType,
    InferenceRequest,
    ModelProfile,
    ModelRole,
    NetworkLink,
    RequestStatus,
    SimulationConfig,
)
from edgeweaver.energy import compute_energy_units, estimate_energy, network_energy_units
from edgeweaver.engine import SimulationEngine, is_device_model_compatible
from edgeweaver.ml.artifacts import LoadedModelArtifacts, ModelArtifactMetadata
from edgeweaver.ml.data import load_uci_har
from edgeweaver.ml.inference import ModelPredictionService
from edgeweaver.network import network_timing, transfer_time_ms
from edgeweaver.scheduler import AssignmentPlan, FixedAssignmentProvider
from edgeweaver.simulation import simulated_inference_time_ms
from edgeweaver.workloads import WorkloadTrace


class CountingPredictionProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def predict(self, model_id: str, feature_vector_id: int) -> int:
        self.calls.append((model_id, feature_vector_id))
        return feature_vector_id % 6 + 1


def _profile(model_id: str, role: ModelRole, mean_ms: float) -> ModelProfile:
    return ModelProfile(
        model_id=model_id,
        display_name=f"{role.value} test model",
        accuracy=0.95,
        macro_f1=0.94,
        model_size_bytes=100,
        local_latency_ms_mean=mean_ms,
        local_latency_ms_median=mean_ms,
        local_latency_ms_std=0.01,
        local_latency_ms_p95=mean_ms + 0.1,
        profile_kind="measured",
        computational_role=role,
        profiling_prediction_count=500,
        profiling_seed=1,
        raw_latency_observations_file=f"{model_id}.timings.json",
    )


def _profiles() -> list[ModelProfile]:
    return [
        _profile("logistic-regression-v1", ModelRole.LIGHT, 4.0),
        _profile("random-forest-v1", ModelRole.BALANCED, 6.0),
        _profile("mlp-v1", ModelRole.HEAVY, 8.0),
    ]


def _devices(queue_capacity: int = 2) -> DeviceCatalog:
    return DeviceCatalog(
        devices=[
            Device(
                id="mobile",
                speed_multiplier=2.5,
                active_power_units=1.0,
                queue_capacity=queue_capacity,
                supported_models=["light", "balanced"],
            ),
            Device(
                id="gateway",
                speed_multiplier=1.3,
                active_power_units=2.0,
                queue_capacity=queue_capacity,
                supported_models=["light", "balanced", "heavy"],
            ),
            Device(
                id="edge-server",
                speed_multiplier=0.6,
                active_power_units=4.0,
                queue_capacity=queue_capacity,
                supported_models=["light", "balanced", "heavy"],
            ),
        ]
    )


def _network(jitter_ms: float = 0.0) -> NetworkCatalog:
    return NetworkCatalog(
        links=[
            NetworkLink(
                link_id="mobile-to-gateway",
                source_device_id="mobile",
                destination_device_id="gateway",
                base_latency_ms=10.0,
                bandwidth_mbps=10.0,
                jitter_ms=jitter_ms,
            ),
            NetworkLink(
                link_id="mobile-to-edge-server",
                source_device_id="mobile",
                destination_device_id="edge-server",
                base_latency_ms=20.0,
                bandwidth_mbps=5.0,
                jitter_ms=jitter_ms,
            ),
        ]
    )


def _config() -> SimulationConfig:
    return SimulationConfig(
        run_id="test-run",
        random_seed=7,
        scheduler_overhead_ms=0.0,
        network_energy_per_kilobyte=0.5,
        return_payload_size_bytes=100,
    )


def _request(
    request_id: str,
    *,
    arrival: float = 0.0,
    deadline: float = 100.0,
    input_size: int = 1000,
    feature_id: int = 0,
) -> InferenceRequest:
    return InferenceRequest(
        request_id=request_id,
        arrival_time_ms=arrival,
        deadline_ms=deadline,
        minimum_accuracy=0.9,
        input_size_bytes=input_size,
        source_device_id="mobile",
        true_label=feature_id % 6 + 1,
        feature_vector_id=feature_id,
    )


def _decision(request: InferenceRequest, device_id: str, model_id: str) -> AssignmentDecision:
    return AssignmentDecision(
        request_id=request.request_id,
        device_id=device_id,
        model_id=model_id,
        predicted_completion_ms=request.arrival_time_ms,
        predicted_energy_units=0.0,
        predicted_model_accuracy=0.95,
        expected_to_meet_deadline=True,
        decision_reason="test fixture decision",
    )


def _run(
    requests: list[InferenceRequest],
    decisions: list[AssignmentDecision],
    predictor: CountingPredictionProvider,
    *,
    queue_capacity: int = 2,
    jitter_ms: float = 0.0,
):
    trace = WorkloadTrace(trace_id="test-trace", seed=7, requests=requests)
    provider = FixedAssignmentProvider(
        AssignmentPlan(scheduler_name="fixed-test", decisions=decisions)
    )
    engine = SimulationEngine(
        devices=_devices(queue_capacity),
        network=_network(jitter_ms),
        model_profiles=_profiles(),
        config=_config(),
        prediction_provider=predictor,
    )
    return engine.run(trace, provider)


def test_network_local_remote_and_energy_calculations() -> None:
    link = _network().links[0]
    assert transfer_time_ms(1000, link) == pytest.approx(10.8)
    local = network_timing(
        source_device_id="mobile",
        destination_device_id="mobile",
        input_size_bytes=1000,
        return_size_bytes=100,
        links=_network().links,
        random_seed=7,
        request_id="local",
    )
    assert local.upload_time_ms == 0.0
    assert local.return_time_ms == 0.0
    assert compute_energy_units(_devices().devices[1], 5.2) == pytest.approx(10.4)
    assert network_energy_units(1100, 0.5) == pytest.approx(0.537109375)
    total = estimate_energy(
        device=_devices().devices[1],
        inference_time_ms=5.2,
        transferred_bytes=1100,
        network_energy_per_kilobyte=0.5,
    )
    assert total.estimated_total_energy_units == pytest.approx(10.937109375)


def test_compatibility_and_simulated_inference_scaling() -> None:
    mobile = _devices().devices[0]
    light, _, heavy = _profiles()
    assert is_device_model_compatible(mobile, light)
    assert not is_device_model_compatible(mobile, heavy)
    assert simulated_inference_time_ms(light, mobile) == 10.0


def test_local_and_remote_lifecycle_component_accounting() -> None:
    local_request = _request("local", feature_id=0)
    remote_request = _request("remote", arrival=30.0, feature_id=1)
    predictor = CountingPredictionProvider()
    result = _run(
        [local_request, remote_request],
        [
            _decision(local_request, "mobile", "logistic-regression-v1"),
            _decision(remote_request, "gateway", "logistic-regression-v1"),
        ],
        predictor,
    )

    local, remote = result.requests
    assert local.upload_time_ms == local.return_time_ms == 0.0
    assert local.inference_time_ms == local.end_to_end_latency_ms == 10.0
    assert remote.upload_time_ms == pytest.approx(10.8)
    assert remote.return_time_ms == pytest.approx(10.08)
    assert remote.inference_time_ms == pytest.approx(5.2)
    assert remote.end_to_end_latency_ms == pytest.approx(26.08)
    assert remote.energy.compute_energy_units == pytest.approx(10.4)
    assert remote.energy.network_energy_units == pytest.approx(0.537109375)
    assert remote.actual_prediction == 2
    assert remote.prediction_correct
    assert result.completed_requests == 2
    assert result.rejected_requests == 0
    decision_event = next(
        event
        for event in result.events
        if event.event_type is EventType.SCHEDULER_DECISION and event.request_id == "remote"
    )
    assert decision_event.details["decision_reason"] == "test fixture decision"
    assert decision_event.details["predicted_model_accuracy"] == 0.95


def test_waiting_queue_capacity_rejects_deterministically() -> None:
    requests = [_request(f"request-{index}", feature_id=index) for index in range(3)]
    decisions = [_decision(request, "mobile", "logistic-regression-v1") for request in requests]
    predictor = CountingPredictionProvider()
    result = _run(requests, decisions, predictor, queue_capacity=1)

    assert [request.status for request in result.requests] == [
        RequestStatus.COMPLETED,
        RequestStatus.COMPLETED,
        RequestStatus.REJECTED,
    ]
    assert result.requests[1].queue_wait_ms == 10.0
    assert result.requests[2].rejection_reason == "device_queue_full"
    assert result.requests[2].energy.network_energy_units == 0.0
    assert len(predictor.calls) == 2
    assert sum(event.event_type is EventType.REQUEST_REJECTED for event in result.events) == 1


def test_invalid_assignment_is_observed_then_rejected_without_prediction() -> None:
    request = _request("invalid")
    predictor = CountingPredictionProvider()
    result = _run(
        [request],
        [_decision(request, "mobile", "mlp-v1")],
        predictor,
    )

    assert result.requests[0].status is RequestStatus.REJECTED
    assert result.requests[0].rejection_reason == "incompatible_device_model"
    assert predictor.calls == []
    assert [event.event_type for event in result.events] == [
        EventType.REQUEST_ARRIVED,
        EventType.SCHEDULER_DECISION,
        EventType.REQUEST_REJECTED,
    ]


def test_deadline_equality_meets_and_late_completion_emits_miss() -> None:
    on_time = _request("on-time", deadline=10.0)
    late = _request("late", arrival=20.0, deadline=9.9)
    result = _run(
        [on_time, late],
        [
            _decision(on_time, "mobile", "logistic-regression-v1"),
            _decision(late, "mobile", "logistic-regression-v1"),
        ],
        CountingPredictionProvider(),
    )

    assert result.requests[0].deadline_met
    assert not result.requests[1].deadline_met
    assert sum(event.event_type is EventType.DEADLINE_MISSED for event in result.events) == 1


def test_full_replay_including_seeded_jitter_is_deterministic() -> None:
    requests = [
        _request("first", feature_id=0),
        _request("second", feature_id=1),
    ]
    decisions = [_decision(request, "gateway", "logistic-regression-v1") for request in requests]
    first = _run(requests, decisions, CountingPredictionProvider(), jitter_ms=1.0)
    second = _run(requests, decisions, CountingPredictionProvider(), jitter_ms=1.0)
    assert first.model_dump() == second.model_dump()
    assert [event.sequence for event in first.events] == list(range(len(first.events)))
    assert [event.timestamp_ms for event in first.events] == sorted(
        event.timestamp_ms for event in first.events
    )


def test_small_real_estimator_prediction_flows_through_engine(
    mini_uci_har_root,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)
    preprocessor = StandardScaler().fit(dataset.train.features)
    estimator = LogisticRegression(random_state=1).fit(
        preprocessor.transform(dataset.train.features), dataset.train.labels
    )
    metadata = ModelArtifactMetadata(
        model_id="logistic-regression-v1",
        display_name="Light Model",
        computational_role=ModelRole.LIGHT,
        random_seed=1,
        estimator_class="LogisticRegression",
        preprocessor_class="StandardScaler",
        hyperparameters={"random_state": 1},
        feature_count=dataset.feature_count,
        class_labels=[1, 2],
        train_sample_count=4,
        test_sample_count=2,
        accuracy=1.0,
        macro_f1=1.0,
        model_size_bytes=1,
        preprocessor_size_bytes=1,
        sklearn_version="test",
        trained_at_utc=datetime.now(UTC),
    )
    service = ModelPredictionService(
        dataset,
        {
            "logistic-regression-v1": LoadedModelArtifacts(
                metadata=metadata,
                preprocessor=preprocessor,
                estimator=estimator,
            )
        },
    )
    request = _request("real-estimator", input_size=24, feature_id=0)
    result = SimulationEngine(
        devices=_devices(),
        network=_network(),
        model_profiles=_profiles(),
        config=_config(),
        prediction_provider=service,
    ).run(
        WorkloadTrace(trace_id="real-estimator", seed=1, requests=[request]),
        FixedAssignmentProvider(
            AssignmentPlan(
                scheduler_name="fixed-test",
                decisions=[_decision(request, "mobile", "logistic-regression-v1")],
            )
        ),
    )
    assert result.requests[0].actual_prediction == int(
        estimator.predict(preprocessor.transform(dataset.test.features[:1]))[0]
    )
