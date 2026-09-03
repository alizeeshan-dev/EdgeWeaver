from edgeweaver.config import LoadedConfiguration
from edgeweaver.domain import EventType
from edgeweaver.simulation import run_fixture_simulation


def test_five_request_fixture_has_expected_metrics(
    phase1_configuration: LoadedConfiguration,
) -> None:
    result = run_fixture_simulation(phase1_configuration)

    assert result.summary.total_requests == 5
    assert result.summary.deadlines_met == 3
    assert result.summary.deadline_satisfaction_rate == 0.6
    assert [metric.queue_wait_ms for metric in result.summary.requests] == [0.0, 1.5, 3.0, 4.5, 5.5]
    assert [metric.completion_time_ms for metric in result.summary.requests] == [
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
    ]
    assert [metric.end_to_end_latency_ms for metric in result.summary.requests] == [
        2.0,
        3.5,
        5.0,
        6.5,
        7.5,
    ]
    assert [metric.deadline_met for metric in result.summary.requests] == [
        True,
        False,
        True,
        False,
        True,
    ]


def test_event_order_is_deterministic(phase1_configuration: LoadedConfiguration) -> None:
    first = run_fixture_simulation(phase1_configuration)
    second = run_fixture_simulation(phase1_configuration)

    assert first.model_dump() == second.model_dump()
    assert [(event.timestamp_ms, event.event_type, event.request_id) for event in first.events] == [
        (0.0, EventType.REQUEST_ARRIVED, "request-001"),
        (0.0, EventType.REQUEST_QUEUED, "request-001"),
        (0.0, EventType.INFERENCE_STARTED, "request-001"),
        (0.5, EventType.REQUEST_ARRIVED, "request-002"),
        (0.5, EventType.REQUEST_QUEUED, "request-002"),
        (1.0, EventType.REQUEST_ARRIVED, "request-003"),
        (1.0, EventType.REQUEST_QUEUED, "request-003"),
        (1.5, EventType.REQUEST_ARRIVED, "request-004"),
        (1.5, EventType.REQUEST_QUEUED, "request-004"),
        (2.0, EventType.INFERENCE_COMPLETED, "request-001"),
        (2.0, EventType.INFERENCE_STARTED, "request-002"),
        (2.5, EventType.REQUEST_ARRIVED, "request-005"),
        (2.5, EventType.REQUEST_QUEUED, "request-005"),
        (4.0, EventType.INFERENCE_COMPLETED, "request-002"),
        (4.0, EventType.INFERENCE_STARTED, "request-003"),
        (6.0, EventType.INFERENCE_COMPLETED, "request-003"),
        (6.0, EventType.INFERENCE_STARTED, "request-004"),
        (8.0, EventType.INFERENCE_COMPLETED, "request-004"),
        (8.0, EventType.INFERENCE_STARTED, "request-005"),
        (10.0, EventType.INFERENCE_COMPLETED, "request-005"),
    ]
