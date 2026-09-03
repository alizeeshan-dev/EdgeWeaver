from edgeweaver.domain import InferenceRequest
from edgeweaver.simulation import meets_deadline


def test_relative_deadline_is_calculated_from_arrival() -> None:
    request = InferenceRequest(
        request_id="deadline-check",
        arrival_time_ms=1_000.0,
        deadline_ms=150.0,
        minimum_accuracy=0.5,
        input_size_bytes=128,
        source_device_id="mobile",
        true_label=1,
        feature_vector_id=1,
    )

    assert request.absolute_deadline_ms == 1_150.0
    assert meets_deadline(request, 1_150.0)
    assert not meets_deadline(request, 1_150.001)
