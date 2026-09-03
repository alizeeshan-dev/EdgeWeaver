"""Fastest isolated-device baseline that intentionally ignores queue state."""

from edgeweaver.domain import AssignmentDecision, InferenceRequest, SimulationState
from edgeweaver.schedulers.candidates import (
    NoValidAssignmentError,
    decision_from_candidate,
    eligible_candidates,
    estimate_candidates,
)


class FastestDeviceScheduler:
    name = "fastest_device"

    def select_assignment(
        self,
        request: InferenceRequest,
        state: SimulationState,
    ) -> AssignmentDecision:
        candidates = estimate_candidates(request, state)
        eligible = eligible_candidates(candidates)
        if not eligible:
            raise NoValidAssignmentError(
                f"no compatible model meets request {request.request_id} accuracy constraint"
            )

        per_device = []
        for device in state.device_profiles:
            device_candidates = [
                candidate for candidate in eligible if candidate.device_id == device.id
            ]
            if not device_candidates:
                continue
            per_device.append(
                min(
                    device_candidates,
                    key=lambda candidate: (
                        -candidate.model_accuracy,
                        candidate.predicted_inference_time_ms,
                        candidate.model_id,
                    ),
                )
            )
        selected = min(
            per_device,
            key=lambda candidate: (
                candidate.predicted_inference_time_ms,
                candidate.device_id,
                candidate.model_id,
            ),
        )
        return decision_from_candidate(
            request,
            selected,
            candidates,
            "selected device with lowest isolated inference time",
        )
