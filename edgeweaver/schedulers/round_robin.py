"""Deterministic Round Robin baseline."""

from edgeweaver.domain import AssignmentDecision, InferenceRequest, SimulationState
from edgeweaver.schedulers.candidates import (
    NoValidAssignmentError,
    decision_from_candidate,
    estimate_candidates,
)


class RoundRobinScheduler:
    name = "round_robin"

    def __init__(self) -> None:
        self._next_device_index = 0

    def reset(self) -> None:
        """Start an independent simulation run from the first configured device."""

        self._next_device_index = 0

    def select_assignment(
        self,
        request: InferenceRequest,
        state: SimulationState,
    ) -> AssignmentDecision:
        candidates = estimate_candidates(request, state)
        device_ids = [device.id for device in state.device_profiles]
        if not device_ids:
            raise NoValidAssignmentError("no configured devices")

        for offset in range(len(device_ids)):
            index = (self._next_device_index + offset) % len(device_ids)
            device_id = device_ids[index]
            device_candidates = [
                candidate
                for candidate in candidates
                if candidate.eligible and candidate.device_id == device_id
            ]
            if not device_candidates:
                continue
            selected = min(
                device_candidates,
                key=lambda candidate: (
                    candidate.predicted_inference_time_ms,
                    candidate.model_id,
                ),
            )
            self._next_device_index = (index + 1) % len(device_ids)
            return decision_from_candidate(
                request,
                selected,
                candidates,
                "selected next compatible device and fastest qualifying model",
            )
        raise NoValidAssignmentError(
            f"no compatible model meets request {request.request_id} accuracy constraint"
        )
