"""Minimum Completion Time baseline."""

from edgeweaver.domain import AssignmentDecision, InferenceRequest, SimulationState
from edgeweaver.schedulers.candidates import (
    NoValidAssignmentError,
    decision_from_candidate,
    eligible_candidates,
    estimate_candidates,
)


class MinimumCompletionTimeScheduler:
    name = "min_completion"

    def select_assignment(
        self,
        request: InferenceRequest,
        state: SimulationState,
    ) -> AssignmentDecision:
        candidates = estimate_candidates(request, state)
        eligible = tuple(
            candidate for candidate in eligible_candidates(candidates) if candidate.queue_admissible
        )
        if not eligible:
            raise NoValidAssignmentError(
                f"no admissible compatible model serves request {request.request_id}"
            )
        selected = min(
            eligible,
            key=lambda candidate: (
                candidate.predicted_completion_ms,
                candidate.device_id,
                candidate.model_id,
            ),
        )
        return decision_from_candidate(
            request,
            selected,
            candidates,
            "selected candidate with earliest predicted completion",
        )
