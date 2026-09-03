"""Deadline-feasible, energy-first EdgeWeaver scheduler with EWMA adaptation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from edgeweaver.compatibility import is_device_model_compatible
from edgeweaver.domain import (
    AssignmentDecision,
    CandidateEstimate,
    ExecutionObservation,
    InferenceRequest,
    LatencyEstimateUpdate,
    SimulationState,
)
from edgeweaver.schedulers.candidates import (
    NoValidAssignmentError,
    decision_from_candidate,
    eligible_candidates,
    estimate_candidates,
)
from edgeweaver.simulation import simulated_inference_time_ms

DEFAULT_EWMA_ALPHA = 0.2
TIE_RELATIVE_TOLERANCE = 1e-12
TIE_ABSOLUTE_TOLERANCE = 1e-12
EdgeWeaverVariant = Literal[
    "edgeweaver",
    "edgeweaver_no_model_switching",
    "edgeweaver_no_online_update",
]
EDGEWEAVER_VARIANTS: tuple[EdgeWeaverVariant, ...] = (
    "edgeweaver",
    "edgeweaver_no_model_switching",
    "edgeweaver_no_online_update",
)


def ewma_latency_ms(old_estimate_ms: float, observed_latency_ms: float, alpha: float) -> float:
    """Update one inference-latency estimate without modifying measured profiles."""

    if old_estimate_ms <= 0.0 or observed_latency_ms <= 0.0:
        raise ValueError("latency values must be positive")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EWMA alpha must be greater than zero and at most one")
    return alpha * observed_latency_ms + (1.0 - alpha) * old_estimate_ms


def _minimum_tied(
    candidates: Sequence[CandidateEstimate],
    attribute: str,
) -> tuple[CandidateEstimate, ...]:
    minimum = min(float(getattr(candidate, attribute)) for candidate in candidates)
    return tuple(
        candidate
        for candidate in candidates
        if math.isclose(
            float(getattr(candidate, attribute)),
            minimum,
            rel_tol=TIE_RELATIVE_TOLERANCE,
            abs_tol=TIE_ABSOLUTE_TOLERANCE,
        )
    )


class EdgeWeaverScheduler:
    def __init__(
        self,
        *,
        alpha: float = DEFAULT_EWMA_ALPHA,
        model_switching_enabled: bool = True,
        online_updates_enabled: bool = True,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EWMA alpha must be greater than zero and at most one")
        if not model_switching_enabled and not online_updates_enabled:
            raise ValueError("EdgeWeaver experiment ablations must be isolated")
        self.alpha = alpha
        self.model_switching_enabled = model_switching_enabled
        self.online_updates_enabled = online_updates_enabled
        self.name: str = (
            "edgeweaver_no_model_switching"
            if not model_switching_enabled
            else "edgeweaver_no_online_update"
            if not online_updates_enabled
            else "edgeweaver"
        )
        self._latency_estimates_ms: dict[tuple[str, str], float] = {}

    def reset(self) -> None:
        """Clear runtime estimates before an independent simulation run."""

        self._latency_estimates_ms.clear()

    @property
    def latency_estimates_ms(self) -> Mapping[tuple[str, str], float]:
        return dict(self._latency_estimates_ms)

    def _initialize_missing_estimates(self, state: SimulationState) -> None:
        for device in state.device_profiles:
            for profile in state.model_profiles:
                if is_device_model_compatible(device, profile):
                    self._latency_estimates_ms.setdefault(
                        (device.id, profile.model_id),
                        simulated_inference_time_ms(profile, device),
                    )

    def select_assignment(
        self,
        request: InferenceRequest,
        state: SimulationState,
    ) -> AssignmentDecision:
        self._initialize_missing_estimates(state)
        candidates = estimate_candidates(
            request,
            state,
            inference_estimates_ms=self._latency_estimates_ms,
        )
        valid = tuple(
            candidate for candidate in eligible_candidates(candidates) if candidate.queue_admissible
        )
        if not valid:
            raise NoValidAssignmentError(
                f"no valid compatible assignment serves request {request.request_id}"
            )

        selection_pool = valid
        restriction = ""
        if not self.model_switching_enabled:
            highest_accuracy = max(candidate.model_accuracy for candidate in valid)
            selection_pool = tuple(
                candidate
                for candidate in valid
                if math.isclose(
                    candidate.model_accuracy,
                    highest_accuracy,
                    rel_tol=TIE_RELATIVE_TOLERANCE,
                    abs_tol=TIE_ABSOLUTE_TOLERANCE,
                )
            )
            restriction = "; restricted to highest measured-accuracy qualifying model"

        feasible = tuple(
            candidate for candidate in selection_pool if candidate.expected_to_meet_deadline
        )
        if feasible:
            energy_tied = _minimum_tied(feasible, "predicted_energy_units")
            completion_tied = _minimum_tied(energy_tied, "predicted_completion_ms")
            selected = min(
                completion_tied,
                key=lambda candidate: (candidate.device_id, candidate.model_id),
            )
            reason = (
                "selected lowest predicted energy among deadline-feasible candidates" + restriction
            )
        else:
            completion_tied = _minimum_tied(selection_pool, "predicted_completion_ms")
            selected = min(
                completion_tied,
                key=lambda candidate: (candidate.device_id, candidate.model_id),
            )
            reason = (
                "no candidate predicted to meet deadline; selected earliest completion"
                + restriction
            )
        return decision_from_candidate(request, selected, candidates, reason)

    def observe_execution(
        self,
        observation: ExecutionObservation,
    ) -> LatencyEstimateUpdate | None:
        if not self.online_updates_enabled:
            return None
        key = (observation.device_id, observation.model_id)
        try:
            old_estimate_ms = self._latency_estimates_ms[key]
        except KeyError as error:
            raise ValueError(
                f"cannot update uninitialized latency estimate for {key[0]}/{key[1]}"
            ) from error
        new_estimate_ms = ewma_latency_ms(
            old_estimate_ms,
            observation.inference_time_ms,
            self.alpha,
        )
        if new_estimate_ms == old_estimate_ms:
            return None
        self._latency_estimates_ms[key] = new_estimate_ms
        return LatencyEstimateUpdate(
            request_id=observation.request_id,
            timestamp_ms=observation.timestamp_ms,
            device_id=observation.device_id,
            model_id=observation.model_id,
            old_estimate_ms=old_estimate_ms,
            observed_inference_time_ms=observation.inference_time_ms,
            new_estimate_ms=new_estimate_ms,
            alpha=self.alpha,
        )


def create_edgeweaver_variant(
    variant: EdgeWeaverVariant,
    *,
    alpha: float = DEFAULT_EWMA_ALPHA,
) -> EdgeWeaverScheduler:
    """Create one core/ablation variant without registering extra core policies."""

    if variant == "edgeweaver":
        return EdgeWeaverScheduler(alpha=alpha)
    if variant == "edgeweaver_no_model_switching":
        return EdgeWeaverScheduler(alpha=alpha, model_switching_enabled=False)
    if variant == "edgeweaver_no_online_update":
        return EdgeWeaverScheduler(alpha=alpha, online_updates_enabled=False)
    raise ValueError(f"unknown EdgeWeaver variant {variant!r}")
