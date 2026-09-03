"""Scheduler boundary and fixed assignment provider used only for engine verification."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

from pydantic import Field, model_validator

from edgeweaver.domain import (
    AssignmentDecision,
    DomainModel,
    ExecutionObservation,
    InferenceRequest,
    LatencyEstimateUpdate,
    SimulationState,
)


class Scheduler(Protocol):
    name: str

    def select_assignment(
        self,
        request: InferenceRequest,
        state: SimulationState,
    ) -> AssignmentDecision: ...


# Phase 3 used this name for its fixed verification provider. Keep it as a
# compatibility alias while all Phase 4 policies implement the common protocol.
AssignmentProvider = Scheduler


class ExecutionObserver(Protocol):
    def observe_execution(
        self,
        observation: ExecutionObservation,
    ) -> LatencyEstimateUpdate | None: ...


class AssignmentPlan(DomainModel):
    scheduler_name: str = Field(min_length=1)
    decisions: list[AssignmentDecision] = Field(min_length=1)

    @model_validator(mode="after")
    def request_ids_are_unique(self) -> Self:
        request_ids = [decision.request_id for decision in self.decisions]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("fixed assignment request IDs must be unique")
        return self


class FixedAssignmentProvider:
    """Replay explicit decisions without implementing a scheduling policy."""

    def __init__(self, plan: AssignmentPlan) -> None:
        self.name = plan.scheduler_name
        self._decisions = {decision.request_id: decision for decision in plan.decisions}

    def select_assignment(
        self,
        request: InferenceRequest,
        state: SimulationState,
    ) -> AssignmentDecision:
        del state
        try:
            return self._decisions[request.request_id]
        except KeyError as error:
            raise ValueError(f"no fixed assignment for request {request.request_id}") from error


def load_assignment_plan(path: Path) -> AssignmentPlan:
    return AssignmentPlan.model_validate_json(path.read_text(encoding="utf-8"))


def save_assignment_plan(path: Path, plan: AssignmentPlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=2) + "\n", encoding="utf-8")
