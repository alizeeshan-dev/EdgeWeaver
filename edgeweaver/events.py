"""Structured raw event-log export for one simulation run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from edgeweaver.domain import DomainModel, SimulationRunResult, StructuredEvent


class EventLog(DomainModel):
    """Raw lifecycle events kept separate from derived metric summaries."""

    format_version: Literal["edgeweaver-event-log-v1"] = "edgeweaver-event-log-v1"
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    scheduler_name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    events: list[StructuredEvent]


def event_log_from_run(run: SimulationRunResult, *, scenario_id: str) -> EventLog:
    return EventLog(
        run_id=run.run_id,
        scenario_id=scenario_id,
        scheduler_name=run.scheduler_name,
        seed=run.random_seed,
        events=run.events,
    )


def save_event_log(path: Path, event_log: EventLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(event_log.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
