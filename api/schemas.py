"""Request schemas for the local-only FastAPI bridge."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ScenarioId = Literal["normal", "bursty", "network_slowdown", "device_slowdown"]
SchedulerId = Literal["round_robin", "fastest_device", "min_completion", "edgeweaver"]


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: ScenarioId
    scheduler_name: SchedulerId
    seed: int = Field(default=1, ge=0)
    force: bool = False
