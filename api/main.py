"""FastAPI application for the local EdgeWeaver presentation interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import SimulationRequest
from api.service import (
    PHASE7_ROOT,
    PROJECT_ROOT,
    event_payload,
    execute_simulation,
    experiment_results_payload,
    failure_cases_payload,
    list_runs,
    model_payload,
    project_payload,
    run_payload,
    scenario_payload,
    scheduler_payload,
)

app = FastAPI(
    title="EdgeWeaver Local API",
    version="0.1.0",
    description="Local-only bridge over the EdgeWeaver research package and saved artifacts.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _call(loader: Callable[[], Any]) -> Any:
    try:
        return loader()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"artifact not found: {error}") from error
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/project")
def project() -> Any:
    return _call(project_payload)


@app.get("/api/models")
def models() -> Any:
    return _call(model_payload)


@app.get("/api/scenarios")
def scenarios() -> Any:
    return _call(scenario_payload)


@app.get("/api/schedulers")
def schedulers() -> Any:
    return scheduler_payload()


@app.get("/api/simulations")
def simulations() -> Any:
    return _call(list_runs)


@app.post("/api/simulations")
def create_simulation(request: SimulationRequest) -> Any:
    return _call(lambda: execute_simulation(request))


@app.get("/api/simulations/{run_id}")
def simulation(run_id: str) -> Any:
    return _call(lambda: run_payload(run_id))


@app.get("/api/simulations/{run_id}/events")
def simulation_events(run_id: str) -> Any:
    return _call(lambda: event_payload(run_id))


@app.get("/api/experiments/results")
def experiment_results() -> Any:
    return _call(experiment_results_payload)


@app.get("/api/experiments/failure-cases")
def failure_cases() -> Any:
    return _call(failure_cases_payload)


@app.get("/api/experiments/figures/{filename}", response_class=FileResponse)
def experiment_figure(filename: str) -> FileResponse:
    if Path(filename).name != filename or not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="invalid figure filename")
    path = PHASE7_ROOT / "figures" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="figure not found")
    return FileResponse(path)


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file() and candidate.resolve().is_relative_to(FRONTEND_DIST):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
