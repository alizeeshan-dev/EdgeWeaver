from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import api.main as api_main
from api.service import _activity_labels

client = TestClient(api_main.app)


def test_local_api_exposes_project_configuration_and_saved_results() -> None:
    assert client.get("/api/health").json() == {"status": "ok"}

    project = client.get("/api/project")
    assert project.status_code == 200
    assert [device["id"] for device in project.json()["devices"]] == [
        "mobile",
        "gateway",
        "edge-server",
    ]

    models = client.get("/api/models")
    assert models.status_code == 200
    assert [profile["computational_role"] for profile in models.json()] == [
        "light",
        "balanced",
        "heavy",
    ]

    assert len(client.get("/api/scenarios").json()) == 4
    assert len(client.get("/api/schedulers").json()) == 4

    results = client.get("/api/experiments/results")
    assert results.status_code == 200
    assert len(results.json()["figures"]) == 8
    assert len(results.json()["rows"]) > 0


def test_saved_simulation_bundle_contains_trace_candidates_and_events() -> None:
    run_id = "core-network_slowdown-fastest_device-seed-1"
    response = client.get(f"/api/simulations/{run_id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["run_id"] == run_id
    assert payload["summary"]["total_requests"] == len(payload["requests"])
    assert payload["requests"][0]["request"]["true_label_name"]
    assert payload["requests"][0]["execution"]["assignment"]["candidates"]

    events = client.get(f"/api/simulations/{run_id}/events")
    assert events.status_code == 200
    assert events.json()["events"][0]["event_type"] == "REQUEST_ARRIVED"


def test_saved_simulation_does_not_require_raw_activity_label_file(
    monkeypatch: Any,
) -> None:
    def denied_read(*args: Any, **kwargs: Any) -> str:
        raise PermissionError("raw dataset is not readable")

    monkeypatch.setattr("pathlib.Path.read_text", denied_read)
    labels = _activity_labels()

    assert labels[1] == "Walking"
    assert labels[2] == "Walking Upstairs"


def test_simulation_run_id_rejects_path_traversal() -> None:
    response = client.get("/api/simulations/not%2Fsafe")
    assert response.status_code in {400, 404}


def test_post_simulation_delegates_to_research_service(monkeypatch: Any) -> None:
    def fake_execute(request: Any) -> dict[str, Any]:
        return {
            "run_id": f"core-{request.scenario_id}-{request.scheduler_name}-seed-{request.seed}",
            "executed": True,
            "summary": {"total_requests": 3},
        }

    monkeypatch.setattr(api_main, "execute_simulation", fake_execute)
    response = client.post(
        "/api/simulations",
        json={"scenario_id": "bursty", "scheduler_name": "edgeweaver", "seed": 9},
    )
    assert response.status_code == 200
    assert response.json()["run_id"] == "core-bursty-edgeweaver-seed-9"

    invalid = client.post(
        "/api/simulations",
        json={"scenario_id": "unknown", "scheduler_name": "edgeweaver", "seed": 1},
    )
    assert invalid.status_code == 422
