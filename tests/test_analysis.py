import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from edgeweaver.analysis import (
    AnalysisRunRecord,
    AnalysisValidationError,
    build_analysis_tables,
    load_analysis_run_record,
    load_per_run_metrics,
    save_analysis_tables,
    validate_experiment_matrix,
)
from edgeweaver.metrics import (
    DeadlineMissCause,
    DeadlineMissCauseMetric,
    DeviceUtilizationMetric,
    ModelSelectionMetric,
    PerRunMetrics,
    save_run_metrics_json,
)


def _summary(*, scenario: str, variant: str, seed: int, value: float) -> PerRunMetrics:
    return PerRunMetrics(
        run_id=f"{scenario}-{variant}-{seed}",
        scenario_id=scenario,
        scheduler_name=variant,
        seed=seed,
        simulation_duration_ms=1000.0,
        total_requests=10,
        assigned_requests=10,
        completed_requests=10,
        rejected_requests=0,
        deadlines_met=8,
        deadline_misses=2,
        useful_valid_predictions=7,
        correct_predictions=9,
        deadline_satisfaction_rate=value,
        useful_goodput_requests_per_second=value * 10.0,
        mean_end_to_end_latency_ms=value * 100.0,
        p95_end_to_end_latency_ms=value * 200.0,
        actual_prediction_accuracy=value,
        total_estimated_energy_units=value * 1000.0,
        estimated_energy_per_completed_request_units=value * 100.0,
        device_utilization=[
            DeviceUtilizationMetric(
                device_id="mobile", busy_time_ms=value * 1000.0, utilization=value
            )
        ],
        model_selection_distribution=[
            ModelSelectionMetric(
                model_id="light-v1",
                model_role="light",
                assigned_requests=10,
                percentage=value * 100.0,
            )
        ],
        deadline_miss_causes=[
            DeadlineMissCauseMetric(
                cause=cause,
                count=2 if cause is DeadlineMissCause.QUEUE_DELAY else 0,
                rate=1.0 if cause is DeadlineMissCause.QUEUE_DELAY else 0.0,
            )
            for cause in DeadlineMissCause
        ],
    )


def _record(*, scenario: str, variant: str, seed: int, value: float) -> AnalysisRunRecord:
    trace_hash = hashlib.sha256(f"{scenario}-{seed}".encode()).hexdigest()
    return AnalysisRunRecord(
        variant_id=variant,
        trace_id=f"{scenario}-seed-{seed}",
        trace_sha256=trace_hash,
        summary=_summary(scenario=scenario, variant=variant, seed=seed, value=value),
    )


def _matrix() -> list[AnalysisRunRecord]:
    return [
        _record(scenario="normal", variant=variant, seed=seed, value=seed / 10.0)
        for variant in ("round_robin", "edgeweaver")
        for seed in (1, 3)
    ]


def test_exact_matrix_validation_and_paired_trace_identity() -> None:
    report = validate_experiment_matrix(
        _matrix(),
        expected_scenarios=["normal"],
        expected_variants=["round_robin", "edgeweaver"],
        expected_seeds=[1, 3],
    )

    assert report.expected_run_count == report.valid_run_count == 4
    assert (report.scenario_count, report.variant_count, report.seed_count) == (1, 2, 2)


@pytest.mark.parametrize("failure", ["duplicate", "missing", "unexpected", "trace"])
def test_matrix_validation_rejects_invalid_or_unpaired_inputs(failure: str) -> None:
    records = _matrix()
    if failure == "duplicate":
        records.append(records[0])
        match = "duplicate logical runs"
    elif failure == "missing":
        records.pop()
        match = "missing="
    elif failure == "unexpected":
        records[-1] = _record(scenario="normal", variant="other", seed=3, value=0.3)
        match = "unexpected="
    else:
        records[-1] = records[-1].model_copy(update={"trace_sha256": "b" * 64})
        match = "different request traces"

    with pytest.raises(AnalysisValidationError, match=match):
        validate_experiment_matrix(
            records,
            expected_scenarios=["normal"],
            expected_variants=["round_robin", "edgeweaver"],
            expected_seeds=[1, 3],
        )


def test_matrix_validation_requires_nested_research_metrics() -> None:
    records = _matrix()
    records[0] = records[0].model_copy(
        update={"summary": records[0].summary.model_copy(update={"device_utilization": []})}
    )
    with pytest.raises(AnalysisValidationError, match="no device-utilization metrics"):
        validate_experiment_matrix(
            records,
            expected_scenarios=["normal"],
            expected_variants=["round_robin", "edgeweaver"],
            expected_seeds=[1, 3],
        )


def test_aggregation_uses_sample_standard_deviation_and_preserves_seed_values() -> None:
    tables = build_analysis_tables(_matrix())
    result = next(
        row
        for row in tables.aggregate_scalar
        if row.variant_id == "edgeweaver" and row.metric == "deadline_satisfaction_rate"
    )

    assert result.run_count == 2
    assert result.mean == pytest.approx(0.2)
    assert result.sample_std == pytest.approx(math.sqrt(0.02))
    assert [(item.seed, item.value) for item in result.seed_values] == [(1, 0.1), (3, 0.3)]

    utilization = next(
        row
        for row in tables.device_utilization_aggregate
        if row.variant_id == "round_robin" and row.dimension == "mobile"
    )
    assert utilization.mean == pytest.approx(0.2)
    assert utilization.unit == "ratio"
    assert len(tables.per_run_scalar) == 4
    assert len(tables.model_selection_per_run) == 4
    assert len(tables.deadline_miss_causes_per_run) == 4 * len(DeadlineMissCause) * 2


def test_single_seed_aggregate_has_zero_sample_standard_deviation() -> None:
    tables = build_analysis_tables(
        [_record(scenario="bursty", variant="edgeweaver", seed=1, value=0.5)]
    )
    assert all(row.sample_std == 0.0 for row in tables.aggregate_scalar)


def test_tidy_exports_are_stable_atomic_and_machine_readable(tmp_path: Path) -> None:
    tables = build_analysis_tables(_matrix())
    paths = save_analysis_tables(tmp_path / "results", tables)
    first_contents = {path.name: path.read_bytes() for path in paths}
    assert len(paths) == 8
    assert not list((tmp_path / "results").glob("*.tmp"))

    save_analysis_tables(tmp_path / "results", tables)
    assert {path.name: path.read_bytes() for path in paths} == first_contents

    with (tmp_path / "results" / "aggregate_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    seed_values = json.loads(rows[0]["seed_values"])
    assert [item["seed"] for item in seed_values] == [1, 3]
    assert "estimated_normalized_units_per_request" in {row["unit"] for row in rows}


def test_saved_summary_loader_is_strict(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    expected = _summary(scenario="normal", variant="edgeweaver", seed=1, value=0.5)
    save_run_metrics_json(summary_path, expected)
    assert load_per_run_metrics(summary_path) == expected

    summary_path.write_text('{"format_version": NaN}', encoding="utf-8")
    with pytest.raises(AnalysisValidationError, match="invalid per-run metrics"):
        load_per_run_metrics(summary_path)


def test_completion_record_adapter_checks_identity_and_summary_hash(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = _summary(scenario="normal", variant="edgeweaver", seed=1, value=0.5)
    save_run_metrics_json(summary_path, summary)
    completion_path = tmp_path / "record.json"
    record = {
        "run_id": summary.run_id,
        "scenario_id": summary.scenario_id,
        "scheduler_name": summary.scheduler_name,
        "seed": summary.seed,
        "trace_id": "normal-seed-1",
        "trace_sha256": "a" * 64,
        "artifact_sha256": {"summary_json": hashlib.sha256(summary_path.read_bytes()).hexdigest()},
    }
    completion_path.write_text(json.dumps(record), encoding="utf-8")

    loaded = load_analysis_run_record(completion_path, summary_path)
    assert loaded.summary == summary
    assert loaded.trace_sha256 == "a" * 64

    record["seed"] = 2
    completion_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(AnalysisValidationError, match="summary seeds differ"):
        load_analysis_run_record(completion_path, summary_path)
