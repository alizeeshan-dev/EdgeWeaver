"""Validation and cross-seed aggregation for saved experiment summaries.

This module intentionally consumes completed per-run summaries.  It does not run
simulations or depend on plotting, so tables can be regenerated without changing
the experiment inputs.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from edgeweaver.domain import DomainModel
from edgeweaver.metrics import DeadlineMissCause, PerRunMetrics


class AnalysisValidationError(ValueError):
    """Raised when saved experiment records cannot support valid aggregation."""


class AnalysisRunRecord(DomainModel):
    """Minimum run identity required by analysis.

    ``variant_id`` keeps ablations distinct from the four core policies, while
    ``scheduler_name`` remains available in the embedded Phase 6 summary.
    Trace pairing is checked using both the stable trace ID and its content hash.
    """

    variant_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: PerRunMetrics

    @property
    def logical_key(self) -> tuple[str, str, int]:
        return (self.summary.scenario_id, self.variant_id, self.summary.seed)


class MatrixValidationReport(DomainModel):
    expected_run_count: int = Field(ge=0)
    valid_run_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    variant_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)


class SeedMetricValue(DomainModel):
    seed: int = Field(ge=0)
    value: float


class PerRunScalarRow(DomainModel):
    run_id: str
    scenario_id: str
    variant_id: str
    scheduler_name: str
    seed: int
    trace_id: str
    trace_sha256: str
    total_requests: int
    completed_requests: int
    rejected_requests: int
    deadline_satisfaction_rate: float
    useful_goodput_requests_per_second: float
    mean_end_to_end_latency_ms: float
    p95_end_to_end_latency_ms: float
    actual_prediction_accuracy: float
    estimated_energy_per_completed_request_units: float


class PerRunDimensionRow(DomainModel):
    run_id: str
    scenario_id: str
    variant_id: str
    scheduler_name: str
    seed: int
    metric: str
    dimension: str
    value: float
    unit: str


class AggregateMetricRow(DomainModel):
    scenario_id: str
    variant_id: str
    scheduler_name: str
    metric: str
    dimension: str = ""
    unit: str
    run_count: int = Field(ge=1)
    mean: float
    sample_std: float = Field(ge=0.0)
    seed_values: tuple[SeedMetricValue, ...]


class AnalysisTables(DomainModel):
    per_run_scalar: tuple[PerRunScalarRow, ...]
    aggregate_scalar: tuple[AggregateMetricRow, ...]
    device_utilization_per_run: tuple[PerRunDimensionRow, ...]
    device_utilization_aggregate: tuple[AggregateMetricRow, ...]
    model_selection_per_run: tuple[PerRunDimensionRow, ...]
    model_selection_aggregate: tuple[AggregateMetricRow, ...]
    deadline_miss_causes_per_run: tuple[PerRunDimensionRow, ...]
    deadline_miss_causes_aggregate: tuple[AggregateMetricRow, ...]


_SCALAR_METRICS: tuple[tuple[str, str, str], ...] = (
    ("deadline_satisfaction_rate", "deadline_satisfaction_rate", "ratio"),
    (
        "useful_goodput_requests_per_second",
        "useful_goodput_requests_per_second",
        "requests_per_second",
    ),
    ("mean_end_to_end_latency_ms", "mean_end_to_end_latency_ms", "ms"),
    ("p95_end_to_end_latency_ms", "p95_end_to_end_latency_ms", "ms"),
    ("actual_prediction_accuracy", "actual_prediction_accuracy", "ratio"),
    (
        "estimated_energy_per_completed_request_units",
        "estimated_energy_per_completed_request_units",
        "estimated_normalized_units_per_request",
    ),
)


def validate_experiment_matrix(
    records: Sequence[AnalysisRunRecord],
    *,
    expected_scenarios: Sequence[str],
    expected_variants: Sequence[str],
    expected_seeds: Sequence[int],
) -> MatrixValidationReport:
    """Require an exact complete matrix and paired traces across variants.

    The caller should validate the 80-run core matrix separately from the two
    intentionally sparse ablation matrices.
    """

    scenarios = _unique_expected("scenario", expected_scenarios)
    variants = _unique_expected("variant", expected_variants)
    seeds = _unique_expected("seed", expected_seeds)
    expected = {
        (scenario, variant, seed)
        for scenario in scenarios
        for variant in variants
        for seed in seeds
    }

    actual_keys = [record.logical_key for record in records]
    duplicate_keys = sorted(key for key in set(actual_keys) if actual_keys.count(key) > 1)
    if duplicate_keys:
        raise AnalysisValidationError(f"duplicate logical runs: {duplicate_keys}")

    actual = set(actual_keys)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise AnalysisValidationError("experiment matrix mismatch: " + "; ".join(details))

    for record in records:
        _validate_required_nested_metrics(record)

    by_pair: dict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        by_pair[(record.summary.scenario_id, record.summary.seed)].add(
            (record.trace_id, record.trace_sha256)
        )
    mismatched = sorted(pair for pair, identities in by_pair.items() if len(identities) != 1)
    if mismatched:
        raise AnalysisValidationError(
            f"paired runs use different request traces for scenario/seed: {mismatched}"
        )

    return MatrixValidationReport(
        expected_run_count=len(expected),
        valid_run_count=len(records),
        scenario_count=len(scenarios),
        variant_count=len(variants),
        seed_count=len(seeds),
    )


def build_analysis_tables(records: Sequence[AnalysisRunRecord]) -> AnalysisTables:
    """Build stable per-seed and aggregate tables from validated run records."""

    if not records:
        raise AnalysisValidationError("at least one completed run is required for aggregation")
    keys = [record.logical_key for record in records]
    if len(keys) != len(set(keys)):
        raise AnalysisValidationError("cannot aggregate duplicate logical runs")

    ordered = sorted(records, key=lambda item: item.logical_key)
    scalar_rows = tuple(_scalar_run_row(record) for record in ordered)

    scalar_values: list[tuple[AnalysisRunRecord, str, str, float, str]] = []
    device_rows: list[PerRunDimensionRow] = []
    model_rows: list[PerRunDimensionRow] = []
    cause_rows: list[PerRunDimensionRow] = []
    for record in ordered:
        for metric, attribute, unit in _SCALAR_METRICS:
            scalar_values.append(
                (record, metric, "", float(getattr(record.summary, attribute)), unit)
            )
        for utilization in record.summary.device_utilization:
            device_rows.append(
                _dimension_row(
                    record,
                    "device_utilization",
                    utilization.device_id,
                    utilization.utilization,
                    "ratio",
                )
            )
        for selection in record.summary.model_selection_distribution:
            model_rows.append(
                _dimension_row(
                    record,
                    "model_selection_percentage",
                    selection.model_id,
                    selection.percentage,
                    "percent_of_assigned_requests",
                )
            )
        for miss_cause in record.summary.deadline_miss_causes:
            cause_rows.extend(
                (
                    _dimension_row(
                        record,
                        "deadline_miss_cause_count",
                        miss_cause.cause.value,
                        float(miss_cause.count),
                        "requests",
                    ),
                    _dimension_row(
                        record,
                        "deadline_miss_cause_rate",
                        miss_cause.cause.value,
                        miss_cause.rate,
                        "ratio_of_completed_deadline_misses",
                    ),
                )
            )

    return AnalysisTables(
        per_run_scalar=scalar_rows,
        aggregate_scalar=_aggregate_values(scalar_values),
        device_utilization_per_run=tuple(device_rows),
        device_utilization_aggregate=_aggregate_dimension_rows(device_rows),
        model_selection_per_run=tuple(model_rows),
        model_selection_aggregate=_aggregate_dimension_rows(model_rows),
        deadline_miss_causes_per_run=tuple(cause_rows),
        deadline_miss_causes_aggregate=_aggregate_dimension_rows(cause_rows),
    )


def save_analysis_tables(output_directory: Path, tables: AnalysisTables) -> tuple[Path, ...]:
    """Atomically save stable CSV tables and return their paths."""

    outputs: tuple[tuple[str, Sequence[DomainModel]], ...] = (
        ("per_run_results.csv", tables.per_run_scalar),
        ("aggregate_results.csv", tables.aggregate_scalar),
        ("device_utilization_per_run.csv", tables.device_utilization_per_run),
        ("device_utilization_aggregate.csv", tables.device_utilization_aggregate),
        ("model_selection_per_run.csv", tables.model_selection_per_run),
        ("model_selection_aggregate.csv", tables.model_selection_aggregate),
        ("deadline_miss_causes_per_run.csv", tables.deadline_miss_causes_per_run),
        ("deadline_miss_causes_aggregate.csv", tables.deadline_miss_causes_aggregate),
    )
    paths = []
    for filename, rows in outputs:
        path = output_directory / filename
        _atomic_write_csv(path, rows)
        paths.append(path)
    return tuple(paths)


def load_per_run_metrics(path: Path) -> PerRunMetrics:
    """Strictly load one saved Phase 6 metrics summary."""

    try:
        raw = path.read_text(encoding="utf-8")
        json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_json_object)
        return PerRunMetrics.model_validate_json(raw, strict=True)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisValidationError(f"invalid per-run metrics file {path}: {error}") from error


def load_analysis_run_record(completion_record_path: Path, summary_path: Path) -> AnalysisRunRecord:
    """Load and cross-check one experiment completion manifest and summary.

    The manifest is handled as a small JSON contract rather than importing the
    experiment runner, keeping saved-result analysis independent from execution.
    """

    try:
        raw = completion_record_path.read_text(encoding="utf-8")
        payload = json.loads(
            raw, parse_constant=_reject_constant, object_pairs_hook=_unique_json_object
        )
        if not isinstance(payload, dict):
            raise ValueError("completion record must be a JSON object")
        summary = load_per_run_metrics(summary_path)
        variant_id = _required_string(payload, "scheduler_name")
        trace_id = _required_string(payload, "trace_id")
        trace_sha256 = _required_string(payload, "trace_sha256")
        if _required_string(payload, "run_id") != summary.run_id:
            raise ValueError("completion record and summary run IDs differ")
        if _required_string(payload, "scenario_id") != summary.scenario_id:
            raise ValueError("completion record and summary scenarios differ")
        if payload.get("seed") != summary.seed:
            raise ValueError("completion record and summary seeds differ")
        if variant_id != summary.scheduler_name:
            raise ValueError("completion record and summary scheduler names differ")
        artifact_hashes = payload.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict):
            raise ValueError("completion record is missing artifact hashes")
        expected_summary_hash = artifact_hashes.get("summary_json")
        actual_summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if expected_summary_hash != actual_summary_hash:
            raise ValueError("saved summary hash does not match completion record")
        return AnalysisRunRecord(
            variant_id=variant_id,
            trace_id=trace_id,
            trace_sha256=trace_sha256,
            summary=summary,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisValidationError(
            f"invalid completed-run record {completion_record_path}: {error}"
        ) from error


def _unique_expected(label: str, values: Sequence[Any]) -> tuple[Any, ...]:
    result = tuple(values)
    if not result:
        raise AnalysisValidationError(f"expected {label} values cannot be empty")
    if len(result) != len(set(result)):
        raise AnalysisValidationError(f"expected {label} values must be unique")
    return result


def _validate_required_nested_metrics(record: AnalysisRunRecord) -> None:
    summary = record.summary
    if not summary.device_utilization:
        raise AnalysisValidationError(f"run {summary.run_id} has no device-utilization metrics")
    if not summary.model_selection_distribution:
        raise AnalysisValidationError(f"run {summary.run_id} has no model-selection metrics")
    causes = [metric.cause for metric in summary.deadline_miss_causes]
    if len(causes) != len(set(causes)) or set(causes) != set(DeadlineMissCause):
        raise AnalysisValidationError(
            f"run {summary.run_id} does not contain exactly one metric for each miss cause"
        )


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"completion record requires non-empty string {key!r}")
    return value


def _scalar_run_row(record: AnalysisRunRecord) -> PerRunScalarRow:
    summary = record.summary
    return PerRunScalarRow(
        run_id=summary.run_id,
        scenario_id=summary.scenario_id,
        variant_id=record.variant_id,
        scheduler_name=summary.scheduler_name,
        seed=summary.seed,
        trace_id=record.trace_id,
        trace_sha256=record.trace_sha256,
        total_requests=summary.total_requests,
        completed_requests=summary.completed_requests,
        rejected_requests=summary.rejected_requests,
        deadline_satisfaction_rate=summary.deadline_satisfaction_rate,
        useful_goodput_requests_per_second=summary.useful_goodput_requests_per_second,
        mean_end_to_end_latency_ms=summary.mean_end_to_end_latency_ms,
        p95_end_to_end_latency_ms=summary.p95_end_to_end_latency_ms,
        actual_prediction_accuracy=summary.actual_prediction_accuracy,
        estimated_energy_per_completed_request_units=(
            summary.estimated_energy_per_completed_request_units
        ),
    )


def _dimension_row(
    record: AnalysisRunRecord, metric: str, dimension: str, value: float, unit: str
) -> PerRunDimensionRow:
    return PerRunDimensionRow(
        run_id=record.summary.run_id,
        scenario_id=record.summary.scenario_id,
        variant_id=record.variant_id,
        scheduler_name=record.summary.scheduler_name,
        seed=record.summary.seed,
        metric=metric,
        dimension=dimension,
        value=value,
        unit=unit,
    )


def _aggregate_values(
    values: Iterable[tuple[AnalysisRunRecord, str, str, float, str]],
) -> tuple[AggregateMetricRow, ...]:
    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for record, metric, dimension, value, unit in values:
        key = (
            record.summary.scenario_id,
            record.variant_id,
            record.summary.scheduler_name,
            metric,
            dimension,
            unit,
        )
        grouped[key].append((record.summary.seed, value))
    return tuple(_aggregate_group(key, seed_values) for key, seed_values in sorted(grouped.items()))


def _aggregate_dimension_rows(
    rows: Sequence[PerRunDimensionRow],
) -> tuple[AggregateMetricRow, ...]:
    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        key = (
            row.scenario_id,
            row.variant_id,
            row.scheduler_name,
            row.metric,
            row.dimension,
            row.unit,
        )
        grouped[key].append((row.seed, row.value))
    return tuple(_aggregate_group(key, seed_values) for key, seed_values in sorted(grouped.items()))


def _aggregate_group(
    key: tuple[str, str, str, str, str, str], seed_values: Sequence[tuple[int, float]]
) -> AggregateMetricRow:
    ordered = sorted(seed_values)
    seeds = [seed for seed, _ in ordered]
    if len(seeds) != len(set(seeds)):
        raise AnalysisValidationError(f"duplicate seed values in aggregate group {key}")
    values = [value for _, value in ordered]
    scenario, variant, scheduler, metric, dimension, unit = key
    return AggregateMetricRow(
        scenario_id=scenario,
        variant_id=variant,
        scheduler_name=scheduler,
        metric=metric,
        dimension=dimension,
        unit=unit,
        run_count=len(values),
        mean=statistics.fmean(values),
        sample_std=statistics.stdev(values) if len(values) > 1 else 0.0,
        seed_values=tuple(SeedMetricValue(seed=seed, value=value) for seed, value in ordered),
    )


def _atomic_write_csv(path: Path, rows: Sequence[DomainModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(type(rows[0]).model_fields)
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            payload = row.model_dump(mode="json")
            if "seed_values" in payload:
                payload["seed_values"] = json.dumps(payload["seed_values"], separators=(",", ":"))
            writer.writerow(payload)
        content = stream.getvalue()
    else:
        content = ""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains invalid constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key: {key!r}")
        result[key] = value
    return result
