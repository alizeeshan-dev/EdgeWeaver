"""Local single-request latency profiling for trained EdgeWeaver artifacts."""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from edgeweaver.domain import DomainModel, ModelProfile
from edgeweaver.ml.artifacts import (
    LoadedModelArtifacts,
    load_model_artifacts,
    predict_samples,
    write_domain_json,
)
from edgeweaver.ml.data import UCIHARDataset
from edgeweaver.ml.train import MODEL_DEFINITIONS

DEFAULT_PROFILING_SEED = 2027
DEFAULT_TIMED_PREDICTIONS = 500
DEFAULT_WARMUP_PREDICTIONS = 20


class LatencyStatistics(DomainModel):
    mean_ms: float = Field(gt=0.0)
    median_ms: float = Field(gt=0.0)
    standard_deviation_ms: float = Field(ge=0.0)
    p95_ms: float = Field(gt=0.0)


class RawTimingObservations(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: str
    profiling_seed: int = Field(ge=0)
    timer: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    percentile_method: Literal["linear"] = "linear"
    standard_deviation_ddof: Literal[0] = 0
    sample_indices: list[int] = Field(min_length=1)
    latency_ns: list[int] = Field(min_length=1)


class ProfilingEnvironment(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    profiled_at_utc: datetime
    measurement_scope: Literal["this local computer only"] = "this local computer only"
    operating_system: str
    platform_description: str
    system_release: str
    system_version: str
    machine: str
    architecture: str
    processor: str
    python_version: str
    python_implementation: str
    library_versions: dict[str, str]
    timer: Literal["time.perf_counter_ns"] = "time.perf_counter_ns"
    methodology: str
    warmup_predictions_per_model: int = Field(ge=1)
    timed_predictions_per_model: int = Field(ge=500)
    profiling_seed: int = Field(ge=0)
    percentile_method: Literal["linear"] = "linear"
    standard_deviation_ddof: Literal[0] = 0


class ProfilingRunManifest(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    profiled_at_utc: datetime
    environment_metadata_file: str
    profile_files: list[str] = Field(min_length=1)


def calculate_latency_statistics(latency_ns: Sequence[int]) -> LatencyStatistics:
    """Summarize complete-run observations using population std and linear P95."""

    if not latency_ns:
        raise ValueError("at least one latency observation is required")
    if any(isinstance(value, bool) or value <= 0 for value in latency_ns):
        raise ValueError("latency observations must be positive integer nanoseconds")
    latency_ms = np.asarray(latency_ns, dtype=np.float64) / 1_000_000.0
    if not np.isfinite(latency_ms).all():
        raise ValueError("latency observations must be finite")
    return LatencyStatistics(
        mean_ms=statistics.fmean(latency_ms),
        median_ms=statistics.median(latency_ms),
        standard_deviation_ms=statistics.pstdev(latency_ms),
        p95_ms=float(np.percentile(latency_ms, 95, method="linear")),
    )


def select_profile_indices(sample_count: int, prediction_count: int, seed: int) -> list[int]:
    if sample_count < 1 or prediction_count < 1:
        raise ValueError("sample_count and prediction_count must be positive")
    generator = np.random.default_rng(seed)
    return [int(index) for index in generator.integers(0, sample_count, size=prediction_count)]


def _environment_metadata(
    *,
    profiled_at: datetime,
    timed_predictions: int,
    warmup_predictions: int,
    seed: int,
) -> ProfilingEnvironment:
    processor = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    return ProfilingEnvironment(
        profiled_at_utc=profiled_at,
        operating_system=platform.system(),
        platform_description=platform.platform(),
        system_release=platform.release(),
        system_version=platform.version(),
        machine=platform.machine(),
        architecture=platform.architecture()[0],
        processor=processor,
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        library_versions={
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib")
        },
        methodology=(
            "Artifacts and dataset are loaded before timing; each observation includes saved "
            "preprocessor.transform plus estimator.predict for exactly one test sample. Warm-ups "
            "are untimed. Timer overhead is not subtracted."
        ),
        warmup_predictions_per_model=warmup_predictions,
        timed_predictions_per_model=timed_predictions,
        profiling_seed=seed,
    )


def profile_models(
    dataset: UCIHARDataset,
    models_root: Path,
    profiles_root: Path,
    *,
    timed_predictions: int = DEFAULT_TIMED_PREDICTIONS,
    warmup_predictions: int = DEFAULT_WARMUP_PREDICTIONS,
    seed: int = DEFAULT_PROFILING_SEED,
) -> dict[str, ModelProfile]:
    """Profile all artifacts with the same reproducibly sampled single-row inputs."""

    if timed_predictions < 500:
        raise ValueError("real model profiling requires at least 500 timed predictions per model")
    if warmup_predictions < 1:
        raise ValueError("at least one warm-up prediction is required")
    profiles_root.mkdir(parents=True, exist_ok=True)
    profiled_at = datetime.now(UTC)

    loaded: dict[str, LoadedModelArtifacts] = {
        definition.model_id: load_model_artifacts(models_root / definition.model_id)
        for definition in MODEL_DEFINITIONS
    }
    sample_indices = select_profile_indices(dataset.test.features.shape[0], timed_predictions, seed)
    warmup_indices = [index % dataset.test.features.shape[0] for index in range(warmup_predictions)]
    profiles: dict[str, ModelProfile] = {}
    profile_files: list[str] = []

    for definition in MODEL_DEFINITIONS:
        artifacts = loaded[definition.model_id]
        for index in warmup_indices:
            predict_samples(artifacts, dataset.test.features[index : index + 1])

        observations_ns: list[int] = []
        for index in sample_indices:
            sample = dataset.test.features[index : index + 1]
            started_ns = time.perf_counter_ns()
            predict_samples(artifacts, sample)
            observations_ns.append(time.perf_counter_ns() - started_ns)

        statistics_ = calculate_latency_statistics(observations_ns)
        raw_filename = f"{definition.model_id}.timings.json"
        raw = RawTimingObservations(
            model_id=definition.model_id,
            profiling_seed=seed,
            sample_indices=sample_indices,
            latency_ns=observations_ns,
        )
        write_domain_json(profiles_root / raw_filename, raw)

        metadata = artifacts.metadata
        profile = ModelProfile(
            model_id=metadata.model_id,
            display_name=metadata.display_name,
            accuracy=metadata.accuracy,
            macro_f1=metadata.macro_f1,
            model_size_bytes=metadata.model_size_bytes,
            local_latency_ms_mean=statistics_.mean_ms,
            local_latency_ms_median=statistics_.median_ms,
            local_latency_ms_std=statistics_.standard_deviation_ms,
            local_latency_ms_p95=statistics_.p95_ms,
            profile_kind="measured",
            computational_role=metadata.computational_role,
            profiling_prediction_count=timed_predictions,
            profiling_seed=seed,
            raw_latency_observations_file=raw_filename,
            notes=(
                "Measured on this local computer only; other EdgeWeaver devices remain simulated "
                "through separate speed multipliers."
            ),
        )
        profile_filename = f"{definition.model_id}.json"
        write_domain_json(profiles_root / profile_filename, profile)
        profiles[definition.model_id] = profile
        profile_files.append(profile_filename)

    environment = _environment_metadata(
        profiled_at=profiled_at,
        timed_predictions=timed_predictions,
        warmup_predictions=warmup_predictions,
        seed=seed,
    )
    environment_filename = "profiling_environment.json"
    write_domain_json(profiles_root / environment_filename, environment)
    manifest = ProfilingRunManifest(
        profiled_at_utc=profiled_at,
        environment_metadata_file=environment_filename,
        profile_files=profile_files,
    )
    write_domain_json(profiles_root / "profile_manifest.json", manifest)
    return profiles


def load_measured_profiles(profiles_root: Path) -> dict[str, ModelProfile]:
    manifest = ProfilingRunManifest.model_validate_json(
        (profiles_root / "profile_manifest.json").read_text(encoding="utf-8")
    )
    profiles = [
        ModelProfile.model_validate_json((profiles_root / filename).read_text(encoding="utf-8"))
        for filename in manifest.profile_files
    ]
    return {profile.model_id: profile for profile in profiles}
