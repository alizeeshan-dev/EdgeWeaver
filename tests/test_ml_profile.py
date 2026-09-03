import math

import pytest
from pydantic import ValidationError

from edgeweaver.domain import ModelProfile, ModelRole
from edgeweaver.ml.profile import calculate_latency_statistics, select_profile_indices


def test_latency_statistics_use_population_std_and_linear_p95() -> None:
    summary = calculate_latency_statistics([1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000])

    assert summary.mean_ms == 3.0
    assert summary.median_ms == 3.0
    assert summary.standard_deviation_ms == pytest.approx(math.sqrt(2.0))
    assert summary.p95_ms == pytest.approx(4.8)


def test_profile_sample_selection_is_seeded() -> None:
    assert select_profile_indices(100, 10, 42) == select_profile_indices(100, 10, 42)
    assert select_profile_indices(100, 10, 42) != select_profile_indices(100, 10, 43)


def test_measured_profile_requires_complete_real_measurements() -> None:
    with pytest.raises(ValidationError, match="measured model profile is missing"):
        ModelProfile(
            model_id="incomplete",
            display_name="Incomplete",
            accuracy=0.9,
            local_latency_ms_mean=0.1,
            profile_kind="measured",
        )

    profile = ModelProfile(
        model_id="logistic-regression-v1",
        display_name="Light Model",
        accuracy=0.9,
        macro_f1=0.9,
        model_size_bytes=100,
        local_latency_ms_mean=0.1,
        local_latency_ms_median=0.09,
        local_latency_ms_std=0.01,
        local_latency_ms_p95=0.12,
        profile_kind="measured",
        computational_role=ModelRole.LIGHT,
        profiling_prediction_count=500,
        profiling_seed=7,
        raw_latency_observations_file="timings.json",
    )
    assert profile.profile_kind == "measured"
