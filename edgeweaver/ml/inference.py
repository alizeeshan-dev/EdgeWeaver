"""Reusable cached inference service for simulation prediction correctness."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from edgeweaver.ml.artifacts import LoadedModelArtifacts, load_model_artifacts, predict_samples
from edgeweaver.ml.data import UCIHARDataset
from edgeweaver.workloads import resolve_test_sample


class ModelPredictionService:
    """Load each trained model once and cache deterministic held-out predictions."""

    def __init__(
        self,
        dataset: UCIHARDataset,
        artifacts: dict[str, LoadedModelArtifacts],
    ) -> None:
        self._dataset = dataset
        self._artifacts = artifacts
        self._prediction_cache: dict[tuple[str, int], int] = {}

    @classmethod
    def from_model_directories(
        cls,
        dataset: UCIHARDataset,
        models_root: Path,
        model_ids: Sequence[str],
    ) -> ModelPredictionService:
        artifacts = {
            model_id: load_model_artifacts(models_root / model_id) for model_id in model_ids
        }
        return cls(dataset, artifacts)

    def predict(self, model_id: str, feature_vector_id: int) -> int:
        key = (model_id, feature_vector_id)
        if key not in self._prediction_cache:
            try:
                artifacts = self._artifacts[model_id]
            except KeyError as error:
                raise ValueError(f"no loaded model artifact for {model_id}") from error
            sample, _ = resolve_test_sample(self._dataset, feature_vector_id)
            self._prediction_cache[key] = int(predict_samples(artifacts, sample)[0])
        return self._prediction_cache[key]
