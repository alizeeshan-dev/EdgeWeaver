"""Stable metadata and reload/predict path for trained model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field

from edgeweaver.domain import DomainModel, JsonScalar, ModelRole

HyperparameterValue = JsonScalar | list[int]


class ModelArtifactMetadata(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    model_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    computational_role: ModelRole
    random_seed: int = Field(ge=0)
    estimator_class: str = Field(min_length=1)
    preprocessor_class: str = Field(min_length=1)
    hyperparameters: dict[str, HyperparameterValue]
    feature_count: int = Field(ge=1)
    class_labels: list[int] = Field(min_length=1)
    train_sample_count: int = Field(ge=1)
    test_sample_count: int = Field(ge=1)
    accuracy: float = Field(ge=0.0, le=1.0)
    macro_f1: float = Field(ge=0.0, le=1.0)
    model_artifact: str = "model.joblib"
    preprocessor_artifact: str = "preprocessor.joblib"
    model_size_bytes: int = Field(ge=1)
    preprocessor_size_bytes: int = Field(ge=1)
    dataset_name: Literal["UCI Human Activity Recognition Using Smartphones"] = (
        "UCI Human Activity Recognition Using Smartphones"
    )
    dataset_split: Literal["official predefined train/test split"] = (
        "official predefined train/test split"
    )
    sklearn_version: str = Field(min_length=1)
    trained_at_utc: datetime
    training_iterations: int | None = Field(default=None, ge=1)
    final_training_loss: float | None = Field(default=None, ge=0.0)


class TrainingRunManifest(DomainModel):
    schema_version: Literal["1.0"] = "1.0"
    trained_at_utc: datetime
    random_seed: int = Field(ge=0)
    dataset_root: str
    model_directories: list[str] = Field(min_length=1)


@dataclass(frozen=True)
class LoadedModelArtifacts:
    metadata: ModelArtifactMetadata
    preprocessor: Any
    estimator: Any


def _artifact_path(model_directory: Path, filename: str) -> Path:
    model_root = model_directory.resolve()
    path = (model_root / filename).resolve()
    if not path.is_relative_to(model_root):
        raise ValueError(f"artifact path escapes model directory: {filename}")
    return path


def load_model_metadata(model_directory: Path) -> ModelArtifactMetadata:
    metadata_path = model_directory / "metadata.json"
    try:
        return ModelArtifactMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"model metadata not found: {metadata_path}") from error


def load_model_artifacts(model_directory: Path) -> LoadedModelArtifacts:
    """Load the exact fitted preprocessor and estimator referenced by metadata."""

    metadata = load_model_metadata(model_directory)
    model_path = _artifact_path(model_directory, metadata.model_artifact)
    preprocessor_path = _artifact_path(model_directory, metadata.preprocessor_artifact)
    if not model_path.is_file():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"preprocessor artifact not found: {preprocessor_path}")
    if model_path.stat().st_size != metadata.model_size_bytes:
        raise ValueError(f"model artifact size does not match metadata: {model_path}")
    if preprocessor_path.stat().st_size != metadata.preprocessor_size_bytes:
        raise ValueError(f"preprocessor artifact size does not match metadata: {preprocessor_path}")
    return LoadedModelArtifacts(
        metadata=metadata,
        preprocessor=joblib.load(preprocessor_path),
        estimator=joblib.load(model_path),
    )


def predict_samples(
    artifacts: LoadedModelArtifacts,
    features: ArrayLike,
) -> NDArray[np.int64]:
    """Apply the saved training preprocessor and predict one or more samples."""

    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != artifacts.metadata.feature_count:
        raise ValueError(
            f"features must have shape (n, {artifacts.metadata.feature_count}); "
            f"received {array.shape}"
        )
    transformed = artifacts.preprocessor.transform(array)
    return np.asarray(artifacts.estimator.predict(transformed), dtype=np.int64)


def load_training_manifest(path: Path) -> TrainingRunManifest:
    return TrainingRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_domain_json(path: Path, value: DomainModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
