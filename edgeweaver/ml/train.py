"""Deterministic preprocessing, training, evaluation, and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from edgeweaver.domain import ModelRole
from edgeweaver.ml.artifacts import (
    HyperparameterValue,
    ModelArtifactMetadata,
    TrainingRunManifest,
    load_model_artifacts,
    predict_samples,
    write_domain_json,
)
from edgeweaver.ml.data import UCIHARDataset

DEFAULT_TRAINING_SEED = 2027


@dataclass(frozen=True)
class ModelDefinition:
    model_id: str
    display_name: str
    role: ModelRole
    hyperparameters: dict[str, HyperparameterValue]


MODEL_DEFINITIONS = (
    ModelDefinition(
        model_id="logistic-regression-v1",
        display_name="Light Model",
        role=ModelRole.LIGHT,
        hyperparameters={
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 2000,
            "tol": 0.0001,
        },
    ),
    ModelDefinition(
        model_id="random-forest-v1",
        display_name="Balanced Model",
        role=ModelRole.BALANCED,
        hyperparameters={
            "n_estimators": 200,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_jobs": 1,
        },
    ),
    ModelDefinition(
        model_id="mlp-v1",
        display_name="Heavy Model",
        role=ModelRole.HEAVY,
        hyperparameters={
            "hidden_layer_sizes": [128, 64],
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "batch_size": 128,
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
        },
    ),
)


def get_model_definition(role: ModelRole) -> ModelDefinition:
    return next(definition for definition in MODEL_DEFINITIONS if definition.role == role)


def create_preprocessor(role: ModelRole) -> TransformerMixin:
    """Create scaling for linear/neural models and explicit identity for the forest."""

    if role in {ModelRole.LIGHT, ModelRole.HEAVY}:
        return StandardScaler()
    if role is ModelRole.BALANCED:
        return FunctionTransformer(validate=True, feature_names_out="one-to-one")
    raise ValueError(f"unsupported model role: {role}")


def create_estimator(role: ModelRole, *, seed: int) -> ClassifierMixin:
    """Create one of the three fixed, seeded scikit-learn model families."""

    if role is ModelRole.LIGHT:
        return LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
        )
    if role is ModelRole.BALANCED:
        return RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=1,
        )
    if role is ModelRole.HEAVY:
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
    raise ValueError(f"unsupported model role: {role}")


def _optional_int_value(estimator: ClassifierMixin, name: str) -> int | None:
    value = getattr(estimator, name, None)
    if isinstance(value, np.generic):
        value = value.item()
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float_value(estimator: ClassifierMixin, name: str) -> float | None:
    value = getattr(estimator, name, None)
    if isinstance(value, np.generic):
        value = value.item()
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def train_models(
    dataset: UCIHARDataset,
    output_root: Path,
    *,
    seed: int = DEFAULT_TRAINING_SEED,
) -> TrainingRunManifest:
    """Train and evaluate all three fixed model definitions on the official split."""

    output_root.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now(UTC)
    model_directories: list[str] = []

    for definition in MODEL_DEFINITIONS:
        preprocessor = create_preprocessor(definition.role)
        estimator = create_estimator(definition.role, seed=seed)
        transformed_train = preprocessor.fit_transform(dataset.train.features)
        transformed_test = preprocessor.transform(dataset.test.features)
        estimator.fit(transformed_train, dataset.train.labels)
        predictions = np.asarray(estimator.predict(transformed_test), dtype=np.int64)
        accuracy = float(accuracy_score(dataset.test.labels, predictions))
        macro_f1 = float(
            f1_score(dataset.test.labels, predictions, average="macro", zero_division=0)
        )

        model_directory = output_root / definition.model_id
        model_directory.mkdir(parents=True, exist_ok=True)
        model_path = model_directory / "model.joblib"
        preprocessor_path = model_directory / "preprocessor.joblib"
        joblib.dump(estimator, model_path, compress=3)
        joblib.dump(preprocessor, preprocessor_path, compress=3)

        metadata = ModelArtifactMetadata(
            model_id=definition.model_id,
            display_name=definition.display_name,
            computational_role=definition.role,
            random_seed=seed,
            estimator_class=type(estimator).__name__,
            preprocessor_class=type(preprocessor).__name__,
            hyperparameters=definition.hyperparameters,
            feature_count=dataset.feature_count,
            class_labels=sorted(int(label) for label in np.unique(dataset.train.labels)),
            train_sample_count=dataset.train.features.shape[0],
            test_sample_count=dataset.test.features.shape[0],
            accuracy=accuracy,
            macro_f1=macro_f1,
            model_size_bytes=model_path.stat().st_size,
            preprocessor_size_bytes=preprocessor_path.stat().st_size,
            sklearn_version=sklearn.__version__,
            trained_at_utc=trained_at,
            training_iterations=_optional_int_value(estimator, "n_iter_"),
            final_training_loss=_optional_float_value(estimator, "loss_"),
        )
        write_domain_json(model_directory / "metadata.json", metadata)

        reloaded = load_model_artifacts(model_directory)
        reloaded_predictions = predict_samples(reloaded, dataset.test.features[:16])
        if not np.array_equal(reloaded_predictions, predictions[:16]):
            raise RuntimeError(f"reloaded predictions differ for {definition.model_id}")
        model_directories.append(definition.model_id)

    manifest = TrainingRunManifest(
        trained_at_utc=trained_at,
        random_seed=seed,
        dataset_root=str(dataset.root),
        model_directories=model_directories,
    )
    write_domain_json(output_root / "training_manifest.json", manifest)
    return manifest


def verify_trained_artifacts(
    dataset: UCIHARDataset,
    models_root: Path,
    *,
    sample_count: int = 32,
) -> dict[str, list[int]]:
    """Reload every trained artifact and predict held-out samples as an explicit smoke path."""

    if sample_count < 1 or sample_count > dataset.test.features.shape[0]:
        raise ValueError("sample_count must fit within the held-out test split")
    predictions: dict[str, list[int]] = {}
    for definition in MODEL_DEFINITIONS:
        artifacts = load_model_artifacts(models_root / definition.model_id)
        output = predict_samples(artifacts, dataset.test.features[:sample_count])
        predictions[definition.model_id] = [int(value) for value in output]
    return predictions
