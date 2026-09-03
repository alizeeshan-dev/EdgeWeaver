from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from edgeweaver.domain import ModelRole
from edgeweaver.ml.artifacts import (
    ModelArtifactMetadata,
    load_model_artifacts,
    load_model_metadata,
    predict_samples,
    write_domain_json,
)
from edgeweaver.ml.train import create_estimator, create_preprocessor


def test_preprocessing_scales_only_roles_that_require_it() -> None:
    features = np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    scaler = create_preprocessor(ModelRole.LIGHT)
    transformed = scaler.fit_transform(features)
    identity = create_preprocessor(ModelRole.BALANCED)

    assert isinstance(scaler, StandardScaler)
    assert np.allclose(transformed.mean(axis=0), 0.0)
    assert isinstance(identity, FunctionTransformer)
    assert np.array_equal(identity.fit_transform(features), features)


def test_model_factories_use_seeded_expected_families() -> None:
    light = create_estimator(ModelRole.LIGHT, seed=17)
    balanced = create_estimator(ModelRole.BALANCED, seed=17)
    heavy = create_estimator(ModelRole.HEAVY, seed=17)

    assert isinstance(light, LogisticRegression)
    assert light.random_state == 17
    assert light.max_iter == 2000
    assert isinstance(balanced, RandomForestClassifier)
    assert balanced.random_state == 17
    assert balanced.n_estimators == 200
    assert balanced.n_jobs == 1
    assert isinstance(heavy, MLPClassifier)
    assert heavy.random_state == 17
    assert heavy.hidden_layer_sizes == (128, 64)
    assert heavy.early_stopping


def test_saved_artifacts_reload_and_predict_identically(tmp_path: Path) -> None:
    features = np.asarray([[0.0, 0.0], [0.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
    labels = np.asarray([1, 1, 2, 2])
    preprocessor = StandardScaler().fit(features)
    estimator = LogisticRegression(random_state=5).fit(preprocessor.transform(features), labels)
    expected = estimator.predict(preprocessor.transform(features))

    model_directory = tmp_path / "logistic-regression-v1"
    model_directory.mkdir()
    model_path = model_directory / "model.joblib"
    preprocessor_path = model_directory / "preprocessor.joblib"
    joblib.dump(estimator, model_path, compress=3)
    joblib.dump(preprocessor, preprocessor_path, compress=3)
    metadata = ModelArtifactMetadata(
        model_id="logistic-regression-v1",
        display_name="Light Model",
        computational_role=ModelRole.LIGHT,
        random_seed=5,
        estimator_class="LogisticRegression",
        preprocessor_class="StandardScaler",
        hyperparameters={"random_state": 5},
        feature_count=2,
        class_labels=[1, 2],
        train_sample_count=4,
        test_sample_count=2,
        accuracy=1.0,
        macro_f1=1.0,
        model_size_bytes=model_path.stat().st_size,
        preprocessor_size_bytes=preprocessor_path.stat().st_size,
        sklearn_version="test-version",
        trained_at_utc=datetime.now(UTC),
    )
    write_domain_json(model_directory / "metadata.json", metadata)

    assert load_model_metadata(model_directory) == metadata
    reloaded = load_model_artifacts(model_directory)
    actual = predict_samples(reloaded, features)
    assert np.array_equal(actual, expected)
    assert predict_samples(reloaded, features[0]).shape == (1,)
