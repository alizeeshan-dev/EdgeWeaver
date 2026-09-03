from pathlib import Path

import pytest

from edgeweaver.config import LoadedConfiguration, load_phase1_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def phase1_configuration() -> LoadedConfiguration:
    return load_phase1_configuration(
        devices_path=PROJECT_ROOT / "configs" / "devices.yaml",
        network_path=PROJECT_ROOT / "configs" / "network.yaml",
        model_profiles_path=PROJECT_ROOT / "artifacts" / "profiles" / "synthetic_fixture.yaml",
        scenario_path=PROJECT_ROOT / "configs" / "phase1_fixture.yaml",
    )


@pytest.fixture
def mini_uci_har_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "UCI HAR Dataset"
    (dataset_root / "train").mkdir(parents=True)
    (dataset_root / "test").mkdir()
    (dataset_root / "features.txt").write_text(
        "1 feature-a\n2 duplicated-name\n3 duplicated-name\n", encoding="utf-8"
    )
    (dataset_root / "activity_labels.txt").write_text("1 WALKING\n2 SITTING\n", encoding="utf-8")
    (dataset_root / "train" / "X_train.txt").write_text(
        "1 10 100\n2 20 200\n3 30 300\n4 40 400\n", encoding="utf-8"
    )
    (dataset_root / "train" / "y_train.txt").write_text("1\n2\n1\n2\n", encoding="utf-8")
    (dataset_root / "train" / "subject_train.txt").write_text("1\n1\n2\n2\n", encoding="utf-8")
    (dataset_root / "test" / "X_test.txt").write_text("5 50 500\n6 60 600\n", encoding="utf-8")
    (dataset_root / "test" / "y_test.txt").write_text("1\n2\n", encoding="utf-8")
    (dataset_root / "test" / "subject_test.txt").write_text("3\n3\n", encoding="utf-8")
    return dataset_root
