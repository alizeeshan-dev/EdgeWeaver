from pathlib import Path

import pytest

from edgeweaver.ml.data import DatasetValidationError, load_uci_har, validate_required_files


def test_controlled_uci_layout_preserves_splits_and_alignment(
    mini_uci_har_root: Path,
) -> None:
    dataset = load_uci_har(mini_uci_har_root, strict_official=False)

    assert dataset.train.features.shape == (4, 3)
    assert dataset.test.features.shape == (2, 3)
    assert dataset.train.labels.tolist() == [1, 2, 1, 2]
    assert dataset.test.subjects.tolist() == [3, 3]
    assert dataset.feature_names == ("feature-a", "duplicated-name", "duplicated-name")
    assert dataset.activity_labels == {1: "WALKING", 2: "SITTING"}


def test_missing_required_file_has_clear_error(mini_uci_har_root: Path) -> None:
    missing_path = mini_uci_har_root / "test" / "y_test.txt"
    missing_path.unlink()

    with pytest.raises(DatasetValidationError, match=r"test\\y_test.txt|test/y_test.txt"):
        validate_required_files(mini_uci_har_root)


def test_feature_label_alignment_mismatch_is_rejected(mini_uci_har_root: Path) -> None:
    (mini_uci_har_root / "test" / "y_test.txt").write_text("1\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="row alignment mismatch"):
        load_uci_har(mini_uci_har_root, strict_official=False)
