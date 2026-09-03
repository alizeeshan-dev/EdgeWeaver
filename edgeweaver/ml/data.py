"""Official UCI HAR download, extraction, validation, and loading."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
)
DATASET_DIRECTORY_NAME = "UCI HAR Dataset"
OFFICIAL_FEATURE_COUNT = 561
OFFICIAL_TRAIN_ROWS = 7_352
OFFICIAL_TEST_ROWS = 2_947
OFFICIAL_ACTIVITIES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
REQUIRED_RELATIVE_PATHS = (
    Path("features.txt"),
    Path("activity_labels.txt"),
    Path("train/X_train.txt"),
    Path("train/y_train.txt"),
    Path("train/subject_train.txt"),
    Path("test/X_test.txt"),
    Path("test/y_test.txt"),
    Path("test/subject_test.txt"),
)


class DatasetValidationError(ValueError):
    """Raised when an extracted UCI HAR dataset is incomplete or malformed."""


@dataclass(frozen=True)
class DatasetSplit:
    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    subjects: NDArray[np.int64]


@dataclass(frozen=True)
class UCIHARDataset:
    root: Path
    train: DatasetSplit
    test: DatasetSplit
    feature_names: tuple[str, ...]
    activity_labels: dict[int, str]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


def validate_required_files(dataset_root: Path) -> None:
    """Validate that all modeling inputs exist as regular files."""

    missing = [str(path) for path in REQUIRED_RELATIVE_PATHS if not (dataset_root / path).is_file()]
    if missing:
        raise DatasetValidationError(
            f"UCI HAR dataset at {dataset_root} is incomplete; missing: {', '.join(missing)}"
        )


def download_uci_har_archive(
    destination_directory: Path,
    *,
    url: str = UCI_HAR_URL,
) -> Path:
    """Download the official archive once, without extracting it."""

    destination_directory.mkdir(parents=True, exist_ok=True)
    archive_path = destination_directory / f"{DATASET_DIRECTORY_NAME}.zip"
    if archive_path.is_file() and archive_path.stat().st_size > 0:
        return archive_path

    partial_path = archive_path.with_suffix(".zip.part")
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,  # noqa: S310 - caller defaults to fixed official URL
            partial_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        if partial_path.stat().st_size == 0:
            raise DatasetValidationError("downloaded UCI HAR archive is empty")
        partial_path.replace(archive_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    metadata = {
        "dataset": "UCI Human Activity Recognition Using Smartphones",
        "source_url": url,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive_size_bytes": archive_path.stat().st_size,
    }
    (destination_directory / "uci_har_download.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return archive_path


def extract_uci_har_archive(archive_path: Path, destination_directory: Path) -> Path:
    """Safely extract an existing official archive and return its dataset root."""

    dataset_root = destination_directory / DATASET_DIRECTORY_NAME
    if dataset_root.exists():
        validate_required_files(dataset_root)
        return dataset_root

    destination_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".uci-har-", dir=destination_directory) as temp_name:
        temporary_root = Path(temp_name).resolve()
        _safe_extract_zip(archive_path, temporary_root)

        extracted_dataset = temporary_root / DATASET_DIRECTORY_NAME
        nested_archive = temporary_root / f"{DATASET_DIRECTORY_NAME}.zip"
        if not extracted_dataset.exists() and nested_archive.is_file():
            _safe_extract_zip(nested_archive, temporary_root)
        validate_required_files(extracted_dataset)
        shutil.move(str(extracted_dataset), dataset_root)
    return dataset_root


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                extracted_path = (destination / member.filename).resolve()
                if not extracted_path.is_relative_to(destination):
                    raise DatasetValidationError(
                        f"unsafe path in UCI HAR archive: {member.filename}"
                    )
            archive.extractall(destination)
    except zipfile.BadZipFile as error:
        raise DatasetValidationError(f"invalid UCI HAR zip archive: {archive_path}") from error


def obtain_uci_har(destination_directory: Path) -> Path:
    """Reuse a valid extracted dataset or download and extract the official archive."""

    dataset_root = destination_directory / DATASET_DIRECTORY_NAME
    if dataset_root.exists():
        validate_required_files(dataset_root)
        return dataset_root
    archive_path = download_uci_har_archive(destination_directory)
    return extract_uci_har_archive(archive_path, destination_directory)


def _parse_indexed_names(path: Path) -> tuple[tuple[int, str], ...]:
    parsed: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1]:
            raise DatasetValidationError(f"malformed indexed name at {path}:{line_number}")
        try:
            identifier = int(parts[0])
        except ValueError as error:
            raise DatasetValidationError(f"invalid integer ID at {path}:{line_number}") from error
        parsed.append((identifier, parts[1]))
    if not parsed:
        raise DatasetValidationError(f"indexed-name file is empty: {path}")
    identifiers = [identifier for identifier, _ in parsed]
    if identifiers != list(range(1, len(parsed) + 1)):
        raise DatasetValidationError(f"IDs in {path} must be sequential starting at 1")
    return tuple(parsed)


def _load_numeric(path: Path, *, two_dimensional: bool) -> NDArray[np.float64]:
    try:
        values = np.loadtxt(path, dtype=np.float64, ndmin=2 if two_dimensional else 1)
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"could not parse numeric data in {path}: {error}") from error
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise DatasetValidationError(f"numeric data contains NaN or infinity: {path}")
    return array


def _load_split(dataset_root: Path, split_name: str, feature_count: int) -> DatasetSplit:
    split_root = dataset_root / split_name
    features = _load_numeric(split_root / f"X_{split_name}.txt", two_dimensional=True)
    raw_labels = _load_numeric(split_root / f"y_{split_name}.txt", two_dimensional=False)
    raw_subjects = _load_numeric(split_root / f"subject_{split_name}.txt", two_dimensional=False)
    if features.shape[1] != feature_count:
        raise DatasetValidationError(
            f"{split_name} feature count is {features.shape[1]}; expected {feature_count}"
        )
    if features.shape[0] != raw_labels.size or features.shape[0] != raw_subjects.size:
        raise DatasetValidationError(
            f"{split_name} row alignment mismatch: X={features.shape[0]}, "
            f"y={raw_labels.size}, subjects={raw_subjects.size}"
        )
    if not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise DatasetValidationError(f"{split_name} labels must be integers")
    if not np.equal(raw_subjects, np.floor(raw_subjects)).all():
        raise DatasetValidationError(f"{split_name} subject IDs must be integers")
    return DatasetSplit(
        features=features,
        labels=raw_labels.astype(np.int64),
        subjects=raw_subjects.astype(np.int64),
    )


def load_uci_har(dataset_root: Path, *, strict_official: bool = True) -> UCIHARDataset:
    """Load and validate the preserved predefined UCI HAR train/test split."""

    validate_required_files(dataset_root)
    indexed_features = _parse_indexed_names(dataset_root / "features.txt")
    indexed_activities = _parse_indexed_names(dataset_root / "activity_labels.txt")
    feature_names = tuple(name for _, name in indexed_features)
    activity_labels = dict(indexed_activities)
    train = _load_split(dataset_root, "train", len(feature_names))
    test = _load_split(dataset_root, "test", len(feature_names))

    if strict_official:
        if len(feature_names) != OFFICIAL_FEATURE_COUNT:
            raise DatasetValidationError(
                f"official UCI HAR data must have {OFFICIAL_FEATURE_COUNT} features; "
                f"found {len(feature_names)}"
            )
        if train.features.shape[0] != OFFICIAL_TRAIN_ROWS:
            raise DatasetValidationError(
                f"official train split must have {OFFICIAL_TRAIN_ROWS} rows; "
                f"found {train.features.shape[0]}"
            )
        if test.features.shape[0] != OFFICIAL_TEST_ROWS:
            raise DatasetValidationError(
                f"official test split must have {OFFICIAL_TEST_ROWS} rows; "
                f"found {test.features.shape[0]}"
            )
        if activity_labels != OFFICIAL_ACTIVITIES:
            raise DatasetValidationError(
                "activity_labels.txt does not match official UCI HAR labels"
            )

    known_labels = set(activity_labels)
    for split_name, split in (("train", train), ("test", test)):
        unknown = set(np.unique(split.labels)) - known_labels
        if unknown:
            raise DatasetValidationError(
                f"{split_name} contains labels absent from activity_labels.txt: {sorted(unknown)}"
            )

    return UCIHARDataset(
        root=dataset_root,
        train=train,
        test=test,
        feature_names=feature_names,
        activity_labels=activity_labels,
    )
