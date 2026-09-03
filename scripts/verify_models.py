"""Reload trained artifacts and predict a small held-out UCI HAR sample."""

import argparse
from pathlib import Path

from edgeweaver.ml.data import DATASET_DIRECTORY_NAME, load_uci_har
from edgeweaver.ml.train import verify_trained_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / DATASET_DIRECTORY_NAME,
    )
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    dataset = load_uci_har(args.dataset_root)
    predictions = verify_trained_artifacts(dataset, args.models_root, sample_count=args.samples)
    for model_id, values in predictions.items():
        print(f"{model_id}: predicted {len(values)} held-out samples")


if __name__ == "__main__":
    main()
