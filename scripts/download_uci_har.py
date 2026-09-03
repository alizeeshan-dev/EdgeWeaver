"""Download and extract the official UCI HAR dataset if it is not already present."""

import argparse
from pathlib import Path

from edgeweaver.ml.data import load_uci_har, obtain_uci_har

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()
    dataset_root = obtain_uci_har(args.raw_data_root)
    dataset = load_uci_har(dataset_root)
    print(
        f"UCI HAR ready at {dataset.root}: train={dataset.train.features.shape}, "
        f"test={dataset.test.features.shape}"
    )


if __name__ == "__main__":
    main()
