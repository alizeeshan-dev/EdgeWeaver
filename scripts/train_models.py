"""Train and evaluate EdgeWeaver's three fixed model variants."""

import argparse
from pathlib import Path

from edgeweaver.ml.artifacts import load_model_metadata
from edgeweaver.ml.data import DATASET_DIRECTORY_NAME, load_uci_har
from edgeweaver.ml.train import DEFAULT_TRAINING_SEED, train_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / DATASET_DIRECTORY_NAME,
    )
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    args = parser.parse_args()

    dataset = load_uci_har(args.dataset_root)
    manifest = train_models(dataset, args.models_root, seed=args.seed)
    for directory_name in manifest.model_directories:
        metadata = load_model_metadata(args.models_root / directory_name)
        print(
            f"{metadata.model_id} ({metadata.computational_role}): "
            f"accuracy={metadata.accuracy:.6f}, macro_f1={metadata.macro_f1:.6f}"
        )
    print(f"Saved training manifest to {args.models_root / 'training_manifest.json'}")


if __name__ == "__main__":
    main()
