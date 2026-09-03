"""Measure local single-request latency for all trained EdgeWeaver models."""

import argparse
from pathlib import Path

from edgeweaver.ml.data import DATASET_DIRECTORY_NAME, load_uci_har
from edgeweaver.ml.profile import (
    DEFAULT_PROFILING_SEED,
    DEFAULT_TIMED_PREDICTIONS,
    DEFAULT_WARMUP_PREDICTIONS,
    profile_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / DATASET_DIRECTORY_NAME,
    )
    parser.add_argument("--models-root", type=Path, default=PROJECT_ROOT / "artifacts" / "models")
    parser.add_argument(
        "--profiles-root", type=Path, default=PROJECT_ROOT / "artifacts" / "profiles"
    )
    parser.add_argument("--predictions", type=int, default=DEFAULT_TIMED_PREDICTIONS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUP_PREDICTIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_PROFILING_SEED)
    args = parser.parse_args()

    dataset = load_uci_har(args.dataset_root)
    profiles = profile_models(
        dataset,
        args.models_root,
        args.profiles_root,
        timed_predictions=args.predictions,
        warmup_predictions=args.warmups,
        seed=args.seed,
    )
    for profile in profiles.values():
        print(
            f"{profile.model_id} ({profile.computational_role}): "
            f"mean={profile.local_latency_ms_mean:.6f} ms, "
            f"median={profile.local_latency_ms_median:.6f} ms, "
            f"p95={profile.local_latency_ms_p95:.6f} ms"
        )
    print(f"Saved measured profiles to {args.profiles_root}")


if __name__ == "__main__":
    main()
