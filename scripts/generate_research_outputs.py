"""Regenerate Phase 7 tables, figures, cases, and findings from saved runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeweaver.experiments import load_experiment_config
from edgeweaver.research import generate_saved_research_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiment.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.output_root is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"output_root": args.output_root})}
        )
    report = generate_saved_research_outputs(config, project_root=PROJECT_ROOT)
    print(
        f"Regenerated {len(report.table_paths)} tables and "
        f"{len(report.figure_paths)} chart groups from "
        f"{report.core_run_count + report.ablation_run_count} saved runs"
    )


if __name__ == "__main__":
    main()
