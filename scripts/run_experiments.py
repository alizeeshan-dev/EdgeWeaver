"""Run or validate the reproducible EdgeWeaver Phase 7 experiment matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgeweaver.experiments import (
    CORE_SCENARIOS,
    CORE_SCHEDULERS,
    ExperimentRunner,
    build_ablation_matrix,
    build_core_matrix,
    load_experiment_config,
    select_specs,
)
from edgeweaver.schedulers import EDGEWEAVER_VARIANTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_SCHEDULER_CHOICES = tuple(dict.fromkeys((*CORE_SCHEDULERS, *EDGEWEAVER_VARIANTS)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scenario", action="append", choices=CORE_SCENARIOS, default=[])
    parser.add_argument(
        "--scheduler",
        action="append",
        choices=EXPERIMENT_SCHEDULER_CHOICES,
        default=[],
    )
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run or validate only the 80-run core matrix, excluding the 10 scoped ablations.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not regenerate tables, figures, failure cases, and findings after a full run.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.output_root is not None:
        config = config.model_copy(
            update={"paths": config.paths.model_copy(update={"output_root": args.output_root})}
        )
    runner = ExperimentRunner(config, project_root=PROJECT_ROOT)
    if args.validate_only:
        validation = runner.validate_matrix(include_ablations=not args.core_only)
        print(
            f"validated {validation.core_run_count} core and "
            f"{validation.ablation_run_count} ablation runs; "
            f"{validation.paired_trace_groups} paired trace groups"
        )
        return

    specs = build_core_matrix(config)
    if not args.core_only:
        specs = (*specs, *build_ablation_matrix(config))
    specs = select_specs(
        specs,
        scenarios=args.scenario,
        schedulers=args.scheduler,
        seeds=args.seed,
    )
    report = runner.run(specs, force=args.force, dry_run=args.dry_run)
    if report.dry_run:
        for run_id in report.run_ids:
            print(run_id)
    print(
        f"planned={report.planned_runs}, executed={report.executed_runs}, "
        f"resumed/skipped={report.skipped_runs}, dry_run={report.dry_run}"
    )
    is_complete_study = (
        not args.core_only and not args.scenario and not args.scheduler and not args.seed
    )
    if not report.dry_run and is_complete_study and not args.skip_analysis:
        from edgeweaver.research import generate_saved_research_outputs

        research = generate_saved_research_outputs(config, project_root=PROJECT_ROOT)
        print(
            f"analyzed={research.core_run_count + research.ablation_run_count}, "
            f"tables={len(research.table_paths)}, figures={len(research.figure_paths)}"
        )


if __name__ == "__main__":
    main()
