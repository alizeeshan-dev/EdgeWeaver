"""Regenerate Phase 7 analysis artifacts from validated saved simulations."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from edgeweaver.analysis import (
    AnalysisRunRecord,
    build_analysis_tables,
    load_analysis_run_record,
    save_analysis_tables,
    validate_experiment_matrix,
)
from edgeweaver.domain import DomainModel
from edgeweaver.experiments import (
    CoreExperimentConfig,
    ExperimentRunSpec,
    build_ablation_matrix,
    build_core_matrix,
)
from edgeweaver.reporting import (
    extract_failure_cases,
    generate_findings_summary,
    generate_research_figures,
    load_reporting_runs_from_experiment,
    save_failure_cases,
    save_findings_summary,
)


class ResearchOutputReport(DomainModel):
    """Paths and counts produced by saved-result analysis."""

    core_run_count: int = Field(ge=0)
    ablation_run_count: int = Field(ge=0)
    table_paths: tuple[Path, ...]
    figure_paths: dict[str, Path]
    failure_cases_path: Path
    findings_summary_path: Path


def _load_records(
    output_root: Path,
    specs: tuple[ExperimentRunSpec, ...],
) -> list[AnalysisRunRecord]:
    return [
        load_analysis_run_record(
            output_root / "results" / "run_records" / f"{spec.run_id}.json",
            output_root / "results" / "per_run" / f"{spec.run_id}.json",
        )
        for spec in specs
    ]


def generate_saved_research_outputs(
    config: CoreExperimentConfig,
    *,
    project_root: Path,
) -> ResearchOutputReport:
    """Validate all 90 runs, then regenerate tables, figures, cases, and findings.

    This function consumes only completed artifacts. It never runs a simulation,
    so analysis and figures can be reproduced independently from execution.
    """

    output_root = (
        config.paths.output_root
        if config.paths.output_root.is_absolute()
        else project_root.resolve() / config.paths.output_root
    )
    core_specs = build_core_matrix(config)
    ablation_specs = build_ablation_matrix(config)
    core_records = _load_records(output_root, core_specs)
    ablation_records = _load_records(output_root, ablation_specs)

    validate_experiment_matrix(
        core_records,
        expected_scenarios=config.scenarios,
        expected_variants=config.schedulers,
        expected_seeds=config.seeds,
    )
    for ablation in config.ablations:
        records = [
            record
            for record in ablation_records
            if record.variant_id == ablation.scheduler_name
            and record.summary.scenario_id == ablation.scenario_id
        ]
        validate_experiment_matrix(
            records,
            expected_scenarios=(ablation.scenario_id,),
            expected_variants=(ablation.scheduler_name,),
            expected_seeds=config.seeds,
        )

    all_records = [*core_records, *ablation_records]
    table_paths = save_analysis_tables(
        output_root / "results" / "tables",
        build_analysis_tables(all_records),
    )
    summaries = [record.summary for record in all_records]
    figure_paths = generate_research_figures(summaries, output_root / "figures")

    reporting_runs = load_reporting_runs_from_experiment(
        project_root=project_root,
        run_records_directory=output_root / "results" / "run_records",
    )
    failure_cases = extract_failure_cases(reporting_runs)
    failure_cases_path = output_root / "results" / "failure_cases.json"
    findings_summary_path = output_root / "results" / "findings_summary.md"
    save_failure_cases(failure_cases_path, failure_cases)
    save_findings_summary(
        findings_summary_path,
        generate_findings_summary(summaries, failure_cases=failure_cases),
    )
    return ResearchOutputReport(
        core_run_count=len(core_records),
        ablation_run_count=len(ablation_records),
        table_paths=table_paths,
        figure_paths=figure_paths,
        failure_cases_path=failure_cases_path,
        findings_summary_path=findings_summary_path,
    )
