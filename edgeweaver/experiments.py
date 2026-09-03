"""Reproducible Phase 7 core and ablation experiment orchestration."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import Field, ValidationError, model_validator

from edgeweaver.config import (
    DeviceCatalog,
    NetworkCatalog,
    load_device_catalog,
    load_network_catalog,
    load_research_scenario,
    load_simulation_config,
)
from edgeweaver.domain import DomainModel, ModelProfile, SimulationRunResult
from edgeweaver.engine import SimulationEngine
from edgeweaver.events import EventLog, event_log_from_run
from edgeweaver.metrics import PerRunMetrics, calculate_run_metrics, summary_to_tidy_rows
from edgeweaver.ml.data import UCIHARDataset, load_uci_har
from edgeweaver.ml.inference import ModelPredictionService
from edgeweaver.ml.profile import load_measured_profiles
from edgeweaver.runtime_conditions import ScenarioRuntimeConditions
from edgeweaver.scenarios import (
    CoreScenarioId,
    ResearchScenarioConfig,
    generate_scenario_trace,
    scenario_config_sha256,
)
from edgeweaver.schedulers import SCHEDULER_NAMES, create_edgeweaver_variant, create_scheduler
from edgeweaver.schedulers.edgeweaver import EdgeWeaverVariant
from edgeweaver.workloads import WorkloadTrace, WorkloadTraceError, load_workload_trace

EXPERIMENT_FORMAT_VERSION: Literal["edgeweaver-experiment-v1"] = "edgeweaver-experiment-v1"
CORE_SCHEDULERS = tuple(SCHEDULER_NAMES)
CORE_SCENARIOS: tuple[CoreScenarioId, ...] = (
    "normal",
    "bursty",
    "network_slowdown",
    "device_slowdown",
)
DEFAULT_SEEDS = (1, 2, 3, 4, 5)

RunKind = Literal["core", "ablation"]


class ExperimentError(RuntimeError):
    """Base error for invalid experiment inputs or outputs."""


class ExperimentValidationError(ExperimentError):
    """Raised when a saved experiment matrix is incomplete or inconsistent."""


class AblationConfig(DomainModel):
    scheduler_name: Literal["edgeweaver_no_model_switching", "edgeweaver_no_online_update"]
    scenario_id: CoreScenarioId


class ExperimentPaths(DomainModel):
    devices_config: Path = Path("configs/devices.yaml")
    network_config: Path = Path("configs/network.yaml")
    simulation_config: Path = Path("configs/simulation.yaml")
    scenario_config_directory: Path = Path("configs/scenarios")
    profiles_root: Path = Path("artifacts/profiles")
    models_root: Path = Path("artifacts/models")
    dataset_root: Path = Path("data/raw/UCI HAR Dataset")
    output_root: Path = Path("experiments/phase7")


class CoreExperimentConfig(DomainModel):
    """The single fixed EdgeWeaver study, without duplicating scenario parameters."""

    format_version: Literal["edgeweaver-experiment-v1"] = EXPERIMENT_FORMAT_VERSION
    experiment_id: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    schedulers: tuple[str, ...]
    scenarios: tuple[CoreScenarioId, ...]
    seeds: tuple[int, ...]
    ablations: tuple[AblationConfig, ...]
    paths: ExperimentPaths = ExperimentPaths()

    @model_validator(mode="after")
    def matrix_is_the_fixed_study(self) -> Self:
        if len(self.schedulers) != 4 or set(self.schedulers) != set(CORE_SCHEDULERS):
            raise ValueError("core experiment requires exactly the four registered schedulers")
        if len(self.scenarios) != 4 or set(self.scenarios) != set(CORE_SCENARIOS):
            raise ValueError("core experiment requires exactly the four core scenarios")
        if (
            len(self.seeds) != 5
            or len(set(self.seeds)) != 5
            or any(seed < 0 for seed in self.seeds)
        ):
            raise ValueError("core experiment requires five unique non-negative seeds")
        expected_ablations = {
            ("edgeweaver_no_model_switching", "bursty"),
            ("edgeweaver_no_online_update", "device_slowdown"),
        }
        actual_ablations = {
            (ablation.scheduler_name, ablation.scenario_id) for ablation in self.ablations
        }
        if len(self.ablations) != 2 or actual_ablations != expected_ablations:
            raise ValueError("experiment requires only the two scoped EdgeWeaver ablations")
        return self


class ExperimentRunSpec(DomainModel):
    kind: RunKind
    scenario_id: CoreScenarioId
    scheduler_name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    run_id: str = Field(min_length=1)


class RunArtifactPaths(DomainModel):
    trace: Path
    raw_run: Path
    event_log: Path
    summary_json: Path
    summary_csv: Path
    completion_record: Path


class CompletedRunRecord(DomainModel):
    """Written last: its presence means every hashed run artifact was committed."""

    format_version: Literal["edgeweaver-experiment-v1"] = EXPERIMENT_FORMAT_VERSION
    experiment_id: str
    code_version: str
    run_id: str
    kind: RunKind
    scenario_id: CoreScenarioId
    scheduler_name: str
    seed: int
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_id: str
    trace_reference: str
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    devices_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    network_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    simulation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_profiles_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_artifacts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_versions: dict[str, str]
    model_profile_ids: tuple[str, ...]
    configuration_references: dict[str, str]
    request_count: int = Field(ge=1)
    simulation_duration_ms: float = Field(gt=0.0)
    executed_at_utc: datetime
    git_commit: str | None = None
    artifact_references: dict[str, str]
    artifact_sha256: dict[str, str]


class ExperimentExecutionReport(DomainModel):
    planned_runs: int = Field(ge=0)
    executed_runs: int = Field(ge=0)
    skipped_runs: int = Field(ge=0)
    dry_run: bool
    run_ids: tuple[str, ...]


class MatrixValidationReport(DomainModel):
    core_run_count: int
    ablation_run_count: int
    paired_trace_groups: int


def load_experiment_config(path: Path) -> CoreExperimentConfig:
    import yaml

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ExperimentError(f"could not read experiment config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ExperimentError(f"experiment configuration must be a YAML mapping: {path}")
    return CoreExperimentConfig.model_validate(payload)


def build_core_matrix(config: CoreExperimentConfig) -> tuple[ExperimentRunSpec, ...]:
    specs = tuple(
        ExperimentRunSpec(
            kind="core",
            scenario_id=scenario,
            scheduler_name=scheduler,
            seed=seed,
            run_id=f"core-{scenario}-{scheduler}-seed-{seed}",
        )
        for scenario in config.scenarios
        for seed in config.seeds
        for scheduler in config.schedulers
    )
    _assert_unique_specs(specs)
    if len(specs) != 80:
        raise ExperimentValidationError(f"expected 80 core runs, generated {len(specs)}")
    return specs


def build_ablation_matrix(config: CoreExperimentConfig) -> tuple[ExperimentRunSpec, ...]:
    specs = tuple(
        ExperimentRunSpec(
            kind="ablation",
            scenario_id=ablation.scenario_id,
            scheduler_name=ablation.scheduler_name,
            seed=seed,
            run_id=(f"ablation-{ablation.scenario_id}-{ablation.scheduler_name}-seed-{seed}"),
        )
        for ablation in config.ablations
        for seed in config.seeds
    )
    _assert_unique_specs(specs)
    if len(specs) != 10:
        raise ExperimentValidationError(f"expected 10 ablation runs, generated {len(specs)}")
    return specs


def select_specs(
    specs: Sequence[ExperimentRunSpec],
    *,
    scenarios: Iterable[str] = (),
    schedulers: Iterable[str] = (),
    seeds: Iterable[int] = (),
) -> tuple[ExperimentRunSpec, ...]:
    scenario_filter = set(scenarios)
    scheduler_filter = set(schedulers)
    seed_filter = set(seeds)
    return tuple(
        spec
        for spec in specs
        if (not scenario_filter or spec.scenario_id in scenario_filter)
        and (not scheduler_filter or spec.scheduler_name in scheduler_filter)
        and (not seed_filter or spec.seed in seed_filter)
    )


def _assert_unique_specs(specs: Sequence[ExperimentRunSpec]) -> None:
    identities = {(spec.kind, spec.scenario_id, spec.scheduler_name, spec.seed) for spec in specs}
    run_ids = {spec.run_id for spec in specs}
    if len(identities) != len(specs) or len(run_ids) != len(specs):
        raise ExperimentValidationError("experiment matrix contains duplicate logical runs")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path, *, suffixes: frozenset[str] | None = None) -> str:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (suffixes is None or path.suffix in suffixes)
    )
    if not files:
        raise ExperimentError(f"artifact directory contains no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_path(path)))
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: DomainModel) -> None:
    _atomic_write_text(
        path,
        json.dumps(value.model_dump(mode="json"), indent=2, allow_nan=False) + "\n",
    )


def _tidy_csv_text(summary: PerRunMetrics) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    rows = summary_to_tidy_rows(summary)
    fieldnames = list(rows[0].model_fields)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(row.model_dump(mode="json") for row in rows)
    return buffer.getvalue()


def _resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _reference(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return str(resolved)


def _git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if len(commit) == 40 else None


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "joblib": importlib.metadata.version("joblib"),
        "numpy": importlib.metadata.version("numpy"),
        "pydantic": importlib.metadata.version("pydantic"),
        "PyYAML": importlib.metadata.version("PyYAML"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "simpy": importlib.metadata.version("simpy"),
    }


class ExperimentRunner:
    """Execute independent specs while sharing one trace per scenario/seed pair."""

    def __init__(self, config: CoreExperimentConfig, *, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        paths = config.paths
        self.output_root = _resolve(self.project_root, paths.output_root)
        self._devices_path = _resolve(self.project_root, paths.devices_config)
        self._network_path = _resolve(self.project_root, paths.network_config)
        self._simulation_path = _resolve(self.project_root, paths.simulation_config)
        self._scenario_root = _resolve(self.project_root, paths.scenario_config_directory)
        self._profiles_root = _resolve(self.project_root, paths.profiles_root)
        self._models_root = _resolve(self.project_root, paths.models_root)
        self._dataset_root = _resolve(self.project_root, paths.dataset_root)
        self._config_digests = {
            "devices": _sha256_path(self._devices_path),
            "network": _sha256_path(self._network_path),
            "simulation": _sha256_path(self._simulation_path),
            "profiles": _tree_sha256(self._profiles_root),
            "models": _tree_sha256(self._models_root),
            "dataset": _tree_sha256(self._dataset_root),
            "source": _tree_sha256(
                Path(__file__).resolve().parent,
                suffixes=frozenset({".py"}),
            ),
        }
        self._runtime_versions = _runtime_versions()
        self._scenario_cache: dict[CoreScenarioId, ResearchScenarioConfig] = {}
        self._trace_cache: dict[tuple[CoreScenarioId, int], WorkloadTrace] = {}
        self._dataset: UCIHARDataset | None = None
        self._devices: DeviceCatalog | None = None
        self._network: NetworkCatalog | None = None
        self._profiles: dict[str, ModelProfile] | None = None
        self._prediction_service: ModelPredictionService | None = None
        self._git_commit = _git_commit(self.project_root)

    def paths_for(self, spec: ExperimentRunSpec) -> RunArtifactPaths:
        return RunArtifactPaths(
            trace=self._trace_path(spec.scenario_id, spec.seed),
            raw_run=self.output_root / "raw" / "runs" / f"{spec.run_id}.json",
            event_log=self.output_root / "raw" / "events" / f"{spec.run_id}.json",
            summary_json=self.output_root / "results" / "per_run" / f"{spec.run_id}.json",
            summary_csv=self.output_root / "results" / "per_run_tidy" / f"{spec.run_id}.csv",
            completion_record=(
                self.output_root / "results" / "run_records" / f"{spec.run_id}.json"
            ),
        )

    def _scenario(self, scenario_id: CoreScenarioId) -> ResearchScenarioConfig:
        if scenario_id not in self._scenario_cache:
            scenario = load_research_scenario(self._scenario_root / f"{scenario_id}.yaml")
            if scenario.scenario_id != scenario_id:
                raise ExperimentError(
                    f"scenario file for {scenario_id} identifies {scenario.scenario_id}"
                )
            self._scenario_cache[scenario_id] = scenario
        return self._scenario_cache[scenario_id]

    def _trace_path(self, scenario_id: CoreScenarioId, seed: int) -> Path:
        scenario_digest = _sha256_path(self._scenario_root / f"{scenario_id}.yaml")[:12]
        return self.output_root / "traces" / f"{scenario_id}-seed-{seed}-{scenario_digest}.json"

    def _ensure_inputs_loaded(self) -> None:
        if self._dataset is not None:
            return
        self._dataset = load_uci_har(self._dataset_root)
        self._devices = load_device_catalog(self._devices_path)
        self._network = load_network_catalog(self._network_path)
        self._profiles = load_measured_profiles(self._profiles_root)
        self._prediction_service = ModelPredictionService.from_model_directories(
            self._dataset, self._models_root, list(self._profiles)
        )

    def prepare_trace(self, scenario_id: CoreScenarioId, seed: int) -> WorkloadTrace:
        self._ensure_inputs_loaded()
        assert self._dataset is not None
        cache_key = (scenario_id, seed)
        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key]
        path = self._trace_path(scenario_id, seed)
        scenario = self._scenario(scenario_id)
        if path.is_file():
            try:
                trace = load_workload_trace(path, dataset=self._dataset)
                valid_identity = (
                    trace.seed == seed
                    and trace.generation is not None
                    and trace.generation.scenario_id == scenario_id
                    and trace.generation.scenario_config_sha256 == scenario_config_sha256(scenario)
                )
            except WorkloadTraceError:
                valid_identity = False
            if not valid_identity:
                trace = generate_scenario_trace(scenario, self._dataset, seed=seed)
                atomic_write_json(path, trace)
        else:
            trace = generate_scenario_trace(scenario, self._dataset, seed=seed)
            atomic_write_json(path, trace)
        self._trace_cache[cache_key] = trace
        return trace

    def _identity(self, spec: ExperimentRunSpec, trace: WorkloadTrace) -> str:
        scenario_sha = _sha256_path(self._scenario_root / f"{spec.scenario_id}.yaml")
        return _canonical_sha256(
            {
                "experiment_id": self.config.experiment_id,
                "code_version": self.config.code_version,
                "kind": spec.kind,
                "scenario_id": spec.scenario_id,
                "scheduler_name": spec.scheduler_name,
                "seed": spec.seed,
                "trace_sha256": _sha256_path(self._trace_path(spec.scenario_id, spec.seed)),
                "scenario_config_sha256": scenario_sha,
                "runtime_versions": self._runtime_versions,
                **self._config_digests,
            }
        )

    def is_complete(self, spec: ExperimentRunSpec, trace: WorkloadTrace) -> bool:
        paths = self.paths_for(spec)
        try:
            record = CompletedRunRecord.model_validate_json(
                paths.completion_record.read_text(encoding="utf-8"), strict=True
            )
            expected_identity = self._identity(spec, trace)
            if (
                record.run_id != spec.run_id
                or record.kind != spec.kind
                or record.scenario_id != spec.scenario_id
                or record.scheduler_name != spec.scheduler_name
                or record.seed != spec.seed
                or record.identity_sha256 != expected_identity
                or record.trace_reference != _reference(paths.trace, self.project_root)
                or record.trace_sha256 != _sha256_path(paths.trace)
            ):
                return False
            artifact_paths = {
                "raw_run": paths.raw_run,
                "event_log": paths.event_log,
                "summary_json": paths.summary_json,
                "summary_csv": paths.summary_csv,
            }
            if set(record.artifact_sha256) != set(artifact_paths):
                return False
            if any(
                not path.is_file() or _sha256_path(path) != record.artifact_sha256[name]
                for name, path in artifact_paths.items()
            ):
                return False
            run = SimulationRunResult.model_validate_json(
                paths.raw_run.read_text(encoding="utf-8"), strict=True
            )
            event_log = EventLog.model_validate_json(
                paths.event_log.read_text(encoding="utf-8"), strict=True
            )
            summary = PerRunMetrics.model_validate_json(
                paths.summary_json.read_text(encoding="utf-8"), strict=True
            )
            return (
                run.run_id == spec.run_id
                and run.scheduler_name == spec.scheduler_name
                and run.random_seed == spec.seed
                and run.request_count == len(trace.requests)
                and event_log.run_id == spec.run_id
                and event_log.scenario_id == spec.scenario_id
                and event_log.scheduler_name == spec.scheduler_name
                and event_log.seed == spec.seed
                and summary.run_id == spec.run_id
                and summary.scenario_id == spec.scenario_id
                and summary.scheduler_name == spec.scheduler_name
                and summary.seed == spec.seed
            )
        except (OSError, ValueError, ValidationError):
            return False

    def _execute(self, spec: ExperimentRunSpec, trace: WorkloadTrace) -> CompletedRunRecord:
        self._ensure_inputs_loaded()
        assert self._devices is not None
        assert self._network is not None
        assert self._profiles is not None
        assert self._prediction_service is not None
        scenario = self._scenario(spec.scenario_id)
        simulation = load_simulation_config(self._simulation_path).model_copy(
            update={"run_id": spec.run_id, "random_seed": spec.seed}
        )
        scheduler = (
            create_scheduler(spec.scheduler_name, edgeweaver_alpha=simulation.edgeweaver_ewma_alpha)
            if spec.scheduler_name in SCHEDULER_NAMES
            else create_edgeweaver_variant(
                cast(EdgeWeaverVariant, spec.scheduler_name),
                alpha=simulation.edgeweaver_ewma_alpha,
            )
        )
        engine = SimulationEngine(
            devices=self._devices,
            network=self._network,
            model_profiles=list(self._profiles.values()),
            config=simulation,
            prediction_provider=self._prediction_service,
            runtime_conditions=ScenarioRuntimeConditions(scenario, self._devices, self._network),
        )
        run = engine.run(trace, scheduler)
        summary = calculate_run_metrics(
            run,
            trace,
            scenario_id=spec.scenario_id,
            simulation_duration_ms=scenario.simulation_duration_ms,
            devices=self._devices.devices,
            model_profiles=list(self._profiles.values()),
        )
        event_log = event_log_from_run(run, scenario_id=spec.scenario_id)
        paths = self.paths_for(spec)
        atomic_write_json(paths.raw_run, run)
        atomic_write_json(paths.event_log, event_log)
        atomic_write_json(paths.summary_json, summary)
        _atomic_write_text(paths.summary_csv, _tidy_csv_text(summary))
        artifact_paths = {
            "raw_run": paths.raw_run,
            "event_log": paths.event_log,
            "summary_json": paths.summary_json,
            "summary_csv": paths.summary_csv,
        }
        record = CompletedRunRecord(
            experiment_id=self.config.experiment_id,
            code_version=self.config.code_version,
            run_id=spec.run_id,
            kind=spec.kind,
            scenario_id=spec.scenario_id,
            scheduler_name=spec.scheduler_name,
            seed=spec.seed,
            identity_sha256=self._identity(spec, trace),
            trace_id=trace.trace_id,
            trace_reference=_reference(paths.trace, self.project_root),
            trace_sha256=_sha256_path(paths.trace),
            scenario_config_sha256=_sha256_path(self._scenario_root / f"{spec.scenario_id}.yaml"),
            devices_config_sha256=self._config_digests["devices"],
            network_config_sha256=self._config_digests["network"],
            simulation_config_sha256=self._config_digests["simulation"],
            model_profiles_sha256=self._config_digests["profiles"],
            model_artifacts_sha256=self._config_digests["models"],
            dataset_sha256=self._config_digests["dataset"],
            source_sha256=self._config_digests["source"],
            runtime_versions=self._runtime_versions,
            model_profile_ids=tuple(sorted(self._profiles)),
            configuration_references={
                "devices": _reference(self._devices_path, self.project_root),
                "network": _reference(self._network_path, self.project_root),
                "simulation": _reference(self._simulation_path, self.project_root),
                "scenario": _reference(
                    self._scenario_root / f"{spec.scenario_id}.yaml", self.project_root
                ),
                "profiles": _reference(self._profiles_root, self.project_root),
                "models": _reference(self._models_root, self.project_root),
                "dataset": _reference(self._dataset_root, self.project_root),
                "source": _reference(Path(__file__).resolve().parent, self.project_root),
            },
            request_count=len(trace.requests),
            simulation_duration_ms=scenario.simulation_duration_ms,
            executed_at_utc=datetime.now(UTC),
            git_commit=self._git_commit,
            artifact_references={
                name: _reference(path, self.project_root) for name, path in artifact_paths.items()
            },
            artifact_sha256={name: _sha256_path(path) for name, path in artifact_paths.items()},
        )
        atomic_write_json(paths.completion_record, record)
        return record

    def run(
        self,
        specs: Sequence[ExperimentRunSpec],
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> ExperimentExecutionReport:
        _assert_unique_specs(specs)
        if dry_run:
            return ExperimentExecutionReport(
                planned_runs=len(specs),
                executed_runs=0,
                skipped_runs=0,
                dry_run=True,
                run_ids=tuple(spec.run_id for spec in specs),
            )
        executed = 0
        skipped = 0
        for spec in specs:
            trace = self.prepare_trace(spec.scenario_id, spec.seed)
            if not force and self.is_complete(spec, trace):
                skipped += 1
                continue
            self._execute(spec, trace)
            executed += 1
        return ExperimentExecutionReport(
            planned_runs=len(specs),
            executed_runs=executed,
            skipped_runs=skipped,
            dry_run=False,
            run_ids=tuple(spec.run_id for spec in specs),
        )

    def validate_matrix(self, *, include_ablations: bool = True) -> MatrixValidationReport:
        core_specs = build_core_matrix(self.config)
        ablation_specs = build_ablation_matrix(self.config) if include_ablations else ()
        expected_logical = {
            (spec.kind, spec.scenario_id, spec.scheduler_name, spec.seed)
            for spec in (*core_specs, *ablation_specs)
        }
        records_root = self.output_root / "results" / "run_records"
        discovered: set[tuple[RunKind, CoreScenarioId, str, int]] = set()
        for path in sorted(records_root.glob("*.json")):
            try:
                record = CompletedRunRecord.model_validate_json(
                    path.read_text(encoding="utf-8"), strict=True
                )
            except (OSError, ValidationError) as error:
                raise ExperimentValidationError(
                    f"invalid completion record encountered: {path}"
                ) from error
            if record.experiment_id != self.config.experiment_id:
                continue
            if not include_ablations and record.kind == "ablation":
                continue
            logical = (record.kind, record.scenario_id, record.scheduler_name, record.seed)
            if logical not in expected_logical:
                raise ExperimentValidationError(f"unexpected logical run: {logical}")
            if logical in discovered:
                raise ExperimentValidationError(f"duplicate logical run: {logical}")
            discovered.add(logical)

        seen_logical: set[tuple[RunKind, CoreScenarioId, str, int]] = set()
        paired_hashes: dict[tuple[CoreScenarioId, int], set[str]] = {}
        for spec in (*core_specs, *ablation_specs):
            trace = self.prepare_trace(spec.scenario_id, spec.seed)
            if not self.is_complete(spec, trace):
                raise ExperimentValidationError(f"missing or invalid completed run: {spec.run_id}")
            logical = (spec.kind, spec.scenario_id, spec.scheduler_name, spec.seed)
            if logical in seen_logical:
                raise ExperimentValidationError(f"duplicate logical run: {logical}")
            seen_logical.add(logical)
            trace_hash = _sha256_path(self.paths_for(spec).trace)
            paired_hashes.setdefault((spec.scenario_id, spec.seed), set()).add(trace_hash)
        mismatched = [key for key, hashes in paired_hashes.items() if len(hashes) != 1]
        if mismatched:
            raise ExperimentValidationError(f"paired trace mismatch for: {mismatched}")
        return MatrixValidationReport(
            core_run_count=len(core_specs),
            ablation_run_count=len(ablation_specs),
            paired_trace_groups=len(paired_hashes),
        )
